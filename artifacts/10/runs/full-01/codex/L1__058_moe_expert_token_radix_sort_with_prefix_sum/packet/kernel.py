import torch
import triton
import triton.language as tl


@triton.jit
def _sort_tiles(
    x_ptr,
    tile_ptr,
    meta_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    META_PACK: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    pos = tile * BLOCK + lane
    valid = pos < N
    key = tl.load(x_ptr + pos, mask=valid, other=0)

    # Sorting (expert, original lane) makes each tile sort stable.
    composite = tl.where(valid, key * BLOCK + lane, 0x7FFFFFFF)
    ordered = tl.sort(composite, dim=0, descending=False)
    tl.store(tile_ptr + pos, ordered, mask=valid)

    # Save the count and local start for every expert in this tile.  Boundary
    # detection is exact and much cheaper here than a second comparison pass.
    meta_row = meta_ptr + tile * 256
    tl.store(meta_row + lane, 0, mask=lane < 256)
    tl.debug_barrier()

    sorted_key = ordered // BLOCK
    previous = tl.load(
        tile_ptr + pos - 1,
        mask=valid & (lane != 0),
        other=-BLOCK,
    )
    is_start = valid & ((lane == 0) | (sorted_key != previous // BLOCK))
    tl.store(meta_row + sorted_key, lane, mask=is_start)
    tl.debug_barrier()

    next_value = tl.load(
        tile_ptr + pos + 1,
        mask=valid & (pos + 1 < N) & (lane + 1 < BLOCK),
        other=0x7FFFFFFF,
    )
    is_end = valid & ((lane + 1 == BLOCK) | (sorted_key != next_value // BLOCK))
    local_start = tl.load(meta_row + sorted_key, mask=is_end, other=0)
    count = lane + 1 - local_start
    tl.store(meta_row + sorted_key, count * META_PACK + local_start, mask=is_end)


@triton.jit
def _prefix_tiles(
    meta_ptr,
    offsets_ptr,
    N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    META_PACK: tl.constexpr,
):
    expert = tl.arange(0, 256)
    total = tl.zeros((256,), tl.int32)
    for tile in range(0, NUM_TILES):
        packed = tl.load(meta_ptr + tile * 256 + expert)
        total += packed // META_PACK

    expert_end = tl.cumsum(total, axis=0)
    expert_base = expert_end - total
    tl.store(offsets_ptr + expert, expert_base)
    tl.store(offsets_ptr + 256, N)

    before = tl.zeros((256,), tl.int32)
    for tile in range(0, NUM_TILES):
        packed = tl.load(meta_ptr + tile * 256 + expert)
        count = packed // META_PACK
        local_start = packed % META_PACK
        global_start = expert_base + before
        tl.store(
            meta_ptr + tile * 256 + expert,
            global_start * META_PACK + local_start,
        )
        before += count


@triton.jit
def _scatter_tiles(
    tile_ptr,
    meta_ptr,
    order_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    META_PACK: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    pos = tile * BLOCK + lane
    valid = pos < N
    composite = tl.load(tile_ptr + pos, mask=valid, other=0)
    expert = composite // BLOCK
    original_lane = composite % BLOCK
    packed = tl.load(meta_ptr + tile * 256 + expert, mask=valid, other=0)
    global_start = packed // META_PACK
    local_start = packed % META_PACK
    destination = global_start + lane - local_start
    tl.store(order_ptr + destination, tile * BLOCK + original_lane, mask=valid)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    n = topk_idx.numel()
    # 34 half-size tiles hit a backend scheduling cliff; the larger tile is
    # faster in that narrow band as well as for the genuinely large inputs.
    if n < 32768 and not (17000 <= n < 17600):
        block, num_warps = 512, 8
    else:
        block, num_warps = 1024, 16
    num_tiles = triton.cdiv(n, block)
    meta_pack = block

    tiles = torch.empty((n,), dtype=torch.int32, device=topk_idx.device)
    meta = torch.empty((num_tiles * 256,), dtype=torch.int32, device=topk_idx.device)
    order = torch.empty((n,), dtype=torch.int32, device=topk_idx.device)
    offsets = torch.empty((257,), dtype=torch.int32, device=topk_idx.device)

    _sort_tiles[(num_tiles,)](
        topk_idx,
        tiles,
        meta,
        N=n,
        BLOCK=block,
        META_PACK=meta_pack,
        num_warps=num_warps,
    )
    _prefix_tiles[(1,)](
        meta,
        offsets,
        N=n,
        NUM_TILES=num_tiles,
        META_PACK=meta_pack,
        num_warps=4,
    )
    _scatter_tiles[(num_tiles,)](
        tiles,
        meta,
        order,
        N=n,
        BLOCK=block,
        META_PACK=meta_pack,
        num_warps=num_warps,
    )
    return order, offsets
