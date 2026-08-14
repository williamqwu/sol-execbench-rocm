import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(selected_ptr, counts_ptr, route_map_ptr, n_routes: tl.constexpr,
                  capacity: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_routes
    expert = tl.load(selected_ptr + offs, mask=mask, other=0).to(tl.int32)
    pos = tl.atomic_add(counts_ptr + expert, 1, mask=mask)
    tl.store(route_map_ptr + expert * capacity + pos, offs, mask=mask)


@triton.jit
def _make_tiles_kernel(counts_ptr, tile_count_ptr, tile_expert_ptr,
                       tile_block_ptr, n_blocks: tl.constexpr,
                       BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr):
    expert = tl.program_id(0)
    count = tl.load(counts_ptr + expert)
    used = (count + BLOCK_M - 1) // BLOCK_M
    base = tl.atomic_add(tile_count_ptr, used)
    block = tl.arange(0, BLOCKS)
    mask = block < used
    tl.store(tile_expert_ptr + base + block, expert, mask=mask)
    tl.store(tile_block_ptr + base + block, block, mask=mask)


@triton.jit
def _stage1_kernel(x_ptr, gate_ptr, up_ptr, counts_ptr, route_map_ptr,
                   tile_count_ptr, tile_expert_ptr, tile_block_ptr,
                   middle_ptr, capacity: tl.constexpr, max_tiles: tl.constexpr,
                   N_TILES: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                   TOPK: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    tile = pid // N_TILES
    nt = pid - tile * N_TILES
    real_tiles = tl.load(tile_count_ptr)
    if tile >= real_tiles:
        return
    tile_valid = tile < real_tiles
    expert = tl.load(tile_expert_ptr + tile, mask=tile_valid, other=0)
    mb = tl.load(tile_block_ptr + tile, mask=tile_valid, other=0)

    mi = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    ni = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert)
    row_mask = tile_valid & (mi < count)
    route = tl.load(route_map_ptr + expert * capacity + mi,
                    mask=row_mask, other=0)
    token = route // TOPK

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + token[:, None] * H + ki[None, :],
                    mask=row_mask[:, None], other=0.0)
        wg = tl.load(gate_ptr + expert * I * H +
                     ni[None, :] * H + ki[:, None],
                     mask=ni[None, :] < I, other=0.0)
        wu = tl.load(up_ptr + expert * I * H +
                     ni[None, :] * H + ki[:, None],
                     mask=ni[None, :] < I, other=0.0)
        gate_acc += tl.dot(x, wg)
        up_acc += tl.dot(x, wu)

    middle = (gate_acc / (1.0 + tl.exp(-gate_acc))) * up_acc
    tl.store(middle_ptr + route[:, None] * I + ni[None, :], middle,
             mask=row_mask[:, None] & (ni[None, :] < I))


@triton.jit
def _stage2_kernel(middle_ptr, down_ptr, routing_ptr, counts_ptr,
                   route_map_ptr, tile_count_ptr, tile_expert_ptr,
                   tile_block_ptr, contrib_ptr, capacity: tl.constexpr,
                   max_tiles: tl.constexpr, N_TILES: tl.constexpr,
                   H: tl.constexpr, I: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    tile = pid // N_TILES
    nt = pid - tile * N_TILES
    real_tiles = tl.load(tile_count_ptr)
    if tile >= real_tiles:
        return
    tile_valid = tile < real_tiles
    expert = tl.load(tile_expert_ptr + tile, mask=tile_valid, other=0)
    mb = tl.load(tile_block_ptr + tile, mask=tile_valid, other=0)

    mi = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    ni = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert)
    row_mask = tile_valid & (mi < count)
    route = tl.load(route_map_ptr + expert * capacity + mi,
                    mask=row_mask, other=0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, I, BLOCK_K):
        ki = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(middle_ptr + route[:, None] * I + ki[None, :],
                    mask=row_mask[:, None], other=0.0)
        w = tl.load(down_ptr + expert * H * I +
                    ni[None, :] * I + ki[:, None],
                    mask=ni[None, :] < H, other=0.0)
        acc += tl.dot(a, w)

    weight = tl.load(routing_ptr + route, mask=row_mask, other=0.0)
    weighted = acc * weight[:, None]
    tl.store(contrib_ptr + route[:, None] * H + ni[None, :], weighted,
             mask=row_mask[:, None] & (ni[None, :] < H))


@triton.jit
def _compare_swap(ea, sa, eb, sb):
    swap = ea > eb
    return (tl.where(swap, eb, ea), tl.where(swap, sb, sa),
            tl.where(swap, ea, eb), tl.where(swap, sa, sb))


@triton.jit
def _accumulate_kernel(contrib_ptr, selected_ptr, out_ptr,
                       H: tl.constexpr, TOPK: tl.constexpr,
                       BLOCK_H: tl.constexpr):
    token = tl.program_id(0)
    hb = tl.program_id(1)
    hi = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = hi < H

    # The reference visits experts in increasing expert index.  Sort the
    # token's eight slots while carrying their slot numbers.
    e0 = tl.load(selected_ptr + token * TOPK + 0)
    e1 = tl.load(selected_ptr + token * TOPK + 1)
    e2 = tl.load(selected_ptr + token * TOPK + 2)
    e3 = tl.load(selected_ptr + token * TOPK + 3)
    e4 = tl.load(selected_ptr + token * TOPK + 4)
    e5 = tl.load(selected_ptr + token * TOPK + 5)
    e6 = tl.load(selected_ptr + token * TOPK + 6)
    e7 = tl.load(selected_ptr + token * TOPK + 7)
    s0, s1, s2, s3, s4, s5, s6, s7 = 0, 1, 2, 3, 4, 5, 6, 7
    # Fully unrolled insertion sort (the selected experts are unique).
    e0, s0, e1, s1 = _compare_swap(e0, s0, e1, s1)
    e1, s1, e2, s2 = _compare_swap(e1, s1, e2, s2)
    e0, s0, e1, s1 = _compare_swap(e0, s0, e1, s1)
    e2, s2, e3, s3 = _compare_swap(e2, s2, e3, s3)
    e1, s1, e2, s2 = _compare_swap(e1, s1, e2, s2)
    e0, s0, e1, s1 = _compare_swap(e0, s0, e1, s1)
    e3, s3, e4, s4 = _compare_swap(e3, s3, e4, s4)
    e2, s2, e3, s3 = _compare_swap(e2, s2, e3, s3)
    e1, s1, e2, s2 = _compare_swap(e1, s1, e2, s2)
    e0, s0, e1, s1 = _compare_swap(e0, s0, e1, s1)
    e4, s4, e5, s5 = _compare_swap(e4, s4, e5, s5)
    e3, s3, e4, s4 = _compare_swap(e3, s3, e4, s4)
    e2, s2, e3, s3 = _compare_swap(e2, s2, e3, s3)
    e1, s1, e2, s2 = _compare_swap(e1, s1, e2, s2)
    e0, s0, e1, s1 = _compare_swap(e0, s0, e1, s1)
    e5, s5, e6, s6 = _compare_swap(e5, s5, e6, s6)
    e4, s4, e5, s5 = _compare_swap(e4, s4, e5, s5)
    e3, s3, e4, s4 = _compare_swap(e3, s3, e4, s4)
    e2, s2, e3, s3 = _compare_swap(e2, s2, e3, s3)
    e1, s1, e2, s2 = _compare_swap(e1, s1, e2, s2)
    e0, s0, e1, s1 = _compare_swap(e0, s0, e1, s1)
    e6, s6, e7, s7 = _compare_swap(e6, s6, e7, s7)
    e5, s5, e6, s6 = _compare_swap(e5, s5, e6, s6)
    e4, s4, e5, s5 = _compare_swap(e4, s4, e5, s5)
    e3, s3, e4, s4 = _compare_swap(e3, s3, e4, s4)
    e2, s2, e3, s3 = _compare_swap(e2, s2, e3, s3)
    e1, s1, e2, s2 = _compare_swap(e1, s1, e2, s2)
    e0, s0, e1, s1 = _compare_swap(e0, s0, e1, s1)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    # index_add_ writes into a bfloat16 destination after every expert.
    v = tl.load(contrib_ptr + (token * TOPK + s0) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    v = tl.load(contrib_ptr + (token * TOPK + s1) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    v = tl.load(contrib_ptr + (token * TOPK + s2) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    v = tl.load(contrib_ptr + (token * TOPK + s3) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    v = tl.load(contrib_ptr + (token * TOPK + s4) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    v = tl.load(contrib_ptr + (token * TOPK + s5) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    v = tl.load(contrib_ptr + (token * TOPK + s6) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    v = tl.load(contrib_ptr + (token * TOPK + s7) * H + hi,
                mask=hmask, other=0.0).to(tl.float32)
    acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + token * H + hi, acc, mask=hmask)


@torch.no_grad()
def run(hidden_states, routing_weights, selected_experts,
        gate_proj_weights, up_proj_weights, down_proj_weights):
    n_tokens = hidden_states.shape[0]
    H = 2048
    I = 768
    E = 128
    TOPK = 8
    routes = n_tokens * TOPK

    # One full-token capacity per expert is a semantic upper bound because a
    # token's selected experts are unique.
    count_storage = torch.zeros((E + 1,), device=hidden_states.device,
                                dtype=torch.int32)
    counts = count_storage[:E]
    route_map = torch.empty((E * n_tokens,), device=hidden_states.device,
                            dtype=torch.int32)
    _route_kernel[(triton.cdiv(routes, 256),)](
        selected_experts, counts, route_map, routes, n_tokens, BLOCK=256,
        num_warps=4)

    BM = 128
    n_blocks = triton.cdiv(n_tokens, BM)
    max_tiles = triton.cdiv(routes + E * (BM - 1), BM)
    tile_count = count_storage[E:]
    tile_expert = torch.empty((max_tiles,), device=hidden_states.device,
                              dtype=torch.int32)
    tile_block = torch.empty_like(tile_expert)
    _make_tiles_kernel[(E,)](
        counts, tile_count, tile_expert, tile_block, n_blocks, BLOCK_M=BM,
        BLOCKS=triton.next_power_of_2(n_blocks), num_warps=1)

    middle = torch.empty((routes, I), device=hidden_states.device,
                         dtype=torch.bfloat16)
    s1_bn = 128
    s1_nt = triton.cdiv(I, s1_bn)
    _stage1_kernel[(max_tiles * s1_nt,)](
        hidden_states, gate_proj_weights, up_proj_weights, counts, route_map,
        tile_count, tile_expert, tile_block, middle, n_tokens, max_tiles,
        s1_nt, H, I, TOPK, BLOCK_M=BM, BLOCK_N=s1_bn, BLOCK_K=32,
        num_warps=8, num_stages=2, matrix_instr_nonkdim=16)

    contributions = torch.empty((routes, H), device=hidden_states.device,
                                dtype=torch.bfloat16)
    s2_bn = 256
    s2_nt = triton.cdiv(H, s2_bn)
    _stage2_kernel[(max_tiles * s2_nt,)](
        middle, down_proj_weights, routing_weights, counts, route_map,
        tile_count, tile_expert, tile_block, contributions, n_tokens,
        max_tiles, s2_nt, H, I, BLOCK_M=BM, BLOCK_N=s2_bn, BLOCK_K=32,
        num_warps=8, num_stages=2, matrix_instr_nonkdim=16)

    output = torch.empty_like(hidden_states)
    _accumulate_kernel[(n_tokens, 1)](
        contributions, selected_experts, output, H, TOPK, BLOCK_H=2048,
        num_warps=8)
    return output
