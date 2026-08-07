import torch
import triton
import triton.language as tl


@triton.jit
def _qk_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)

    tiles_n = tl.cdiv(seq_len, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n

    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, 256)

    q_offsets = batch_head * seq_len * 256 + rows[:, None] * 256 + depth[None, :]
    k_offsets = batch * seq_len * 256 + cols[:, None] * 256 + depth[None, :]
    q = tl.load(query + q_offsets, mask=rows[:, None] < seq_len, other=0.0)
    k = tl.load(key + k_offsets, mask=cols[:, None] < seq_len, other=0.0)

    scores = tl.dot(q, tl.trans(k))
    scores *= scaling

    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(
        output + out_offsets,
        scores,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_tiled_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_n = tl.cdiv(seq_len, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n

    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 256, BLOCK_K):
        q_offsets = (
            batch_head * seq_len * 256
            + rows[:, None] * 256
            + k_start
            + depth[None, :]
        )
        k_offsets = (
            batch * seq_len * 256
            + cols[:, None] * 256
            + k_start
            + depth[None, :]
        )
        q = tl.load(query + q_offsets, mask=rows[:, None] < seq_len, other=0.0)
        k = tl.load(key + k_offsets, mask=cols[:, None] < seq_len, other=0.0)
        accumulator += tl.dot(q, tl.trans(k))

    accumulator *= scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(
        output + out_offsets,
        accumulator,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_persistent_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    TOTAL_TILES: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    tiles_m: tl.constexpr = tl.cdiv(seq_len, BLOCK_M)
    tiles_n: tl.constexpr = tl.cdiv(seq_len, BLOCK_N)
    tiles_per_head: tl.constexpr = tiles_m * tiles_n
    depth = tl.arange(0, BLOCK_K)

    for tile_id in range(program, TOTAL_TILES, NUM_PROGRAMS):
        batch_head = tile_id // tiles_per_head
        head_tile = tile_id - batch_head * tiles_per_head
        tile_m = head_tile // tiles_n
        tile_n = head_tile - tile_m * tiles_n
        batch = batch_head // 4
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, 256, BLOCK_K):
            q_offsets = (
                batch_head * seq_len * 256
                + rows[:, None] * 256
                + k_start
                + depth[None, :]
            )
            k_offsets = (
                batch * seq_len * 256
                + cols[:, None] * 256
                + k_start
                + depth[None, :]
            )
            q = tl.load(query + q_offsets, mask=rows[:, None] < seq_len, other=0.0)
            k = tl.load(key + k_offsets, mask=cols[:, None] < seq_len, other=0.0)
            accumulator += tl.dot(q, tl.trans(k))

        accumulator *= scaling
        out_offsets = (
            batch_head * seq_len * seq_len
            + rows[:, None] * seq_len
            + cols[None, :]
        )
        tl.store(
            output + out_offsets,
            accumulator,
            mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
        )


@triton.jit
def _qk_direct_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_n = tl.cdiv(seq_len, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 256, BLOCK_K):
        q_offsets = (
            batch_head * seq_len * 256
            + rows[:, None] * 256
            + k_start
            + depth[None, :]
        )
        k_offsets = (
            batch * seq_len * 256
            + cols[None, :] * 256
            + k_start
            + depth[:, None]
        )
        q = tl.load(query + q_offsets, mask=rows[:, None] < seq_len, other=0.0)
        k = tl.load(key + k_offsets, mask=cols[None, :] < seq_len, other=0.0)
        accumulator += tl.dot(q, k)

    accumulator *= scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(
        output + out_offsets,
        accumulator,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_block_ptr_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_n = tl.cdiv(seq_len, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    batch = batch_head // 4

    q_block = tl.make_block_ptr(
        base=query + batch_head * seq_len * 256,
        shape=(seq_len, 256),
        strides=(256, 1),
        offsets=(tile_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    k_block = tl.make_block_ptr(
        base=key + batch * seq_len * 256,
        shape=(256, seq_len),
        strides=(1, 256),
        offsets=(0, tile_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, 256, BLOCK_K):
        q = tl.load(q_block, boundary_check=(0,), padding_option="zero")
        k = tl.load(k_block, boundary_check=(1,), padding_option="zero")
        accumulator += tl.dot(q, k)
        q_block = tl.advance(q_block, (0, BLOCK_K))
        k_block = tl.advance(k_block, (BLOCK_K, 0))

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator *= scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(
        output + out_offsets,
        accumulator,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_3d_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    tile_m = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 256, BLOCK_K):
        q_offsets = (
            batch_head * seq_len * 256
            + rows[:, None] * 256
            + k_start
            + depth[None, :]
        )
        k_offsets = (
            batch * seq_len * 256
            + cols[:, None] * 256
            + k_start
            + depth[None, :]
        )
        q = tl.load(query + q_offsets, mask=rows[:, None] < seq_len, other=0.0)
        k = tl.load(key + k_offsets, mask=cols[:, None] < seq_len, other=0.0)
        accumulator += tl.dot(q, tl.trans(k))

    accumulator *= scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(
        output + out_offsets,
        accumulator,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_grouped_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_m: tl.constexpr = tl.cdiv(seq_len, BLOCK_M)
    tiles_n: tl.constexpr = tl.cdiv(seq_len, BLOCK_N)
    tiles_per_group: tl.constexpr = GROUP_M * tiles_n
    group = tile // tiles_per_group
    first_m = group * GROUP_M
    group_m = tl.minimum(tiles_m - first_m, GROUP_M)
    in_group = tile - group * tiles_per_group
    tile_m = first_m + (in_group % group_m)
    tile_n = in_group // group_m

    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 256, BLOCK_K):
        q_offsets = (
            batch_head * seq_len * 256
            + rows[:, None] * 256
            + k_start
            + depth[None, :]
        )
        k_offsets = (
            batch * seq_len * 256
            + cols[:, None] * 256
            + k_start
            + depth[None, :]
        )
        q = tl.load(query + q_offsets, mask=rows[:, None] < seq_len, other=0.0)
        k = tl.load(key + k_offsets, mask=cols[:, None] < seq_len, other=0.0)
        accumulator += tl.dot(q, tl.trans(k))

    accumulator *= scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(
        output + out_offsets,
        accumulator,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_static_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    UNROLL: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_n = tl.cdiv(seq_len, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(
        0, 256, BLOCK_K, loop_unroll_factor=UNROLL
    ):
        q_offsets = (
            batch_head * seq_len * 256
            + rows[:, None] * 256
            + k_start
            + depth[None, :]
        )
        k_offsets = (
            batch * seq_len * 256
            + cols[:, None] * 256
            + k_start
            + depth[None, :]
        )
        q = tl.load(query + q_offsets, mask=rows[:, None] < seq_len, other=0.0)
        k = tl.load(key + k_offsets, mask=cols[:, None] < seq_len, other=0.0)
        accumulator += tl.dot(q, tl.trans(k))

    accumulator *= scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(
        output + out_offsets,
        accumulator,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_nomask_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_n: tl.constexpr = seq_len // BLOCK_N
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 256, BLOCK_K):
        q_offsets = (
            batch_head * seq_len * 256
            + rows[:, None] * 256
            + k_start
            + depth[None, :]
        )
        k_offsets = (
            batch * seq_len * 256
            + cols[:, None] * 256
            + k_start
            + depth[None, :]
        )
        q = tl.load(query + q_offsets)
        k = tl.load(key + k_offsets)
        accumulator += tl.dot(q, tl.trans(k))

    accumulator *= scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(output + out_offsets, accumulator)


@triton.jit
def _qk_full_nomask_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_n: tl.constexpr = seq_len // BLOCK_N
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, 256)
    q = tl.load(
        query
        + batch_head * seq_len * 256
        + rows[:, None] * 256
        + depth[None, :]
    )
    k = tl.load(
        key
        + batch * seq_len * 256
        + cols[:, None] * 256
        + depth[None, :]
    )
    scores = tl.dot(q, tl.trans(k)) * scaling
    out_offsets = batch_head * seq_len * seq_len + rows[:, None] * seq_len + cols[None, :]
    tl.store(output + out_offsets, scores)


@triton.jit
def _qk_padded_full_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    out_pitch: tl.constexpr,
    scaling: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    tiles_n = tl.cdiv(seq_len, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, 256)
    q = tl.load(
        query
        + batch_head * seq_len * 256
        + rows[:, None] * 256
        + depth[None, :],
        mask=rows[:, None] < seq_len,
        other=0.0,
    )
    k = tl.load(
        key
        + batch * seq_len * 256
        + cols[:, None] * 256
        + depth[None, :],
        mask=cols[:, None] < seq_len,
        other=0.0,
    )
    scores = tl.dot(q, tl.trans(k)) * scaling
    out_offsets = (
        batch_head * seq_len * out_pitch
        + rows[:, None] * out_pitch
        + cols[None, :]
    )
    tl.store(
        output + out_offsets,
        scores,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


@triton.jit
def _qk_padded_3d_full_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    out_pitch: tl.constexpr,
    scaling: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_n = tl.program_id(0)
    tile_m = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // 4
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    depth = tl.arange(0, 256)
    q = tl.load(
        query
        + batch_head * seq_len * 256
        + rows[:, None] * 256
        + depth[None, :],
        mask=rows[:, None] < seq_len,
        other=0.0,
    )
    k = tl.load(
        key
        + batch * seq_len * 256
        + cols[:, None] * 256
        + depth[None, :],
        mask=cols[:, None] < seq_len,
        other=0.0,
    )
    scores = tl.dot(q, tl.trans(k)) * scaling
    out_offsets = (
        batch_head * seq_len * out_pitch
        + rows[:, None] * out_pitch
        + cols[None, :]
    )
    tl.store(
        output + out_offsets,
        scores,
        mask=(rows[:, None] < seq_len) & (cols[None, :] < seq_len),
    )


def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, _, seq_len, _ = key.shape
    out_pitch = 1 << (seq_len - 1).bit_length()
    if out_pitch == seq_len:
        output = torch.empty(
            (batch, 4, seq_len, seq_len), device=query.device, dtype=query.dtype
        )
    else:
        output = torch.empty_strided(
            (batch, 4, seq_len, seq_len),
            (4 * seq_len * out_pitch, seq_len * out_pitch, out_pitch, 1),
            device=query.device,
            dtype=query.dtype,
        )

    # The short and irregular cases favor a single unrolled reduction.  Larger
    # matrices benefit from a pipelined K loop and a wider output tile.
    if seq_len <= 384 and batch < 4:
        block_m, block_n, warps = 32, 32, 4
        grid = (
            triton.cdiv(seq_len, block_m) * triton.cdiv(seq_len, block_n),
            batch * 4,
        )
        if out_pitch == seq_len:
            _qk_kernel[grid](
                query,
                key,
                output,
                seq_len,
                scaling,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=warps,
            )
        else:
            _qk_padded_full_kernel[grid](
                query,
                key,
                output,
                seq_len,
                out_pitch,
                scaling,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=warps,
            )
    elif seq_len <= 384 and batch < 16:
        block_m, block_n, warps = 64, 64, 4
        grid = (
            triton.cdiv(seq_len, block_m) * triton.cdiv(seq_len, block_n),
            batch * 4,
        )
        if out_pitch == seq_len:
            _qk_kernel[grid](
                query,
                key,
                output,
                seq_len,
                scaling,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=warps,
            )
        else:
            _qk_padded_full_kernel[grid](
                query,
                key,
                output,
                seq_len,
                out_pitch,
                scaling,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=warps,
            )
    elif (384 < seq_len < 896) or (1280 < seq_len < 2048):
        block_m, block_n, warps = 64, 64, 4
        grid = (
            triton.cdiv(seq_len, block_m) * triton.cdiv(seq_len, block_n),
            batch * 4,
        )
        if out_pitch == seq_len:
            _qk_kernel[grid](
                query,
                key,
                output,
                seq_len,
                scaling,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=warps,
            )
        else:
            _qk_padded_full_kernel[grid](
                query,
                key,
                output,
                seq_len,
                out_pitch,
                scaling,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=warps,
            )
    else:
        # More independent matrices can profitably use a 256-wide N tile.
        if (seq_len <= 1280 and batch >= 2) or (
            seq_len <= 384 and batch >= 16
        ):
            block_m, block_n = 128, 256
        else:
            block_m, block_n = 128, 128

        if block_n == 256:
            if seq_len <= 384 or batch <= 2:
                block_k, warps, stages = 64, 8, 2
            else:
                block_k, warps, stages = 32, 8, 3
        elif seq_len <= 1280:
            block_k, warps, stages = 64, 8, 2
        elif seq_len < 3072:
            if batch == 1:
                block_k, warps, stages = 64, 8, 2
            else:
                block_k, warps, stages = 64, 4, 1
        elif seq_len < 6144:
            block_k, warps, stages = 32, 4, 2
        else:
            block_k, warps, stages = 64, 4, 1
        grid = (
            triton.cdiv(seq_len, block_m) * triton.cdiv(seq_len, block_n),
            batch * 4,
        )
        _qk_tiled_kernel[grid](
            query,
            key,
            output,
            seq_len,
            scaling,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
    return output
