import torch
import triton
import triton.language as tl


CHUNK_SIZE = 256
NUM_HEADS = 16
D_STATE = 256


@triton.jit
def _discretize_kernel(
    dt_ptr,
    a_log_ptr,
    cumsum_ptr,
    decay_ptr,
    BLOCK: tl.constexpr,
    N_HEADS: tl.constexpr,
):
    bc = tl.program_id(0)
    head = tl.program_id(1)
    pos = tl.arange(0, BLOCK)

    # dt is contiguous as [batch, chunks, position, head].  The two float32
    # transcendentals reproduce exp(A_log.float()) and softplus(dt.float()).
    dt_offsets = (bc * BLOCK + pos) * N_HEADS + head
    x = tl.load(dt_ptr + dt_offsets).to(tl.float32)
    a_log = tl.load(a_log_ptr + head).to(tl.float32)
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    discrete = -tl.exp(a_log) * softplus

    cumulative = tl.cumsum(discrete, axis=0)
    out_offsets = (bc * N_HEADS + head) * BLOCK + pos
    tl.store(cumsum_ptr + out_offsets, cumulative)

    last = tl.sum(tl.where(pos == BLOCK - 1, cumulative, 0.0), axis=0)
    tl.store(decay_ptr + out_offsets, tl.exp(last - cumulative))


@triton.jit
def _g_l_m_kernel(
    b_ptr,
    c_ptr,
    cumsum_ptr,
    l_ptr,
    g_ptr,
    m_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    bc = tl.program_id(1)
    head_group = tl.program_id(2)
    head_start = head_group * HEAD_BLOCK
    tiles_n = N // BLOCK_N
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    bc_base = N * K
    for k_start in range(0, K, BLOCK_K):
        c_tile = tl.load(
            c_ptr
            + bc * bc_base
            + rows[:, None] * K
            + (k_start + ks[None, :])
        )
        b_tile = tl.load(
            b_ptr
            + bc * bc_base
            + cols[:, None] * K
            + (k_start + ks[None, :])
        )
        acc += tl.dot(c_tile, tl.trans(b_tile))

    # The reference expands the single B/C group over all heads.  Its G value
    # is therefore identical in each head, while L still depends on the head.
    ij = rows[:, None] * N + cols[None, :]
    m_bc_base = bc * (N_HEADS * N * N)
    g_bc_base = bc * (N * N)
    l_bc_base = bc * (N_HEADS * N * N)
    cs_bc_base = bc * (N_HEADS * N)
    is_lower = rows[:, None] >= cols[None, :]

    # G is head-invariant for n_groups=1.  Materialize it once and expose its
    # head dimension as an expanded (stride-zero) view in run().
    tl.store(g_ptr + g_bc_base + ij, acc, mask=head_group == 0)

    for head_offset in tl.static_range(0, HEAD_BLOCK):
        head = head_start + head_offset
        cs_base = cs_bc_base + head * N
        row_cumsum = tl.load(cumsum_ptr + cs_base + rows)
        col_cumsum = tl.load(cumsum_ptr + cs_base + cols)
        exponent = tl.where(
            is_lower,
            row_cumsum[:, None] - col_cumsum[None, :],
            -float("inf"),
        )
        l_value = tl.exp(exponent)

        l_offsets = (
            l_bc_base
            + head * (N * N)
            + ij
        )
        m_offsets = m_bc_base + head * (N * N) + ij
        tl.store(l_ptr + l_offsets, l_value)
        tl.store(m_ptr + m_offsets, acc * l_value)


@triton.jit
def _g_only_kernel(
    b_ptr,
    c_ptr,
    g_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    tile = tl.program_id(0)
    bc = tl.program_id(1)
    tiles_n = N // BLOCK_N
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        c_tile = tl.load(
            c_ptr + bc * (N * K) + rows[:, None] * K + k_start + ks[None, :]
        )
        b_tile = tl.load(
            b_ptr + bc * (N * K) + cols[:, None] * K + k_start + ks[None, :]
        )
        acc += tl.dot(c_tile, tl.trans(b_tile))
    tl.store(g_ptr + bc * (N * N) + rows[:, None] * N + cols[None, :], acc)


@triton.jit
def _g_and_discretize_kernel(
    dt_ptr,
    a_log_ptr,
    b_ptr,
    c_ptr,
    cumsum_ptr,
    decay_ptr,
    g_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_HEADS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    stats_programs = NUM_CHUNKS * N_HEADS
    if pid < stats_programs:
        bc = pid // N_HEADS
        head = pid - bc * N_HEADS
        pos = tl.arange(0, N)
        x = tl.load(dt_ptr + (bc * N + pos) * N_HEADS + head).to(tl.float32)
        a_log = tl.load(a_log_ptr + head).to(tl.float32)
        softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        discrete = -tl.exp(a_log) * softplus
        cumulative = tl.cumsum(discrete, axis=0)
        out_offsets = (bc * N_HEADS + head) * N + pos
        tl.store(cumsum_ptr + out_offsets, cumulative)
        last = tl.sum(tl.where(pos == N - 1, cumulative, 0.0), axis=0)
        tl.store(decay_ptr + out_offsets, tl.exp(last - cumulative))
    else:
        tiles_n = N // BLOCK_N
        tile_count = (N // BLOCK_M) * tiles_n
        dot_pid = pid - stats_programs
        bc = dot_pid // tile_count
        tile = dot_pid - bc * tile_count
        tile_m = tile // tiles_n
        tile_n = tile - tile_m * tiles_n
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            c_tile = tl.load(
                c_ptr + bc * (N * K) + rows[:, None] * K + k_start + ks[None, :]
            )
            b_tile = tl.load(
                b_ptr + bc * (N * K) + cols[:, None] * K + k_start + ks[None, :]
            )
            acc += tl.dot(c_tile, tl.trans(b_tile))
        tl.store(g_ptr + bc * (N * N) + rows[:, None] * N + cols[None, :], acc)


@triton.jit
def _l_m_from_g_kernel(
    cumsum_ptr,
    g_ptr,
    l_ptr,
    m_ptr,
    N: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = offsets % N
    row = (offsets // N) % N
    head = (offsets // (N * N)) % N_HEADS
    bc = offsets // (N * N * N_HEADS)

    cs_base = (bc * N_HEADS + head) * N
    row_cumsum = tl.load(cumsum_ptr + cs_base + row)
    col_cumsum = tl.load(cumsum_ptr + cs_base + col)
    exponent = tl.where(row >= col, row_cumsum - col_cumsum, -float("inf"))
    l_value = tl.exp(exponent)
    g_value = tl.load(g_ptr + bc * (N * N) + row * N + col).to(tl.float32)
    tl.store(l_ptr + offsets, l_value)
    tl.store(m_ptr + offsets, g_value * l_value)


@triton.jit
def _l_m_tile_kernel(
    cumsum_ptr,
    g_ptr,
    l_ptr,
    m_ptr,
    N: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_block = tl.program_id(0)
    bc_head = tl.program_id(1)
    bc = bc_head // N_HEADS
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, N)

    cs_base = bc_head * N
    row_cumsum = tl.load(cumsum_ptr + cs_base + rows)
    col_cumsum = tl.load(cumsum_ptr + cs_base + cols)
    exponent = tl.where(
        rows[:, None] >= cols[None, :],
        row_cumsum[:, None] - col_cumsum[None, :],
        -float("inf"),
    )
    l_value = tl.exp(exponent)
    matrix_offsets = rows[:, None] * N + cols[None, :]
    g_value = tl.load(g_ptr + bc * (N * N) + matrix_offsets).to(tl.float32)
    out_offsets = bc_head * (N * N) + matrix_offsets
    tl.store(l_ptr + out_offsets, l_value)
    tl.store(m_ptr + out_offsets, g_value * l_value)


@torch.no_grad()
def run(hidden_states, dt, A_log, B, C):
    batch_size, num_chunks, _, _ = dt.shape
    chunks = batch_size * num_chunks

    # One allocation backs all five returned tensors.  L and the float32
    # results use chunk-major physical storage and are exposed with cheap views.
    full_head_elems = chunks * NUM_HEADS * CHUNK_SIZE * CHUNK_SIZE
    g_elems = chunks * CHUNK_SIZE * CHUNK_SIZE
    bf16_elems = 2 * full_head_elems + g_elems
    float_elems = 2 * chunks * NUM_HEADS * CHUNK_SIZE
    raw_storage = torch.empty(
        (bf16_elems * 2 + float_elems * 4,),
        device=dt.device,
        dtype=torch.uint8,
    )
    bf16_bytes = bf16_elems * 2
    bf16_storage = raw_storage[:bf16_bytes].view(torch.bfloat16)
    l_storage = bf16_storage[:full_head_elems].view(
        chunks, NUM_HEADS, CHUNK_SIZE, CHUNK_SIZE
    )
    m_storage = bf16_storage[full_head_elems : 2 * full_head_elems].view(
        chunks, NUM_HEADS, CHUNK_SIZE, CHUNK_SIZE
    )
    g_storage = bf16_storage[2 * full_head_elems :].view(
        chunks, CHUNK_SIZE, CHUNK_SIZE
    )
    float_storage = raw_storage[bf16_bytes:].view(torch.float32).view(
        2, chunks, NUM_HEADS, CHUNK_SIZE
    )

    if chunks <= 2:
        _discretize_kernel[(chunks, NUM_HEADS)](
            dt,
            A_log,
            float_storage[0],
            float_storage[1],
            BLOCK=CHUNK_SIZE,
            N_HEADS=NUM_HEADS,
            num_warps=1,
        )
        if chunks == 1:
            block_m, block_n, block_k, head_block, num_warps = 32, 16, 32, 4, 8
        elif chunks <= 4:
            block_m, block_n, block_k, head_block, num_warps = 32, 16, 128, 16, 8
        elif chunks <= 8:
            block_m, block_n, block_k, head_block, num_warps = 16, 128, 128, 16, 8
        else:
            block_m, block_n, block_k, head_block, num_warps = 16, 128, 128, 16, 4
        tile_count = (CHUNK_SIZE // block_m) * (CHUNK_SIZE // block_n)
        _g_l_m_kernel[(tile_count, chunks, NUM_HEADS // head_block)](
            B,
            C,
            float_storage[0],
            l_storage,
            g_storage,
            m_storage,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            N=CHUNK_SIZE,
            K=D_STATE,
            N_HEADS=NUM_HEADS,
            HEAD_BLOCK=head_block,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        if chunks == 4:
            g_block_m, g_block_n = 64, 32
        elif chunks <= 32:
            g_block_m, g_block_n = 64, 64
        else:
            g_block_m, g_block_n = 128, 128
        g_tiles = (CHUNK_SIZE // g_block_m) * (CHUNK_SIZE // g_block_n)
        combined_programs = chunks * (NUM_HEADS + g_tiles)
        _g_and_discretize_kernel[(combined_programs,)](
            dt,
            A_log,
            B,
            C,
            float_storage[0],
            float_storage[1],
            g_storage,
            BLOCK_M=g_block_m,
            BLOCK_N=g_block_n,
            BLOCK_K=64,
            N=CHUNK_SIZE,
            K=D_STATE,
            N_HEADS=NUM_HEADS,
            NUM_CHUNKS=chunks,
            num_warps=8,
            num_stages=1,
        )
        lm_rows = 64 if chunks == 16 else 16
        lm_warps = 2
        _l_m_tile_kernel[(CHUNK_SIZE // lm_rows, chunks * NUM_HEADS)](
            float_storage[0],
            g_storage,
            l_storage,
            m_storage,
            N=CHUNK_SIZE,
            N_HEADS=NUM_HEADS,
            BLOCK_ROWS=lm_rows,
            num_warps=lm_warps,
        )

    L = l_storage.view(
        batch_size, num_chunks, NUM_HEADS, CHUNK_SIZE, CHUNK_SIZE
    ).permute(0, 2, 1, 3, 4)
    G = g_storage.view(
        batch_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, 1
    ).expand(batch_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
    M = m_storage.view(
        batch_size, num_chunks, NUM_HEADS, CHUNK_SIZE, CHUNK_SIZE
    ).permute(0, 1, 3, 4, 2)
    A_cumsum = float_storage[0].view(
        batch_size, num_chunks, NUM_HEADS, CHUNK_SIZE
    ).permute(0, 2, 1, 3)
    decay_states = float_storage[1].view(
        batch_size, num_chunks, NUM_HEADS, CHUNK_SIZE
    ).permute(0, 2, 1, 3)
    return L, G, M, A_cumsum, decay_states
