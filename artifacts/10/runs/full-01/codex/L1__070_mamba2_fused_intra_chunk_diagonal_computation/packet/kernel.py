import torch
import triton
import triton.language as tl


CHUNK = 128
HEADS = 32
GROUPS = 8
WIDTH = 128


@triton.jit
def _prefix_kernel(a_ptr, p_ptr, num_chunks: tl.constexpr):
    q = tl.program_id(0)
    h = q % 32
    z = q // 32
    b = z // num_chunks
    c = z - b * num_chunks
    k = tl.arange(0, 128)
    off = ((b * 32 + h) * num_chunks + c) * 128 + k
    a = tl.load(a_ptr + off).to(tl.float32)
    p = tl.cumsum(a, axis=0)
    tl.store(p_ptr + off, p)


@triton.jit
def _gram_kernel(c_ptr, b_ptr, g_ptr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr,
                 PACKED_TRIANGLE: tl.constexpr):
    tile = tl.program_id(0)
    q = tl.program_id(1)
    z = q // 8
    group = q - z * 8

    if PACKED_TRIANGLE:
        # Packed mapping for the three lower tiles of a 2x2 tiling:
        # (0,0), (1,0), (1,1).
        tm = (tile + 1) // 2
        tn = tile // 2
    else:
        tiles_n: tl.constexpr = tl.cdiv(128, BLOCK_N)
        tm = tile // tiles_n
        tn = tile - tm * tiles_n
        # The consumer never reads strictly upper-triangular tiles.
        if tn > tm:
            return

    mi = tm * BLOCK_M + tl.arange(0, BLOCK_M)
    nj = tn * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    base = z * 128 * 8 * 128 + group * 128
    for k0 in range(0, 128, BLOCK_K):
        ck = k0 + kk
        cv = tl.load(c_ptr + base + mi[:, None] * 8 * 128 + ck[None, :])
        bv = tl.load(b_ptr + base + nj[None, :] * 8 * 128 + ck[:, None])
        acc = tl.dot(cv, bv, acc)

    goff = (((z * 8 + group) * 128 + mi[:, None]) * 128
            + nj[None, :])
    tl.store(g_ptr + goff, acc)


@triton.jit
def _output_kernel(x_ptr, a_or_p_ptr, g_ptr, y_ptr,
                   num_chunks: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                   BLOCK_K: tl.constexpr,
                   FUSED_PREFIX: tl.constexpr):
    tile = tl.program_id(0)
    tiles_d: tl.constexpr = tl.cdiv(128, BLOCK_D)
    tm = tile // tiles_d
    td = tile - tm * tiles_d
    q = tl.program_id(1)
    h = q % 32
    z = q // 32
    group = h // 4
    b = z // num_chunks
    c = z - b * num_chunks

    ii = tm * BLOCK_M + tl.arange(0, BLOCK_M)
    dd = td * BLOCK_D + tl.arange(0, BLOCK_D)
    kk = tl.arange(0, BLOCK_K)

    abase = ((b * 32 + h) * num_chunks + c) * 128
    if FUSED_PREFIX:
        aa = tl.load(a_or_p_ptr + abase + tl.arange(0, 128)).to(tl.float32)
        pp = tl.cumsum(aa, axis=0)
        pi = tl.gather(pp, ii, axis=0)
    else:
        pi = tl.load(a_or_p_ptr + abase + ii)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Only source blocks at or before this output block can contribute.
    for tk in range(0, tm + 1):
        jj = tk * BLOCK_K + kk
        if FUSED_PREFIX:
            pj = tl.gather(pp, jj, axis=0)
        else:
            pj = tl.load(a_or_p_ptr + abase + jj)

        goff = (((z * 8 + group) * 128 + ii[:, None]) * 128
                + jj[None, :])
        weights = tl.load(g_ptr + goff)
        causal = ii[:, None] >= jj[None, :]
        weights = tl.where(causal, weights * tl.exp(pi[:, None] - pj[None, :]), 0.0)

        xoff = (((z * 128 + jj[:, None]) * 32 + h) * 128
                + dd[None, :])
        xv = tl.load(x_ptr + xoff)
        weights_hi = weights.to(tl.bfloat16)
        weights_lo = (weights - weights_hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(weights_hi, xv, acc)
        acc = tl.dot(weights_lo, xv, acc)

    yoff = (((z * 128 + ii[:, None]) * 32 + h) * 128
            + dd[None, :])
    tl.store(y_ptr + yoff, acc)


@triton.jit
def _output_pair_kernel(x_ptr, a_ptr, g_ptr, y_ptr,
                        num_chunks: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    tile = tl.program_id(0)
    tiles_d: tl.constexpr = tl.cdiv(128, BLOCK_D)
    tm = tile // tiles_d
    td = tile - tm * tiles_d
    q = tl.program_id(1)
    pair = q % 16
    z = q // 16
    h0 = pair * 2
    h1 = h0 + 1
    group = h0 // 4
    b = z // num_chunks
    c = z - b * num_chunks

    ii = tm * BLOCK_M + tl.arange(0, BLOCK_M)
    dd = td * BLOCK_D + tl.arange(0, BLOCK_D)
    kk = tl.arange(0, BLOCK_K)
    pos = tl.arange(0, 128)

    abase0 = ((b * 32 + h0) * num_chunks + c) * 128
    abase1 = ((b * 32 + h1) * num_chunks + c) * 128
    pp0 = tl.cumsum(tl.load(a_ptr + abase0 + pos).to(tl.float32), axis=0)
    pp1 = tl.cumsum(tl.load(a_ptr + abase1 + pos).to(tl.float32), axis=0)
    pi0 = tl.gather(pp0, ii, axis=0)
    pi1 = tl.gather(pp1, ii, axis=0)
    acc0 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for tk in range(0, tm + 1):
        jj = tk * BLOCK_K + kk
        pj0 = tl.gather(pp0, jj, axis=0)
        pj1 = tl.gather(pp1, jj, axis=0)
        goff = (((z * 8 + group) * 128 + ii[:, None]) * 128
                + jj[None, :])
        gram = tl.load(g_ptr + goff).to(tl.float32)
        causal = ii[:, None] >= jj[None, :]
        w0 = tl.where(causal, gram * tl.exp(pi0[:, None] - pj0[None, :]), 0.0)
        w1 = tl.where(causal, gram * tl.exp(pi1[:, None] - pj1[None, :]), 0.0)

        xoff0 = (((z * 128 + jj[:, None]) * 32 + h0) * 128
                 + dd[None, :])
        xoff1 = (((z * 128 + jj[:, None]) * 32 + h1) * 128
                 + dd[None, :])
        xv0 = tl.load(x_ptr + xoff0)
        xv1 = tl.load(x_ptr + xoff1)
        w0_hi = w0.to(tl.bfloat16)
        w1_hi = w1.to(tl.bfloat16)
        w0_lo = (w0 - w0_hi.to(tl.float32)).to(tl.bfloat16)
        w1_lo = (w1 - w1_hi.to(tl.float32)).to(tl.bfloat16)
        acc0 = tl.dot(w0_hi, xv0, acc0)
        acc0 = tl.dot(w0_lo, xv0, acc0)
        acc1 = tl.dot(w1_hi, xv1, acc1)
        acc1 = tl.dot(w1_lo, xv1, acc1)

    yoff0 = (((z * 128 + ii[:, None]) * 32 + h0) * 128
             + dd[None, :])
    yoff1 = (((z * 128 + ii[:, None]) * 32 + h1) * 128
             + dd[None, :])
    tl.store(y_ptr + yoff0, acc0)
    tl.store(y_ptr + yoff1, acc1)


@triton.jit
def _fully_fused_kernel(x_ptr, a_ptr, b_ptr, c_ptr, y_ptr,
                        num_chunks: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                        BLOCK_K: tl.constexpr,
                        STATE_BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    tiles_d: tl.constexpr = tl.cdiv(128, BLOCK_D)
    tm = tile // tiles_d
    td = tile - tm * tiles_d
    q = tl.program_id(1)
    h = q % 32
    z = q // 32
    group = h // 4
    bbatch = z // num_chunks
    chunk = z - bbatch * num_chunks

    ii = tm * BLOCK_M + tl.arange(0, BLOCK_M)
    dd = td * BLOCK_D + tl.arange(0, BLOCK_D)
    jj_local = tl.arange(0, BLOCK_K)
    state = tl.arange(0, STATE_BLOCK)

    abase = ((bbatch * 32 + h) * num_chunks + chunk) * 128
    pp = tl.cumsum(
        tl.load(a_ptr + abase + tl.arange(0, 128)).to(tl.float32),
        axis=0)
    pi = tl.gather(pp, ii, axis=0)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    bcbase = z * 128 * 8 * 128 + group * 128

    for tk in range(0, tm + 1):
        jj = tk * BLOCK_K + jj_local
        gram_acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
        for sk in range(0, 128, STATE_BLOCK):
            nn = sk + state
            cv = tl.load(c_ptr + bcbase
                         + ii[:, None] * 8 * 128 + nn[None, :])
            bv = tl.load(b_ptr + bcbase
                         + jj[None, :] * 8 * 128 + nn[:, None])
            gram_acc = tl.dot(cv, bv, gram_acc)

        # Match the compact intermediate used by the two-kernel path.
        gram = gram_acc.to(tl.float16).to(tl.float32)
        pj = tl.gather(pp, jj, axis=0)
        causal = ii[:, None] >= jj[None, :]
        weights = tl.where(causal,
                           gram * tl.exp(pi[:, None] - pj[None, :]), 0.0)
        xoff = (((z * 128 + jj[:, None]) * 32 + h) * 128
                + dd[None, :])
        xv = tl.load(x_ptr + xoff)
        weights_hi = weights.to(tl.bfloat16)
        weights_lo = (weights - weights_hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(weights_hi, xv, acc)
        acc = tl.dot(weights_lo, xv, acc)

    yoff = (((z * 128 + ii[:, None]) * 32 + h) * 128
            + dd[None, :])
    tl.store(y_ptr + yoff, acc)


@torch.no_grad()
def run(hidden_states, A_cumsum, B, C):
    batch, num_chunks = hidden_states.shape[:2]
    z = batch * num_chunks
    gram = torch.empty((z, GROUPS, CHUNK, CHUNK),
                       device=hidden_states.device, dtype=torch.float16)
    out = torch.empty_like(hidden_states)

    bm = 64
    bn = 64
    gram_warps = 2 if z >= 32 else 4
    _gram_kernel[(3, z * GROUPS)](
        C, B, gram, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128,
        num_warps=gram_warps, PACKED_TRIANGLE=True)

    bm = 64
    if z <= 4:
        bd = 64
    else:
        bd = 128
    _output_kernel[((CHUNK // bm) * (CHUNK // bd), z * HEADS)](
        hidden_states, A_cumsum, gram, out, num_chunks=num_chunks,
        BLOCK_M=bm, BLOCK_D=bd, BLOCK_K=bm, num_warps=4,
        num_stages=1, FUSED_PREFIX=True)
    return out
