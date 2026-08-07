import torch
import triton
import triton.language as tl
import aiter


_aiter_ready = False
_NN_SOLUTIONS = {
    256: 436722,
    1571: 436469,
    2048: 436722,
    2164: 436470,
    4096: 436613,
    8192: 436609,
}
_TN_SOLUTIONS = {
    2048: 435566,
    2164: 435359,
    4096: 435613,
    8192: 435613,
}


@triton.jit
def _query_kernel(
    grad_ptr, x_ptr, cos_ptr, sin_ptr, weight_ptr, rstd_ptr,
    dqkv_ptr, grad_cos_ptr, grad_sin_ptr, partial_weight_ptr,
    SEQ: tl.constexpr,
    H: tl.constexpr = 32,
    D: tl.constexpr = 128,
):
    token = tl.program_id(0)
    batch = token // SEQ
    seq = token - batch * SEQ
    heads = tl.arange(0, H)[:, None]
    dims = tl.arange(0, 64)[None, :]
    base = ((batch * H + heads) * SEQ + seq) * D
    offs0 = base + dims
    offs1 = offs0 + 64

    g0 = tl.load(grad_ptr + offs0).to(tl.float32)
    g1 = tl.load(grad_ptr + offs1).to(tl.float32)
    x0 = tl.load(x_ptr + offs0).to(tl.float32)
    x1 = tl.load(x_ptr + offs1).to(tl.float32)
    w0 = tl.load(weight_ptr + dims).to(tl.float32)
    w1 = tl.load(weight_ptr + dims + 64).to(tl.float32)
    r = tl.load(rstd_ptr + (batch * H + heads) * SEQ + seq).to(tl.float32)

    weight_term0 = g0 * (x0 * r)
    weight_term1 = g1 * (x1 * r)
    grad_r = tl.sum((g0 * w0) * x0, axis=1) + tl.sum((g1 * w1) * x1, axis=1)
    r3 = (r * r) * r
    scale = 1.0 / 128.0
    g0 = (((g0 * w0) * r) + grad_r[:, None] * (((-r3) * x0) * scale)).to(tl.bfloat16)
    g1 = (((g1 * w1) * r) + grad_r[:, None] * (((-r3) * x1) * scale)).to(tl.bfloat16)

    c0 = tl.load(cos_ptr + token * D + dims).to(tl.bfloat16)
    c1 = tl.load(cos_ptr + token * D + dims + 64).to(tl.bfloat16)
    s0 = tl.load(sin_ptr + token * D + dims).to(tl.bfloat16)
    s1 = tl.load(sin_ptr + token * D + dims + 64).to(tl.bfloat16)
    out0 = ((g0 * c0).to(tl.bfloat16) + (g1 * s0).to(tl.bfloat16)).to(tl.bfloat16)
    out1 = ((g1 * c1).to(tl.bfloat16) + ((-g0) * s1).to(tl.bfloat16)).to(tl.bfloat16)

    x0 = x0.to(tl.bfloat16)
    x1 = x1.to(tl.bfloat16)
    xo0 = ((x0 * c0).to(tl.bfloat16) + ((-x1) * s0).to(tl.bfloat16)).to(tl.bfloat16)
    xo1 = ((x1 * c1).to(tl.bfloat16) + (x0 * s1).to(tl.bfloat16)).to(tl.bfloat16)
    gc0 = tl.sum((g0 * xo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    gc1 = tl.sum((g1 * xo1).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    gs0 = tl.sum((g0 * (-xo1)).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    gs1 = tl.sum((g1 * xo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    pw0 = tl.sum(weight_term0, axis=0)
    pw1 = tl.sum(weight_term1, axis=0)

    out_offs = token * 6144 + heads * D + dims
    tl.store(dqkv_ptr + out_offs, out0)
    tl.store(dqkv_ptr + out_offs + 64, out1)
    flat = tl.arange(0, 64)
    tl.store(grad_cos_ptr + token * D + flat, gc0)
    tl.store(grad_cos_ptr + token * D + flat + 64, gc1)
    tl.store(grad_sin_ptr + token * D + flat, gs0)
    tl.store(grad_sin_ptr + token * D + flat + 64, gs1)
    tl.store(partial_weight_ptr + token * D + flat, pw0)
    tl.store(partial_weight_ptr + token * D + flat + 64, pw1)


@triton.jit
def _key_value_kernel(
    grad_ptr, value_ptr, x_ptr, cos_ptr, sin_ptr, weight_ptr, rstd_ptr,
    dqkv_ptr, grad_cos_ptr, grad_sin_ptr, partial_weight_ptr,
    SEQ: tl.constexpr,
    H: tl.constexpr = 8,
    D: tl.constexpr = 128,
):
    token = tl.program_id(0)
    batch = token // SEQ
    seq = token - batch * SEQ
    heads = tl.arange(0, H)[:, None]
    dims = tl.arange(0, 64)[None, :]
    base = ((batch * H + heads) * SEQ + seq) * D
    offs0 = base + dims
    offs1 = offs0 + 64
    g0 = tl.load(grad_ptr + offs0).to(tl.float32)
    g1 = tl.load(grad_ptr + offs1).to(tl.float32)
    x0 = tl.load(x_ptr + offs0).to(tl.float32)
    x1 = tl.load(x_ptr + offs1).to(tl.float32)
    w0 = tl.load(weight_ptr + dims).to(tl.float32)
    w1 = tl.load(weight_ptr + dims + 64).to(tl.float32)
    r = tl.load(rstd_ptr + (batch * H + heads) * SEQ + seq).to(tl.float32)
    weight_term0 = g0 * (x0 * r)
    weight_term1 = g1 * (x1 * r)
    grad_r = tl.sum((g0 * w0) * x0, axis=1) + tl.sum((g1 * w1) * x1, axis=1)
    r3 = (r * r) * r
    scale = 1.0 / 128.0
    g0 = (((g0 * w0) * r) + grad_r[:, None] * (((-r3) * x0) * scale)).to(tl.bfloat16)
    g1 = (((g1 * w1) * r) + grad_r[:, None] * (((-r3) * x1) * scale)).to(tl.bfloat16)

    c0 = tl.load(cos_ptr + token * D + dims).to(tl.bfloat16)
    c1 = tl.load(cos_ptr + token * D + dims + 64).to(tl.bfloat16)
    s0 = tl.load(sin_ptr + token * D + dims).to(tl.bfloat16)
    s1 = tl.load(sin_ptr + token * D + dims + 64).to(tl.bfloat16)
    out0 = ((g0 * c0).to(tl.bfloat16) + (g1 * s0).to(tl.bfloat16)).to(tl.bfloat16)
    out1 = ((g1 * c1).to(tl.bfloat16) + ((-g0) * s1).to(tl.bfloat16)).to(tl.bfloat16)
    x0 = x0.to(tl.bfloat16)
    x1 = x1.to(tl.bfloat16)
    xo0 = ((x0 * c0).to(tl.bfloat16) + ((-x1) * s0).to(tl.bfloat16)).to(tl.bfloat16)
    xo1 = ((x1 * c1).to(tl.bfloat16) + (x0 * s1).to(tl.bfloat16)).to(tl.bfloat16)
    kc0 = tl.sum((g0 * xo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    kc1 = tl.sum((g1 * xo1).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    ks0 = tl.sum((g0 * (-xo1)).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    ks1 = tl.sum((g1 * xo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    pw0 = tl.sum(weight_term0, axis=0)
    pw1 = tl.sum(weight_term1, axis=0)

    out_offs = token * 6144 + 4096 + heads * D + dims
    tl.store(dqkv_ptr + out_offs, out0)
    tl.store(dqkv_ptr + out_offs + 64, out1)
    tl.store(dqkv_ptr + out_offs + 1024, tl.load(value_ptr + offs0))
    tl.store(dqkv_ptr + out_offs + 1088, tl.load(value_ptr + offs1))

    flat = tl.arange(0, 64)
    old_c0 = tl.load(grad_cos_ptr + token * D + flat)
    old_c1 = tl.load(grad_cos_ptr + token * D + flat + 64)
    old_s0 = tl.load(grad_sin_ptr + token * D + flat)
    old_s1 = tl.load(grad_sin_ptr + token * D + flat + 64)
    tl.store(grad_cos_ptr + token * D + flat, (old_c0 + kc0).to(tl.bfloat16))
    tl.store(grad_cos_ptr + token * D + flat + 64, (old_c1 + kc1).to(tl.bfloat16))
    tl.store(grad_sin_ptr + token * D + flat, (old_s0 + ks0).to(tl.bfloat16))
    tl.store(grad_sin_ptr + token * D + flat + 64, (old_s1 + ks1).to(tl.bfloat16))
    tl.store(partial_weight_ptr + token * D + flat, pw0)
    tl.store(partial_weight_ptr + token * D + flat + 64, pw1)


@triton.jit
def _qkv_kernel(
    q_grad_ptr, q_x_ptr, k_grad_ptr, value_ptr, k_x_ptr,
    cos_ptr, sin_ptr, q_weight_ptr, k_weight_ptr, q_rstd_ptr, k_rstd_ptr,
    dqkv_ptr, grad_cos_ptr, grad_sin_ptr, q_partial_ptr, k_partial_ptr,
    SEQ: tl.constexpr, D: tl.constexpr = 128,
):
    token = tl.program_id(0)
    batch = token // SEQ
    seq = token - batch * SEQ
    dims = tl.arange(0, 64)[None, :]
    flat = tl.arange(0, 64)
    scale = 1.0 / 128.0

    # Query path.
    qh = tl.arange(0, 32)[:, None]
    qbase = ((batch * 32 + qh) * SEQ + seq) * D
    qo0 = qbase + dims
    qo1 = qo0 + 64
    qg0 = tl.load(q_grad_ptr + qo0).to(tl.float32)
    qg1 = tl.load(q_grad_ptr + qo1).to(tl.float32)
    qx0 = tl.load(q_x_ptr + qo0).to(tl.float32)
    qx1 = tl.load(q_x_ptr + qo1).to(tl.float32)
    qw0 = tl.load(q_weight_ptr + dims).to(tl.float32)
    qw1 = tl.load(q_weight_ptr + dims + 64).to(tl.float32)
    qr = tl.load(q_rstd_ptr + (batch * 32 + qh) * SEQ + seq).to(tl.float32)
    qpw0 = qg0 * (qx0 * qr)
    qpw1 = qg1 * (qx1 * qr)
    qgr = tl.sum((qg0 * qw0) * qx0, axis=1) + tl.sum((qg1 * qw1) * qx1, axis=1)
    qr3 = (qr * qr) * qr
    qg0 = (((qg0 * qw0) * qr) + qgr[:, None] * (((-qr3) * qx0) * scale)).to(tl.bfloat16)
    qg1 = (((qg1 * qw1) * qr) + qgr[:, None] * (((-qr3) * qx1) * scale)).to(tl.bfloat16)

    c0 = tl.load(cos_ptr + token * D + dims).to(tl.bfloat16)
    c1 = tl.load(cos_ptr + token * D + dims + 64).to(tl.bfloat16)
    s0 = tl.load(sin_ptr + token * D + dims).to(tl.bfloat16)
    s1 = tl.load(sin_ptr + token * D + dims + 64).to(tl.bfloat16)
    qout0 = ((qg0 * c0).to(tl.bfloat16) + (qg1 * s0).to(tl.bfloat16)).to(tl.bfloat16)
    qout1 = ((qg1 * c1).to(tl.bfloat16) + ((-qg0) * s1).to(tl.bfloat16)).to(tl.bfloat16)
    qx0 = qx0.to(tl.bfloat16)
    qx1 = qx1.to(tl.bfloat16)
    qxo0 = ((qx0 * c0).to(tl.bfloat16) + ((-qx1) * s0).to(tl.bfloat16)).to(tl.bfloat16)
    qxo1 = ((qx1 * c1).to(tl.bfloat16) + (qx0 * s1).to(tl.bfloat16)).to(tl.bfloat16)
    qgc0 = tl.sum((qg0 * qxo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    qgc1 = tl.sum((qg1 * qxo1).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    qgs0 = tl.sum((qg0 * (-qxo1)).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    qgs1 = tl.sum((qg1 * qxo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    qdst = token * 6144 + qh * D + dims
    tl.store(dqkv_ptr + qdst, qout0)
    tl.store(dqkv_ptr + qdst + 64, qout1)
    tl.store(q_partial_ptr + token * D + flat, tl.sum(qpw0, axis=0))
    tl.store(q_partial_ptr + token * D + flat + 64, tl.sum(qpw1, axis=0))

    # Key and value paths.  The four query head reductions remain live and
    # are combined here, avoiding an intermediate global round trip.
    kh = tl.arange(0, 8)[:, None]
    kbase = ((batch * 8 + kh) * SEQ + seq) * D
    ko0 = kbase + dims
    ko1 = ko0 + 64
    kg0 = tl.load(k_grad_ptr + ko0).to(tl.float32)
    kg1 = tl.load(k_grad_ptr + ko1).to(tl.float32)
    kx0 = tl.load(k_x_ptr + ko0).to(tl.float32)
    kx1 = tl.load(k_x_ptr + ko1).to(tl.float32)
    kw0 = tl.load(k_weight_ptr + dims).to(tl.float32)
    kw1 = tl.load(k_weight_ptr + dims + 64).to(tl.float32)
    kr = tl.load(k_rstd_ptr + (batch * 8 + kh) * SEQ + seq).to(tl.float32)
    kpw0 = kg0 * (kx0 * kr)
    kpw1 = kg1 * (kx1 * kr)
    kgr = tl.sum((kg0 * kw0) * kx0, axis=1) + tl.sum((kg1 * kw1) * kx1, axis=1)
    kr3 = (kr * kr) * kr
    kg0 = (((kg0 * kw0) * kr) + kgr[:, None] * (((-kr3) * kx0) * scale)).to(tl.bfloat16)
    kg1 = (((kg1 * kw1) * kr) + kgr[:, None] * (((-kr3) * kx1) * scale)).to(tl.bfloat16)
    kout0 = ((kg0 * c0).to(tl.bfloat16) + (kg1 * s0).to(tl.bfloat16)).to(tl.bfloat16)
    kout1 = ((kg1 * c1).to(tl.bfloat16) + ((-kg0) * s1).to(tl.bfloat16)).to(tl.bfloat16)
    kx0 = kx0.to(tl.bfloat16)
    kx1 = kx1.to(tl.bfloat16)
    kxo0 = ((kx0 * c0).to(tl.bfloat16) + ((-kx1) * s0).to(tl.bfloat16)).to(tl.bfloat16)
    kxo1 = ((kx1 * c1).to(tl.bfloat16) + (kx0 * s1).to(tl.bfloat16)).to(tl.bfloat16)
    kgc0 = tl.sum((kg0 * kxo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    kgc1 = tl.sum((kg1 * kxo1).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    kgs0 = tl.sum((kg0 * (-kxo1)).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    kgs1 = tl.sum((kg1 * kxo0).to(tl.bfloat16).to(tl.float32), axis=0).to(tl.bfloat16)
    kdst = token * 6144 + 4096 + kh * D + dims
    tl.store(dqkv_ptr + kdst, kout0)
    tl.store(dqkv_ptr + kdst + 64, kout1)
    tl.store(dqkv_ptr + kdst + 1024, tl.load(value_ptr + ko0))
    tl.store(dqkv_ptr + kdst + 1088, tl.load(value_ptr + ko1))
    tl.store(k_partial_ptr + token * D + flat, tl.sum(kpw0, axis=0))
    tl.store(k_partial_ptr + token * D + flat + 64, tl.sum(kpw1, axis=0))
    tl.store(grad_cos_ptr + token * D + flat, (qgc0 + kgc0).to(tl.bfloat16))
    tl.store(grad_cos_ptr + token * D + flat + 64, (qgc1 + kgc1).to(tl.bfloat16))
    tl.store(grad_sin_ptr + token * D + flat, (qgs0 + kgs0).to(tl.bfloat16))
    tl.store(grad_sin_ptr + token * D + flat + 64, (qgs1 + kgs1).to(tl.bfloat16))


@triton.jit
def _weight_reduce_kernel(q_partial, k_partial, q_out, k_out, M: tl.constexpr, BLOCK_M: tl.constexpr):
    d = tl.program_id(0)
    rows = tl.arange(0, BLOCK_M)
    mask = rows < M
    q = tl.load(q_partial + rows * 128 + d, mask=mask, other=0.0)
    k = tl.load(k_partial + rows * 128 + d, mask=mask, other=0.0)
    tl.store(q_out + d, tl.sum(q, axis=0).to(tl.bfloat16))
    tl.store(k_out + d, tl.sum(k, axis=0).to(tl.bfloat16))


@triton.jit
def _weight_block_reduce_kernel(
    q_partial, k_partial, q_block, k_block,
    M: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    dims = tl.arange(0, 128)[None, :]
    mask = rows < M
    q = tl.load(q_partial + rows * 128 + dims, mask=mask, other=0.0)
    k = tl.load(k_partial + rows * 128 + dims, mask=mask, other=0.0)
    tl.store(q_block + block * 128 + tl.arange(0, 128), tl.sum(q, axis=0))
    tl.store(k_block + block * 128 + tl.arange(0, 128), tl.sum(k, axis=0))


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * num_n
    group = pid // group_width
    first_m = group * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_width) % group_m
    pid_n = (pid % group_width) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        ka = k + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + ka[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (ka[None, :] < K), other=0.0,
        )
        b = tl.load(
            b_ptr + ka[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(ka[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@torch.no_grad()
def run(
    grad_query, grad_key, grad_value, hidden_states, cos, sin, qkv_weight,
    q_norm_weight, k_norm_weight, query_pre_norm, key_pre_norm, q_rstd,
    k_rstd, eps,
):
    b, seq, _ = hidden_states.shape
    m = b * seq
    device = hidden_states.device
    dqkv = torch.empty((m, 6144), device=device, dtype=torch.bfloat16)
    grad_cos = torch.empty((m, 128), device=device, dtype=torch.bfloat16)
    grad_sin = torch.empty_like(grad_cos)
    q_partial = torch.empty((m, 128), device=device, dtype=torch.float32)
    k_partial = torch.empty_like(q_partial)

    if m < 1024:
        _qkv_kernel[(m,)](
            grad_query, query_pre_norm, grad_key, grad_value, key_pre_norm,
            cos, sin, q_norm_weight, k_norm_weight, q_rstd, k_rstd,
            dqkv, grad_cos, grad_sin, q_partial, k_partial,
            SEQ=seq, num_warps=4,
        )
    else:
        _query_kernel[(m,)](
            grad_query, query_pre_norm, cos, sin, q_norm_weight, q_rstd,
            dqkv, grad_cos, grad_sin, q_partial, SEQ=seq, num_warps=2,
        )
        _key_value_kernel[(m,)](
            grad_key, grad_value, key_pre_norm, cos, sin, k_norm_weight, k_rstd,
            dqkv, grad_cos, grad_sin, k_partial, SEQ=seq, num_warps=1,
        )

    grad_q_weight = torch.empty((128,), device=device, dtype=torch.bfloat16)
    grad_k_weight = torch.empty_like(grad_q_weight)
    block_m = triton.next_power_of_2(m)
    if m >= 4096:
        block_rows = 64
        num_blocks = triton.cdiv(m, block_rows)
        q_blocks = torch.empty((num_blocks, 128), device=device, dtype=torch.float32)
        k_blocks = torch.empty_like(q_blocks)
        _weight_block_reduce_kernel[(num_blocks,)](
            q_partial, k_partial, q_blocks, k_blocks,
            M=m, BLOCK_ROWS=block_rows, num_warps=8,
        )
        _weight_reduce_kernel[(128,)](
            q_blocks, k_blocks, grad_q_weight, grad_k_weight,
            M=num_blocks, BLOCK_M=triton.next_power_of_2(num_blocks), num_warps=4,
        )
    else:
        _weight_reduce_kernel[(128,)](
            q_partial, k_partial, grad_q_weight, grad_k_weight,
            M=m, BLOCK_M=block_m, num_warps=8 if m < 1024 else 4,
        )

    hidden = hidden_states.reshape(m, 4096)
    nn_solution = _NN_SOLUTIONS.get(m)
    use_triton_tn = m <= 1571
    tn_solution = _TN_SOLUTIONS.get(m)
    if nn_solution is not None or tn_solution is not None:
        global _aiter_ready
        if not _aiter_ready:
            aiter.hipb_create_extension()
            _aiter_ready = True
    if nn_solution is None:
        grad_hidden = torch.mm(dqkv, qkv_weight)
    else:
        grad_hidden = aiter.hipb_mm(dqkv, qkv_weight, nn_solution)
    if use_triton_tn:
        grad_weight = torch.empty((6144, 4096), device=device, dtype=torch.bfloat16)
        bm = 128
        bn = 128
        _gemm_kernel[(triton.cdiv(6144, bm) * triton.cdiv(4096, bn),)](
            dqkv, hidden, grad_weight,
            M=6144, N=4096, K=m,
            stride_am=1, stride_ak=6144,
            stride_bk=4096, stride_bn=1,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=32, GROUP_M=2,
            num_warps=8 if m == 586 else 4, num_stages=2,
        )
    elif tn_solution is None:
        grad_weight = torch.mm(dqkv.T, hidden)
    else:
        grad_weight = aiter.hipb_mm(dqkv.T, hidden, tn_solution)
    grad_hidden = grad_hidden.reshape(b, seq, 4096)
    return grad_hidden, grad_cos.reshape(b, seq, 128), grad_sin.reshape(b, seq, 128), grad_weight, grad_q_weight, grad_k_weight
