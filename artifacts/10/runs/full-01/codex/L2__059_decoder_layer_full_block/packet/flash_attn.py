import torch
import triton
import triton.language as tl


@triton.jit
def _attention_fwd(
    Q, K, V, Bias, Out,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qs: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_ks: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vs: tl.constexpr,
    stride_bb: tl.constexpr, stride_bs: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_os: tl.constexpr,
    N_CTX: tl.constexpr, SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 40
    h = bh - b * 40
    kv_h = h // 5

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for start_n in range(0, N_CTX, BLOCK_N):
        n = start_n + offs_n
        k_ptrs = K + b * stride_kb + kv_h * stride_kh + n[None, :] * stride_ks + offs_d[:, None]
        k = tl.load(k_ptrs, mask=n[None, :] < N_CTX, other=0.0)
        scores = tl.dot(q, k) * SM_SCALE
        bias_ptrs = Bias + b * stride_bb + offs_m[:, None] * stride_bs + n[None, :]
        bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < N_CTX) & (n[None, :] < N_CTX), other=-float("inf"))
        scores += bias
        scores = tl.where(n[None, :] < N_CTX, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2((m_i - m_new) * log2e)
        p = tl.exp2((scores - m_new[:, None]) * log2e)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]

        v_ptrs = V + b * stride_vb + kv_h * stride_vh + n[:, None] * stride_vs + offs_d[None, :]
        v = tl.load(v_ptrs, mask=n[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    out = acc / l_i[:, None]
    out_ptrs = Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :]
    tl.store(out_ptrs, out, mask=offs_m[:, None] < N_CTX)


def attention(q, k, v, bias, scale):
    b, _, s, _ = q.shape
    # Write B,S,H,D directly: the following output projection consumes that
    # layout, so there is no attention-output transpose/copy.
    out = torch.empty((b, s, 40, 128), device=q.device, dtype=q.dtype)
    if b != 1:
        block_m, block_n, num_warps = 128, 64, 4
    elif s <= 160:
        block_m, block_n, num_warps = 32, 32, 2
    elif s <= 384:
        block_m, block_n, num_warps = 64, 64, 2
    elif s <= 768:
        block_m, block_n, num_warps = 128, 64, 4
    elif s <= 1536:
        block_m, block_n, num_warps = 32, 32, 2
    elif s <= 4096:
        block_m, block_n, num_warps = 128, 64, 4
    else:
        block_m, block_n, num_warps = 64, 32, 2
    _attention_fwd[(triton.cdiv(s, block_m), b * 40)](
        q, k, v, bias, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        bias.stride(0), bias.stride(2),
        out.stride(0), out.stride(2), out.stride(1),
        N_CTX=s, SM_SCALE=scale,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=1,
    )
    return out
