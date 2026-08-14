import torch
import triton
import triton.language as tl


@triton.jit
def _rope_bwd_kernel(
    GQE, GKE, Q, K, COS, SIN,
    GQ, GK, GCOS, GSIN,
    S, H, HK,
    sqb, sqh,   # strides (elements) for [B, H,  S, D] tensors: batch, head
    skb, skh,   # strides (elements) for [B, HK, S, D] tensors: batch, head
    scb,        # stride  (elements) for [B, S, D] tensors: batch
    NUM_SB: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HD: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NUM_SB
    sb = pid - b * NUM_SB

    offs_s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    mm = (offs_s < S)[:, None]
    offs_d = tl.arange(0, HD)
    row = offs_s[:, None] * D + offs_d[None, :]

    cptr = COS + b * scb + row
    cos1 = tl.load(cptr, mask=mm, other=0.0)
    cos2 = tl.load(cptr + HD, mask=mm, other=0.0)
    sptr = SIN + b * scb + row
    sin1 = tl.load(sptr, mask=mm, other=0.0)
    sin2 = tl.load(sptr + HD, mask=mm, other=0.0)

    # accumulators for the query contribution
    qc1 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)
    qc2 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)
    qs1 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)
    qs2 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)

    qbase = b * sqb + row
    for h in range(H):
        off = h * sqh
        gp = GQE + qbase + off
        g1 = tl.load(gp, mask=mm, other=0.0)
        g2 = tl.load(gp + HD, mask=mm, other=0.0)
        xp = Q + qbase + off
        x1 = tl.load(xp, mask=mm, other=0.0)
        x2 = tl.load(xp + HD, mask=mm, other=0.0)

        op = GQ + qbase + off
        tl.store(op, g1 * cos1 + g2 * sin1, mask=mm)
        tl.store(op + HD, g2 * cos2 - g1 * sin2, mask=mm)

        qc1 += g1 * x1
        qc2 += g2 * x2
        qs1 -= g1 * x2
        qs2 += g2 * x1

    # accumulators for the key contribution (kept separate: the reference sums
    # the two head reductions independently and only then adds them)
    kc1 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)
    kc2 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)
    ks1 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)
    ks2 = tl.zeros([BLOCK_S, HD], dtype=tl.float32)

    kbase = b * skb + row
    for h in range(HK):
        off = h * skh
        gp = GKE + kbase + off
        g1 = tl.load(gp, mask=mm, other=0.0)
        g2 = tl.load(gp + HD, mask=mm, other=0.0)
        xp = K + kbase + off
        x1 = tl.load(xp, mask=mm, other=0.0)
        x2 = tl.load(xp + HD, mask=mm, other=0.0)

        op = GK + kbase + off
        tl.store(op, g1 * cos1 + g2 * sin1, mask=mm)
        tl.store(op + HD, g2 * cos2 - g1 * sin2, mask=mm)

        kc1 += g1 * x1
        kc2 += g2 * x2
        ks1 -= g1 * x2
        ks2 += g2 * x1

    ocp = GCOS + b * scb + row
    tl.store(ocp, qc1 + kc1, mask=mm)
    tl.store(ocp + HD, qc2 + kc2, mask=mm)
    osp = GSIN + b * scb + row
    tl.store(osp, qs1 + ks1, mask=mm)
    tl.store(osp + HD, qs2 + ks2, mask=mm)


def _pick_block(nrows):
    for bs in (8, 4, 2):
        if (nrows + bs - 1) // bs >= 2048:
            return bs
    return 1


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, cos, sin):
    B, H, S, D = grad_q_embed.shape
    HK = grad_k_embed.shape[1]
    HD = D // 2

    grad_q = torch.empty_like(grad_q_embed)
    grad_k = torch.empty_like(grad_k_embed)
    grad_cos = torch.empty_like(cos)
    grad_sin = torch.empty_like(sin)

    BLOCK_S = _pick_block(B * S)
    num_sb = (S + BLOCK_S - 1) // BLOCK_S

    _rope_bwd_kernel[(B * num_sb,)](
        grad_q_embed, grad_k_embed, q, k, cos, sin,
        grad_q, grad_k, grad_cos, grad_sin,
        S, H, HK,
        H * S * D, S * D,
        HK * S * D, S * D,
        S * D,
        NUM_SB=num_sb, BLOCK_S=BLOCK_S, HD=HD, D=D,
        num_warps=4, num_stages=2,
    )
    return grad_q, grad_k, grad_cos, grad_sin
