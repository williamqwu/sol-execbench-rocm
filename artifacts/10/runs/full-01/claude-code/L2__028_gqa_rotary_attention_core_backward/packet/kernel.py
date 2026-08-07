import torch
import triton
import triton.language as tl

HQ = 64
HKV = 8
G = 8
DH = 128
D2 = 64


@triton.jit
def _rope_fwd_kernel(P, COS, SIN, S, HD,
                     BH: tl.constexpr, D2: tl.constexpr, DH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    sq = row % S
    j = tl.arange(0, D2)
    c = tl.load(COS + sq * D2 + j)[None, :]
    s = tl.load(SIN + sq * D2 + j)[None, :]

    h = blk * BH + tl.arange(0, BH)
    off = row * HD + h[:, None] * DH + tl.arange(0, DH)[None, :]
    x = tl.load(P + off).to(tl.float32)
    x0, x1 = tl.split(tl.reshape(x, (BH, D2, 2)))
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    y = tl.reshape(tl.join(y0, y1), (BH, DH))
    tl.store(P + off, y.to(P.dtype.element_ty))


@triton.jit
def _rope_bwd_kernel(SRC, DST, COS, SIN, S, H, HD, HOFF, sb, sh,
                     BH: tl.constexpr, D2: tl.constexpr, DH: tl.constexpr,
                     ROPE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    b = row // S
    sq = row % S

    h = blk * BH + tl.arange(0, BH)
    d = tl.arange(0, DH)
    src_off = b * sb + h[:, None] * sh + sq * DH + d[None, :]
    g = tl.load(SRC + src_off).to(tl.float32)

    if ROPE:
        j = tl.arange(0, D2)
        c = tl.load(COS + sq * D2 + j)[None, :]
        s = tl.load(SIN + sq * D2 + j)[None, :]
        g0, g1 = tl.split(tl.reshape(g, (BH, D2, 2)))
        y0 = g0 * c + g1 * s
        y1 = g1 * c - g0 * s
        out = tl.reshape(tl.join(y0, y1), (BH, DH))
    else:
        out = g

    dst_off = row * HD + (HOFF + h[:, None]) * DH + d[None, :]
    tl.store(DST + dst_off, out.to(DST.dtype.element_ty))


def _make_cos_sin(inv_freq, S, device):
    # Reproduce the reference's cos/sin construction bit-for-bit.
    position_ids = torch.arange(S, device=device, dtype=torch.float32).unsqueeze(0)
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    return freqs[0].cos().contiguous(), freqs[0].sin().contiguous()


@torch.no_grad()
def run(grad_output, hidden_states, q_weight, k_weight, v_weight, o_weight,
        inv_freq, scaling):
    B, S, HS = hidden_states.shape
    M = B * S
    KV = HKV * DH
    dt = hidden_states.dtype
    dev = hidden_states.device

    hs2 = hidden_states.reshape(M, HS)
    go2 = grad_output.reshape(M, HS)

    cos, sin = _make_cos_sin(inv_freq, S, dev)

    # ---------- forward recomputation: QKV projections ----------
    q = torch.mm(hs2, q_weight.t())
    k = torch.mm(hs2, k_weight.t())
    v = torch.mm(hs2, v_weight.t())

    # RoPE, fused and in-place
    _rope_fwd_kernel[(M, HQ // 8)](q, cos, sin, S, HS,
                                   BH=8, D2=D2, DH=DH, num_warps=4)
    _rope_fwd_kernel[(M, 1)](k, cos, sin, S, KV,
                             BH=8, D2=D2, DH=DH, num_warps=4)

    qh = q.view(B, S, HQ, DH).transpose(1, 2)
    kh = k.view(B, S, HKV, DH).transpose(1, 2)
    vh = v.view(B, S, HKV, DH).transpose(1, 2)

    # grad wrt attention output (independent of attention itself)
    gao_all = torch.mm(go2, o_weight).view(B, S, HQ, DH).transpose(1, 2)

    causal = torch.triu(torch.ones(S, S, device=dev, dtype=torch.bool), 1)[None, None]

    gq = torch.empty((B, HQ, S, DH), dtype=dt, device=dev)
    gk = torch.empty((B, HKV, S, DH), dtype=dt, device=dev)
    gv = torch.empty((B, HKV, S, DH), dtype=dt, device=dev)
    gow = torch.zeros((HS, HS), dtype=torch.float32, device=dev)

    # chunk over batch so the S x S score tensors stay bounded
    budget = 3 << 30
    per = HQ * S * S * 4
    nb = max(1, min(B, budget // max(per, 1)))

    for b0 in range(0, B, nb):
        b1 = min(B, b0 + nb)
        n = b1 - b0
        qr = qh[b0:b1]
        kr = kh[b0:b1]
        vs = vh[b0:b1]

        ke = kr[:, :, None].expand(n, HKV, G, S, DH).reshape(n, HQ, S, DH)
        ve = vs[:, :, None].expand(n, HKV, G, S, DH).reshape(n, HQ, S, DH)

        aw = torch.matmul(qr, ke.transpose(2, 3)) * scaling
        aw = aw.masked_fill(causal, float('-inf'))
        awf = torch.softmax(aw.float(), dim=-1)
        del aw
        awd = awf.to(dt)

        ao = torch.matmul(awd, ve)
        aoc = ao.transpose(1, 2).reshape(n * S, HS)
        del ao
        gow += torch.mm(go2[b0 * S:b1 * S].t(), aoc).float()
        del aoc

        gaoh = gao_all[b0:b1]
        gaw = torch.matmul(gaoh, ve.transpose(2, 3))
        gv[b0:b1] = torch.matmul(awd.transpose(2, 3), gaoh).view(
            n, HKV, G, S, DH).sum(2)
        del awd

        gawf = gaw.float()
        del gaw
        gas = awf * (gawf - (gawf * awf).sum(dim=-1, keepdim=True))
        del gawf, awf
        gas = (gas * scaling).to(dt)

        gq[b0:b1] = torch.matmul(gas, ke)
        gke = torch.matmul(qr.transpose(2, 3), gas).transpose(2, 3)
        del gas
        gk[b0:b1] = gke.view(n, HKV, G, S, DH).sum(2)
        del gke

    del gao_all, causal

    # ---------- RoPE backward + transpose + concat, fused ----------
    dqkv = torch.empty((M, HS + 2 * KV), dtype=dt, device=dev)
    _rope_bwd_kernel[(M, HQ // 8)](gq, dqkv, cos, sin, S, HQ, dqkv.shape[1], 0,
                                   HQ * S * DH, S * DH,
                                   BH=8, D2=D2, DH=DH, ROPE=True, num_warps=4)
    _rope_bwd_kernel[(M, 1)](gk, dqkv, cos, sin, S, HKV, dqkv.shape[1], HQ,
                             HKV * S * DH, S * DH,
                             BH=8, D2=D2, DH=DH, ROPE=True, num_warps=4)
    _rope_bwd_kernel[(M, 1)](gv, dqkv, cos, sin, S, HKV, dqkv.shape[1], HQ + HKV,
                             HKV * S * DH, S * DH,
                             BH=8, D2=D2, DH=DH, ROPE=False, num_warps=4)
    del gq, gk, gv

    dqr = dqkv[:, :HS]
    dkr = dqkv[:, HS:HS + KV]
    dvr = dqkv[:, HS + KV:]

    gh = torch.mm(dqr, q_weight)
    gh += torch.mm(dkr, k_weight)
    gh += torch.mm(dvr, v_weight)

    grad_q_weight = torch.mm(dqr.t(), hs2)
    grad_k_weight = torch.mm(dkr.t(), hs2)
    grad_v_weight = torch.mm(dvr.t(), hs2)

    return (gh.view(B, S, HS), grad_q_weight, grad_k_weight, grad_v_weight,
            gow.to(dt))
