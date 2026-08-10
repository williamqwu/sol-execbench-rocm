import torch
import triton
import triton.language as tl


@triton.jit
def _paged_gqa(q, kc, vc, qip, kip, ids, out, lse,
               scale: tl.constexpr, B: tl.constexpr,
               BN: tl.constexpr, D: tl.constexpr):
    qidx = tl.program_id(0)
    h = tl.program_id(1)

    # indptrs are small (at most 64 entries).  Keeping this lookup on device
    # avoids a synchronizing .item() and all Python batch loops.
    b = 0
    for i in tl.static_range(1, B + 1):
        b += (qidx >= tl.load(qip + i)).to(tl.int32)
    qs = tl.load(qip + b)
    qe = tl.load(qip + b + 1)
    ks = tl.load(kip + b)
    ke = tl.load(kip + b + 1)
    nk = ke - ks
    nq = qe - qs
    upto = tl.minimum(qidx - qs + 1 + nk - nq, nk)

    d = tl.arange(0, D)
    qv = tl.load(q + (qidx * 32 + h) * D + d).to(tl.float32)
    kh = h // 8
    m = -float("inf")
    denom = 0.0
    acc = tl.zeros((D,), tl.float32)

    for start in range(0, upto, BN):
        n = start + tl.arange(0, BN)
        valid = n < upto
        pg = tl.load(ids + ks + n, mask=valid, other=0)
        kval = tl.load(kc + (pg[:, None] * 4 + kh) * D + d[None, :],
                       mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(kval * qv[None, :], axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))
        nm = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - nm)
        p = tl.exp(scores - nm)
        vval = tl.load(vc + (pg[:, None] * 4 + kh) * D + d[None, :],
                       mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vval, axis=0)
        denom = denom * alpha + tl.sum(p, axis=0)
        m = nm

    good = upto > 0
    result = tl.where(good, acc / denom, 0.0)
    tl.store(out + (qidx * 32 + h) * D + d, result)
    # Required LSE is base two, while the softmax above uses natural exp.
    tl.store(lse + qidx * 32 + h,
             tl.where(good, (m + tl.log(denom)) * 1.4426950408889634,
                      -float("inf")))


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q.shape[0]
    output = torch.empty_like(q)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)
    _paged_gqa[(total_q, 32)](
        q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, output, lse,
        scale=float(sm_scale), B=qo_indptr.shape[0] - 1, BN=16, D=128,
        num_warps=4,
    )
    return output, lse
