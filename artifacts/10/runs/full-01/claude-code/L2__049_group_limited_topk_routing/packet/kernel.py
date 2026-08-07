"""Group-limited top-k expert routing, fused for MI355X (gfx950).

Pipeline:
  1. fp32 GEMM  hidden_states @ weight.T   -> logits          (torch/rocBLAS)
  2. torch.sigmoid                          -> scores          (bitwise vs reference)
  3. one fused Triton kernel does everything else in registers:
       bias add, per-group top-2 sum, top-4 group select, mask,
       top-8 expert select, normalize, scale.

Numerics notes (the reference's order of operations is the spec):
  * The fp32 GEMM and torch.sigmoid are kept as-is because the tolerance is
    atol=0/rtol=0 (99% match). A bf16-MFMA GEMM keeps the *indices* exact but
    perturbs the weights by ~30%, and every Triton sigmoid variant tried differs
    from torch's by a few ULP. Both are therefore left on the torch path.
  * torch.topk(..., sorted=False) on ROCm returns elements in stable descending
    order (ties broken by lowest index). This was verified over many shapes and
    is reproduced exactly by packing (value, index) into a single monotone
    int64 key: an order-preserving float32->uint32 map in the high bits and
    (511 - index) in the low 9 bits, so a plain integer max reproduces both the
    value comparison and the lowest-index tie-break.
  * torch's .sum(dim=-1) over 8 contiguous fp32 lanes is a pairwise tree
    ((a+b)+(c+d)) + ((e+f)+(g+h)), not a sequential accumulation. Verified
    bitwise; the sequential and stride-4 orders both differ on ~10% of rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(v, idx):
    """(value, index) -> int64 key that is monotone in stable-descending order.

    Integer comparison of the result matches float comparison of `v`, with ties
    broken toward the lower `idx` (which is what torch.topk does here).
    """
    v = tl.where(v == 0.0, 0.0, v)              # -0.0 and +0.0 compare equal
    u = v.to(tl.int32, bitcast=True)
    key = tl.where(u >= 0, u | tl.full((), -2147483648, tl.int32), ~u)
    k64 = key.to(tl.int64) & 0xFFFFFFFF          # unsigned reinterpret, no sign-extend
    return (k64 << 9) | (511 - idx).to(tl.int64)


@triton.jit
def route_kernel(
    SCORES, BIAS, OUT_IDX, OUT_W, T, rsf,
    TPB: tl.constexpr, E: tl.constexpr, NG: tl.constexpr,
    EPG: tl.constexpr, TOPK: tl.constexpr, TOPG: tl.constexpr,
):
    toks = tl.program_id(0) * TPB + tl.arange(0, TPB)
    tm = toks < T
    ecol = tl.arange(0, E)

    s = tl.load(SCORES + toks[:, None] * E + ecol[None, :],
                mask=tm[:, None], other=0.0).to(tl.float32)
    bias = tl.load(BIAS + ecol).to(tl.float32)
    sfr = s + bias[None, :]                       # scores_for_routing

    NEG = float('-inf')

    # ---- per-group top-2, summed --------------------------------------
    g = tl.reshape(sfr, (TPB, NG, EPG))
    gkey = _pack(g, tl.arange(0, EPG)[None, None, :].broadcast_to(TPB, NG, EPG))
    top1_key = tl.max(gkey, axis=2, keep_dims=True)
    v1 = tl.max(g, axis=2)
    v2 = tl.max(tl.where(gkey == top1_key, NEG, g), axis=2)
    gs = v1 + v2                                  # [TPB, NG]

    # ---- top-4 groups, by rank among unique packed keys ---------------
    gkey2 = _pack(gs, tl.arange(0, NG)[None, :])
    rank = tl.sum((gkey2[:, None, :] > gkey2[:, :, None]).to(tl.int32), axis=2)
    gsel = rank < TOPG                            # [TPB, NG]
    emask = tl.reshape(gsel[:, :, None].broadcast_to(TPB, NG, EPG), (TPB, E))

    # ---- top-8 experts among unmasked ---------------------------------
    FMIN = -3.4028234663852886e+38                # torch.finfo(float32).min
    mkey = _pack(tl.where(emask, sfr, FMIN), ecol[None, :])
    MINK = tl.full((), -9223372036854775808, tl.int64)

    sv0 = tl.zeros((TPB,), tl.float32); sv1 = tl.zeros((TPB,), tl.float32)
    sv2 = tl.zeros((TPB,), tl.float32); sv3 = tl.zeros((TPB,), tl.float32)
    sv4 = tl.zeros((TPB,), tl.float32); sv5 = tl.zeros((TPB,), tl.float32)
    sv6 = tl.zeros((TPB,), tl.float32); sv7 = tl.zeros((TPB,), tl.float32)

    for j in tl.static_range(TOPK):
        hit = mkey == tl.max(mkey, axis=1, keep_dims=True)
        sj = tl.sum(tl.where(hit, s, 0.0), axis=1)
        ij = tl.sum(tl.where(hit, ecol[None, :], 0), axis=1)
        tl.store(OUT_IDX + toks * TOPK + j, ij.to(tl.int64), mask=tm)
        mkey = tl.where(hit, MINK, mkey)
        if j == 0: sv0 = sj
        elif j == 1: sv1 = sj
        elif j == 2: sv2 = sj
        elif j == 3: sv3 = sj
        elif j == 4: sv4 = sj
        elif j == 5: sv5 = sj
        elif j == 6: sv6 = sj
        else: sv7 = sj

    # torch .sum(dim=-1) over 8 fp32 lanes is a pairwise tree
    acc = ((sv0 + sv1) + (sv2 + sv3)) + ((sv4 + sv5) + (sv6 + sv7))
    den = acc + 1e-20

    for j in tl.static_range(TOPK):
        sj = sv0
        if j == 1: sj = sv1
        elif j == 2: sj = sv2
        elif j == 3: sj = sv3
        elif j == 4: sj = sv4
        elif j == 5: sj = sv5
        elif j == 6: sj = sv6
        elif j == 7: sj = sv7
        tl.store(OUT_W + toks * TOPK + j, sj / den * rsf, mask=tm)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    experts_per_group = num_experts // n_group

    T = hidden_states.shape[0]

    # fp32 GEMM + sigmoid, bitwise-identical to F.linear(hs.float(), w.float())
    scores = torch.sigmoid(torch.mm(hidden_states.float(), weight.float().t()))

    topk_idx = torch.empty((T, top_k), dtype=torch.int64, device=scores.device)
    topk_weight = torch.empty((T, top_k), dtype=torch.float32, device=scores.device)

    TPB = 8
    route_kernel[(triton.cdiv(T, TPB),)](
        scores, expert_bias, topk_idx, topk_weight, T,
        float(routed_scaling_factor),
        TPB=TPB, E=num_experts, NG=n_group, EPG=experts_per_group,
        TOPK=top_k, TOPG=topk_group,
        num_warps=8, num_stages=1,
    )
    return topk_idx, topk_weight
