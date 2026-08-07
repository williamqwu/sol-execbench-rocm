import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- kernels ---

@triton.jit
def _gateup_kernel(
    A, WG, WU, OUT,
    SORTED, BLKEXP, NBLK,
    P, K, N,
    s_am, s_we, s_wn, s_om,
    TOPK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = N // BN
    pid_m = pid // num_n
    pid_n = pid % num_n

    nb = tl.load(NBLK)
    if pid_m >= nb:
        return

    e = tl.load(BLKEXP + pid_m)

    offs_m = pid_m * BM + tl.arange(0, BM)
    pidx = tl.load(SORTED + offs_m)
    mask_m = pidx < P
    tok = pidx // TOPK

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + tok[:, None] * s_am + offs_k[None, :]
    woff = e * s_we + offs_n[None, :] * s_wn + offs_k[:, None]
    wg_ptrs = WG + woff
    wu_ptrs = WU + woff

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        bg = tl.load(wg_ptrs)
        bu = tl.load(wu_ptrs)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        a_ptrs += BK
        wg_ptrs += BK
        wu_ptrs += BK

    # emulate the reference's intermediate bf16 rounding
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    d = (1.0 + tl.exp(-g)).to(tl.bfloat16).to(tl.float32)
    s = (g / d).to(tl.bfloat16).to(tl.float32)
    inter = (s * u).to(tl.bfloat16)

    o_ptrs = OUT + pidx[:, None] * s_om + offs_n[None, :]
    tl.store(o_ptrs, inter, mask=mask_m[:, None])


@triton.jit
def _down_kernel(
    I, WD, OUT, RW,
    SORTED, BLKEXP, NBLK,
    P, K, N,
    s_im, s_we, s_wn, s_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = N // BN
    pid_m = pid // num_n
    pid_n = pid % num_n

    nb = tl.load(NBLK)
    if pid_m >= nb:
        return

    e = tl.load(BLKEXP + pid_m)

    offs_m = pid_m * BM + tl.arange(0, BM)
    pidx = tl.load(SORTED + offs_m)
    mask_m = pidx < P

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = I + pidx[:, None] * s_im + offs_k[None, :]
    w_ptrs = WD + e * s_we + offs_n[None, :] * s_wn + offs_k[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        w_ptrs += BK

    o = acc.to(tl.bfloat16).to(tl.float32)
    w = tl.load(RW + pidx, mask=mask_m, other=0.0).to(tl.float32)
    res = (o * w[:, None]).to(tl.bfloat16)

    o_ptrs = OUT + pidx[:, None] * s_om + offs_n[None, :]
    tl.store(o_ptrs, res, mask=mask_m[:, None])


@triton.jit
def _reduce_kernel(
    BUF, SHARED, OUT,
    H,
    TOPK: tl.constexpr, BH: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs = pid_h * BH + tl.arange(0, BH)

    acc = tl.zeros((BH,), dtype=tl.float32)
    base = BUF + (t * TOPK) * H + offs
    for j in tl.static_range(TOPK):
        acc += tl.load(base + j * H).to(tl.float32)

    acc = acc.to(tl.bfloat16).to(tl.float32)
    sh = tl.load(SHARED + t * H + offs).to(tl.float32)
    tl.store(OUT + t * H + offs, (acc + sh).to(tl.bfloat16))


@triton.jit
def _shared_gateup_kernel(
    A, WG, WU, WGATE, OUT, GOUT,
    M, K, N,
    s_am, s_wn, s_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = N // BN
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + offs_m[:, None] * s_am + offs_k[None, :]
    woff = offs_n[None, :] * s_wn + offs_k[:, None]
    wg_ptrs = WG + woff
    wu_ptrs = WU + woff
    wgate_ptrs = WGATE + offs_k

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    acc_gate = tl.zeros((BM,), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        bg = tl.load(wg_ptrs)
        bu = tl.load(wu_ptrs)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        if pid_n == 0:
            wgt = tl.load(wgate_ptrs).to(tl.float32)
            acc_gate += tl.sum(a.to(tl.float32) * wgt[None, :], axis=1)
        a_ptrs += BK
        wg_ptrs += BK
        wu_ptrs += BK
        wgate_ptrs += BK

    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    d = (1.0 + tl.exp(-g)).to(tl.bfloat16).to(tl.float32)
    s = (g / d).to(tl.bfloat16).to(tl.float32)
    inter = (s * u).to(tl.bfloat16)

    o_ptrs = OUT + offs_m[:, None] * s_om + offs_n[None, :]
    tl.store(o_ptrs, inter, mask=mask_m[:, None])

    if pid_n == 0:
        gate = 1.0 / (1.0 + tl.exp(-acc_gate.to(tl.bfloat16).to(tl.float32)))
        tl.store(GOUT + offs_m, gate.to(tl.bfloat16), mask=mask_m)


@triton.jit
def _shared_down_kernel(
    I, WD, GATE, OUT,
    M, K, N,
    s_im, s_wn, s_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = N // BN
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = I + offs_m[:, None] * s_im + offs_k[None, :]
    w_ptrs = WD + offs_n[None, :] * s_wn + offs_k[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        w_ptrs += BK

    o = acc.to(tl.bfloat16).to(tl.float32)
    gt = tl.load(GATE + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    res = (gt[:, None] * o).to(tl.bfloat16)
    tl.store(OUT + offs_m[:, None] * s_om + offs_n[None, :], res,
             mask=mask_m[:, None])


# ------------------------------------------------------------- host logic ---

_ARANGE_CACHE = {}


def _arange(n, device):
    key = (n, device)
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device, dtype=torch.int32)
        _ARANGE_CACHE[key] = t
    return t


def _pick_bm(p, e):
    avg = p / max(e, 1)
    target = avg + 4.0 * (avg ** 0.5) + 8.0
    bm = 32
    while bm < target and bm < 512:
        bm *= 2
    return bm


def _align(topk_ids_flat, num_experts, bm):
    """Sort token-expert pairs by expert, pad each expert to a multiple of bm."""
    device = topk_ids_flat.device
    p = topk_ids_flat.numel()

    cnt = torch.bincount(topk_ids_flat, minlength=num_experts)
    nblk = (cnt + (bm - 1)) // bm
    blk_cum = torch.cumsum(nblk, 0)
    pad_start = (blk_cum - nblk) * bm

    order = torch.argsort(topk_ids_flat, stable=True)
    sorted_e = topk_ids_flat[order]
    cnt_excl = torch.cumsum(cnt, 0) - cnt
    rank = _arange(p, device).to(torch.int64) - cnt_excl[sorted_e]
    dest = pad_start[sorted_e] + rank

    maxblk = (p + bm - 1) // bm + num_experts
    sorted_ids = torch.full((maxblk * bm,), p, dtype=torch.int32, device=device)
    sorted_ids[dest] = order.to(torch.int32)

    blk_exp = torch.searchsorted(
        blk_cum, _arange(maxblk, device).to(torch.int64), right=True
    ).to(torch.int32)
    nblk_total = blk_cum[num_experts - 1:num_experts].to(torch.int32)
    return sorted_ids, blk_exp, nblk_total, maxblk


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_gate_proj_weights: torch.Tensor,
    expert_up_proj_weights: torch.Tensor,
    expert_down_proj_weights: torch.Tensor,
    shared_expert_gate_proj_weight: torch.Tensor,
    shared_expert_up_proj_weight: torch.Tensor,
    shared_expert_down_proj_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
):
    M, H = hidden_states.shape
    E, I, _ = expert_gate_proj_weights.shape
    TOPK = selected_experts.shape[1]
    P = M * TOPK
    device = hidden_states.device
    SI = shared_expert_gate_proj_weight.shape[0]

    flat_ids = selected_experts.reshape(-1)
    bm = _pick_bm(P, E)
    sorted_ids, blk_exp, nblk_total, maxblk = _align(flat_ids, E, bm)

    inter = torch.empty((P, I), dtype=torch.bfloat16, device=device)
    buf = torch.empty((P, H), dtype=torch.bfloat16, device=device)
    sh_inter = torch.empty((M, SI), dtype=torch.bfloat16, device=device)
    sh_gate = torch.empty((M,), dtype=torch.bfloat16, device=device)
    sh_out = torch.empty((M, H), dtype=torch.bfloat16, device=device)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=device)

    rw = routing_weights.reshape(-1)

    # --- shared expert (independent of the routed path) ---
    BM_S, BN_S, BK_S = 128, 128, 64
    grid_s = (triton.cdiv(M, BM_S) * (SI // BN_S),)
    _shared_gateup_kernel[grid_s](
        hidden_states, shared_expert_gate_proj_weight,
        shared_expert_up_proj_weight, shared_expert_gate_weight,
        sh_inter, sh_gate,
        M, H, SI,
        hidden_states.stride(0), shared_expert_gate_proj_weight.stride(0),
        sh_inter.stride(0),
        BM=BM_S, BN=BN_S, BK=BK_S, num_warps=8, num_stages=2,
    )
    grid_sd = (triton.cdiv(M, BM_S) * (H // BN_S),)
    _shared_down_kernel[grid_sd](
        sh_inter, shared_expert_down_proj_weight, sh_gate, sh_out,
        M, SI, H,
        sh_inter.stride(0), shared_expert_down_proj_weight.stride(0),
        sh_out.stride(0),
        BM=BM_S, BN=BN_S, BK=BK_S, num_warps=8, num_stages=2,
    )

    # --- routed experts ---
    BN1, BK1 = 128, 64
    grid1 = (maxblk * (I // BN1),)
    _gateup_kernel[grid1](
        hidden_states, expert_gate_proj_weights, expert_up_proj_weights, inter,
        sorted_ids, blk_exp, nblk_total,
        P, H, I,
        hidden_states.stride(0), expert_gate_proj_weights.stride(0),
        expert_gate_proj_weights.stride(1), inter.stride(0),
        TOPK=TOPK, BM=bm, BN=BN1, BK=BK1, num_warps=8, num_stages=2,
    )

    BN2, BK2 = 128, 64
    grid2 = (maxblk * (H // BN2),)
    _down_kernel[grid2](
        inter, expert_down_proj_weights, buf, rw,
        sorted_ids, blk_exp, nblk_total,
        P, I, H,
        inter.stride(0), expert_down_proj_weights.stride(0),
        expert_down_proj_weights.stride(1), buf.stride(0),
        BM=bm, BN=BN2, BK=BK2, num_warps=8, num_stages=2,
    )

    BH = 512
    _reduce_kernel[(M, H // BH)](
        buf, sh_out, out, H, TOPK=TOPK, BH=BH, num_warps=4,
    )
    return out
