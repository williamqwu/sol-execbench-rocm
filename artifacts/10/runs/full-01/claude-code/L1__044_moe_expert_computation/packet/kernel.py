import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused gate + up projection with SwiGLU
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_kernel(
    A,            # (num_tokens, H) bf16
    WG,           # (E, I, H) bf16
    WU,           # (E, I, H) bf16
    INTER,        # (EM, I) bf16
    SORTED,       # (EM,) int32  pair index or sentinel
    EOB,          # (num_blocks_ub,) int32  expert of block, -1 = inactive
    num_valid,
    H,
    stride_am,
    stride_we, stride_wn,
    stride_im,
    NUM_N: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N
    pid_n = pid % NUM_N

    e = tl.load(EOB + pid_m)
    if e < 0:
        return

    offs_m = pid_m * BM + tl.arange(0, BM)
    pair = tl.load(SORTED + offs_m)
    mask_m = pair < num_valid
    row = tl.where(mask_m, pair // TOPK, 0)

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + row[:, None] * stride_am + offs_k[None, :]
    wbase = e.to(tl.int64) * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None]
    wg_ptrs = WG + wbase
    wu_ptrs = WU + wbase

    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(0, H, BK):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        wg = tl.load(wg_ptrs)
        wu = tl.load(wu_ptrs)
        accg = tl.dot(a, wg, accg)
        accu = tl.dot(a, wu, accu)
        a_ptrs += BK
        wg_ptrs += BK
        wu_ptrs += BK

    # match reference rounding: matmul outputs are bf16, silu in bf16, product bf16
    g = accg.to(tl.bfloat16).to(tl.float32)
    u = accu.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    res = (s * u).to(tl.bfloat16)

    tl.store(INTER + offs_m[:, None] * stride_im + offs_n[None, :], res,
             mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Kernel 2: down projection + routing weight + scatter accumulate
# ---------------------------------------------------------------------------
@triton.jit
def _down_kernel(
    INTER,        # (EM, I) bf16
    WD,           # (E, H, I) bf16
    OUT,          # (num_tokens, H) fp32
    EW,           # (num_tokens, TOPK) bf16
    SORTED,       # (EM,) int32
    EOB,          # int32
    num_valid,
    I,
    stride_im,
    stride_we, stride_wn,
    stride_om,
    NUM_N: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NUM_N
    pid_n = pid % NUM_N

    e = tl.load(EOB + pid_m)
    if e < 0:
        return

    offs_m = pid_m * BM + tl.arange(0, BM)
    pair = tl.load(SORTED + offs_m)
    mask_m = pair < num_valid
    pairc = tl.where(mask_m, pair, 0)
    row = pairc // TOPK

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    i_ptrs = INTER + offs_m[:, None] * stride_im + offs_k[None, :]
    w_ptrs = WD + e.to(tl.int64) * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, I, BK):
        a = tl.load(i_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc)
        i_ptrs += BK
        w_ptrs += BK

    rw = tl.load(EW + pairc, mask=mask_m, other=0.0).to(tl.float32)
    res = acc.to(tl.bfloat16).to(tl.float32) * rw[:, None]

    tl.atomic_add(OUT + row[:, None] * stride_om + offs_n[None, :], res,
                  mask=mask_m[:, None], sem="relaxed")


# ---------------------------------------------------------------------------
# Metadata: block-aligned sort of (token, slot) pairs by expert
# ---------------------------------------------------------------------------
def _build_meta(expert_ids, E, BM, ub_blocks):
    ids = expert_ids.reshape(-1)
    npair = ids.numel()
    dev = ids.device

    order = torch.argsort(ids, stable=True)
    counts = torch.bincount(ids, minlength=E)
    cum_c = torch.cumsum(counts, 0)
    starts = cum_c - counts
    nblk = (counts + BM - 1) // BM
    cum_b = torch.cumsum(nblk, 0)
    blk_start = cum_b - nblk

    blk = torch.arange(ub_blocks, device=dev)
    eob = torch.searchsorted(cum_b, blk, right=True)
    active = eob < E
    eobc = torch.where(active, eob, 0)

    within = torch.arange(BM, device=dev)
    local = (blk - blk_start[eobc]).unsqueeze(1) * BM + within.unsqueeze(0)
    valid = (local < counts[eobc].unsqueeze(1)) & active.unsqueeze(1)
    src = (starts[eobc].unsqueeze(1) + local).clamp_(0, npair - 1)
    sorted_ids = torch.where(valid, order[src], npair).to(torch.int32).reshape(-1)
    eob_i32 = torch.where(active, eob, -1).to(torch.int32)
    return sorted_ids, eob_i32


def _pick_bm(npair, E):
    avg = npair / E
    if avg <= 120:
        return 128
    return 256


def run(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
) -> torch.Tensor:
    T, H = hidden_states.shape
    TOPK = expert_ids.shape[1]
    E, I, _ = gate_proj_weights.shape
    npair = T * TOPK

    BM = _pick_bm(npair, E)
    ub_blocks = E + npair // BM
    sorted_ids, eob = _build_meta(expert_ids, E, BM, ub_blocks)

    EM = ub_blocks * BM
    inter = torch.empty((EM, I), dtype=torch.bfloat16, device=hidden_states.device)

    BN1, BK1 = 128, 64
    num_n1 = I // BN1
    _gate_up_kernel[(ub_blocks * num_n1,)](
        hidden_states, gate_proj_weights, up_proj_weights, inter,
        sorted_ids, eob, npair, H,
        hidden_states.stride(0),
        gate_proj_weights.stride(0), gate_proj_weights.stride(1),
        inter.stride(0),
        NUM_N=num_n1, TOPK=TOPK, BM=BM, BN=BN1, BK=BK1,
        num_warps=8, num_stages=2,
    )

    out32 = torch.zeros((T, H), dtype=torch.float32, device=hidden_states.device)
    BN2, BK2 = 128, 64
    num_n2 = H // BN2
    _down_kernel[(ub_blocks * num_n2,)](
        inter, down_proj_weights, out32, expert_weights,
        sorted_ids, eob, npair, I,
        inter.stride(0),
        down_proj_weights.stride(0), down_proj_weights.stride(1),
        out32.stride(0),
        NUM_N=num_n2, TOPK=TOPK, BM=BM, BN=BN2, BK=BK2,
        num_warps=8, num_stages=2,
    )

    return out32.to(torch.bfloat16)
