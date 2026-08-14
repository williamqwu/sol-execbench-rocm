import torch
import triton
import triton.language as tl

TOP_K = 8
TK = tl.constexpr(8)


# ---------------------------------------------------------------------------
# Kernel 1: gather tokens -> gate_proj / up_proj -> SiLU(gate) * up
# ---------------------------------------------------------------------------
@triton.jit
def _gu_kernel(
    X, WG, WU, INTER, SORTED, EIDS, NVB,
    N, K, NROWS,
    stride_xm,
    stride_we, stride_wn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NPN: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NPN
    pid_n = pid % NPN
    nvb = tl.load(NVB)
    if pid_m < nvb:
        offs_m = pid_m * BM + tl.arange(0, BM)
        rows = tl.load(SORTED + offs_m)
        mask_m = rows < NROWS
        tok = (rows // TK).to(tl.int64)
        e = tl.load(EIDS + pid_m).to(tl.int64)

        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        a_ptrs = X + tok[:, None] * stride_xm + offs_k[None, :]
        wbase = e * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None]
        wg_ptrs = WG + wbase
        wu_ptrs = WU + wbase

        accg = tl.zeros((BM, BN), dtype=tl.float32)
        accu = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            bg = tl.load(wg_ptrs)
            bu = tl.load(wu_ptrs)
            accg = tl.dot(a, bg, accg)
            accu = tl.dot(a, bu, accu)
            a_ptrs += BK
            wg_ptrs += BK
            wu_ptrs += BK

        g = accg.to(tl.bfloat16).to(tl.float32)
        u = accu.to(tl.bfloat16).to(tl.float32)
        sg = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        s = (g * sg).to(tl.bfloat16).to(tl.float32)
        out = (s * u).to(tl.bfloat16)

        o_ptrs = INTER + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(o_ptrs, out, mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Kernel 2: down_proj, scale by routing weight, write to per-slot buffer
# ---------------------------------------------------------------------------
@triton.jit
def _down_kernel(
    INTER, WD, Y, SORTED, EIDS, NVB, TW, RANKF,
    N, K, NROWS,
    stride_we, stride_wn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NPN: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // NPN
    pid_n = pid % NPN
    nvb = tl.load(NVB)
    if pid_m < nvb:
        offs_m = pid_m * BM + tl.arange(0, BM)
        rows = tl.load(SORTED + offs_m)
        mask_m = rows < NROWS
        e = tl.load(EIDS + pid_m).to(tl.int64)

        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        a_ptrs = INTER + offs_m[:, None].to(tl.int64) * K + offs_k[None, :]
        w_ptrs = WD + e * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None]

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(w_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            w_ptrs += BK

        o = acc.to(tl.bfloat16).to(tl.float32)
        w = tl.load(TW + rows, mask=mask_m, other=0.0).to(tl.float32)
        res = (o * w[:, None]).to(tl.bfloat16)

        tok = rows // TK
        rnk = tl.load(RANKF + rows, mask=mask_m, other=0)
        drow = (tok * TK + rnk).to(tl.int64)
        y_ptrs = Y + drow[:, None] * N + offs_n[None, :]
        tl.store(y_ptrs, res, mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Kernel 3: sequential bf16 accumulation of the 8 slots (expert-index order)
# ---------------------------------------------------------------------------
@triton.jit
def _combine_kernel(Y, OUT, H, BH: tl.constexpr):
    t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BH + tl.arange(0, BH)
    base = Y + t.to(tl.int64) * (TK * H) + offs
    acc = tl.load(base).to(tl.float32)
    for i in tl.static_range(1, TK):
        v = tl.load(base + i * H).to(tl.float32)
        acc = (acc + v).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Alignment helpers (Triton, to keep launch count low)
# ---------------------------------------------------------------------------
@triton.jit
def _count_rank_kernel(SEL, CNT, RANKPOS, NROWS, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < NROWS
    e = tl.load(SEL + offs, mask=mask, other=0)
    r = tl.atomic_add(CNT + e, 1, mask=mask)
    tl.store(RANKPOS + offs, r, mask=mask)


@triton.jit
def _prefix_kernel(CNT, STARTS, NVB, E: tl.constexpr, BM: tl.constexpr):
    offs = tl.arange(0, E)
    c = tl.load(CNT + offs)
    padded = ((c + BM - 1) // BM) * BM
    incl = tl.cumsum(padded, axis=0)
    starts = incl - padded
    tl.store(STARTS + offs, starts)
    total = tl.sum(padded, axis=0)
    tl.store(NVB, total // BM)


@triton.jit
def _scatter_kernel(SEL, RANKPOS, STARTS, SORTED, NROWS, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < NROWS
    e = tl.load(SEL + offs, mask=mask, other=0)
    r = tl.load(RANKPOS + offs, mask=mask, other=0)
    s = tl.load(STARTS + e, mask=mask, other=0)
    tl.store(SORTED + s + r, offs.to(tl.int32), mask=mask)


@triton.jit
def _eids_kernel(STARTS, EIDS, NBLOCKS, E: tl.constexpr, BM: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < NBLOCKS
    st = tl.load(STARTS + tl.arange(0, E)) // BM
    cnt = tl.sum((st[None, :] <= offs[:, None]).to(tl.int32), axis=1) - 1
    tl.store(EIDS + offs, cnt, mask=mask)


CFG = dict(BM=128, BN1=64, BK1=64, w1=4, s1=2, BN2=128, BK2=64, w2=8, s2=2)


def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    norm_topk_prob: bool,
):
    B, S, H = hidden_states.shape
    E = gate_weight.shape[0]
    I = expert_gate_proj.shape[1]
    x = hidden_states.reshape(-1, H)
    T = x.shape[0]
    dev = x.device

    # ---- routing (exact torch semantics) ----
    logits = torch.matmul(x, gate_weight.t())
    rw = torch.softmax(logits.float(), dim=1).to(x.dtype)
    tw, sel = torch.topk(rw, TOP_K, dim=-1)
    if norm_topk_prob:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-9)

    # slot position of each (token, k) in ascending-expert order
    rank8 = (sel.unsqueeze(-1) > sel.unsqueeze(-2)).sum(-1).to(torch.int32)

    nrows = T * TOP_K
    sel_flat = sel.reshape(-1).to(torch.int32)
    tw_flat = tw.reshape(-1).contiguous()
    rank_flat = rank8.reshape(-1).contiguous()

    max_blocks = triton.cdiv(nrows, CFG['BM']) + E
    cnt = torch.zeros(E, dtype=torch.int32, device=dev)
    starts = torch.empty(E, dtype=torch.int32, device=dev)
    nvb = torch.empty(1, dtype=torch.int32, device=dev)
    rankpos = torch.empty(nrows, dtype=torch.int32, device=dev)
    sorted_ids = torch.full((max_blocks * CFG['BM'],), nrows, dtype=torch.int32, device=dev)
    eids = torch.zeros(max_blocks, dtype=torch.int32, device=dev)

    BLK = 1024
    _count_rank_kernel[(triton.cdiv(nrows, BLK),)](sel_flat, cnt, rankpos, nrows, BLK=BLK)
    _prefix_kernel[(1,)](cnt, starts, nvb, E=E, BM=CFG['BM'])
    _scatter_kernel[(triton.cdiv(nrows, BLK),)](sel_flat, rankpos, starts, sorted_ids, nrows, BLK=BLK)
    _eids_kernel[(triton.cdiv(max_blocks, 256),)](starts, eids, max_blocks, E=E, BM=CFG['BM'], BLK=256)

    inter = torch.empty((max_blocks * CFG['BM'], I), dtype=torch.bfloat16, device=dev)
    npn1 = triton.cdiv(I, CFG['BN1'])
    _gu_kernel[(max_blocks * npn1,)](
        x, expert_gate_proj, expert_up_proj, inter, sorted_ids, eids, nvb,
        I, H, nrows,
        x.stride(0),
        expert_gate_proj.stride(0), expert_gate_proj.stride(1),
        BM=CFG['BM'], BN=CFG['BN1'], BK=CFG['BK1'], NPN=npn1,
        num_warps=CFG['w1'], num_stages=CFG['s1'],
    )

    y = torch.empty((nrows, H), dtype=torch.bfloat16, device=dev)
    npn2 = triton.cdiv(H, CFG['BN2'])
    _down_kernel[(max_blocks * npn2,)](
        inter, expert_down_proj, y, sorted_ids, eids, nvb, tw_flat, rank_flat,
        H, I, nrows,
        expert_down_proj.stride(0), expert_down_proj.stride(1),
        BM=CFG['BM'], BN=CFG['BN2'], BK=CFG['BK2'], NPN=npn2,
        num_warps=CFG['w2'], num_stages=CFG['s2'],
    )

    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    BH = 512
    _combine_kernel[(T, triton.cdiv(H, BH))](y, out, H, BH=BH, num_warps=4)

    return out.view(B, S, H)
