import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Routing: bucket the (token, slot) pairs by expert, padding each expert's
# region up to a multiple of BLOCK_M so every GEMM tile touches one expert.
# ---------------------------------------------------------------------------

@triton.jit
def _count_kernel(sel_ptr, counts_ptr, NP, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NP
    e = tl.load(sel_ptr + offs, mask=m, other=0).to(tl.int32)
    tl.atomic_add(counts_ptr + e, 1, mask=m)


@triton.jit
def _prefix_kernel(counts_ptr, cend_ptr, vend_ptr, cursor_ptr,
                   E: tl.constexpr, EP: tl.constexpr, BLOCK_M: tl.constexpr):
    offs = tl.arange(0, EP)
    m = offs < E
    c = tl.load(counts_ptr + offs, mask=m, other=0)
    pad = ((c + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    pad = tl.where(m, pad, 0)
    cend = tl.cumsum(pad, axis=0)
    cstart = cend - pad
    tl.store(cend_ptr + offs, cend, mask=m)
    tl.store(vend_ptr + offs, cstart + c, mask=m)
    tl.store(cursor_ptr + offs, cstart, mask=m)


@triton.jit
def _blockmap_kernel(cend_ptr, vend_e_ptr, blk_expert_ptr, blk_vend_ptr,
                     NUM_M, E: tl.constexpr, EP: tl.constexpr,
                     BLOCK_M: tl.constexpr, CH: tl.constexpr):
    pid = tl.program_id(0)
    b = pid * CH + tl.arange(0, CH)
    bm = b < NUM_M
    bs = b * BLOCK_M

    eo = tl.arange(0, EP)
    em = eo < E
    cend = tl.load(cend_ptr + eo, mask=em, other=0)
    vend_e = tl.load(vend_e_ptr + eo, mask=em, other=0)

    le = (cend[:, None] <= bs[None, :]) & em[:, None]
    eidx = tl.sum(le.to(tl.int32), axis=0)
    inrange = eidx < E
    eidx = tl.minimum(eidx, E - 1)
    sel = (eo[:, None] == eidx[None, :]) & em[:, None]
    vend = tl.sum(tl.where(sel, vend_e[:, None], 0), axis=0)
    vend = tl.where(inrange, vend, 0)

    tl.store(blk_expert_ptr + b, eidx, mask=bm)
    tl.store(blk_vend_ptr + b, vend, mask=bm)


@triton.jit
def _scatter_kernel(sel_ptr, cursor_ptr, sorted_ids_ptr, NP, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NP
    e = tl.load(sel_ptr + offs, mask=m, other=0).to(tl.int32)
    pos = tl.atomic_add(cursor_ptr + e, 1, mask=m)
    tl.store(sorted_ids_ptr + pos, offs.to(tl.int32), mask=m)


# ---------------------------------------------------------------------------
# Stage 1: gate & up projections + SwiGLU   ->  inter[EM, I] (bf16)
# ---------------------------------------------------------------------------

@triton.jit
def _moe_gate_up(
    a_ptr, gw_ptr, uw_ptr, out_ptr,
    sorted_ids_ptr, blk_expert_ptr, blk_vend_ptr,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_N: tl.constexpr, NUM_M, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    n_in_group = GROUP_M * NUM_N
    gid = pid // n_in_group
    first_m = gid * GROUP_M
    gsz = tl.minimum(NUM_M - first_m, GROUP_M)
    r = pid % n_in_group
    pid_m = first_m + (r % gsz)
    pid_n = r // gsz

    vend = tl.load(blk_vend_ptr + pid_m)
    bs = pid_m * BLOCK_M
    if bs < vend:
        e = tl.load(blk_expert_ptr + pid_m)

        offs_m = bs + tl.arange(0, BLOCK_M)
        mm = offs_m < vend
        tp = tl.load(sorted_ids_ptr + offs_m, mask=mm, other=0)
        rows = tp // TOPK

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + rows[:, None].to(tl.int64) * H + offs_k[None, :]
        woff = (e.to(tl.int64) * (I * H)
                + offs_n[None, :].to(tl.int64) * H + offs_k[:, None])
        g_ptrs = gw_ptr + woff
        u_ptrs = uw_ptr + woff

        accg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        accu = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in tl.range(0, H // BLOCK_K):
            a = tl.load(a_ptrs, mask=mm[:, None], other=0.0)
            bg = tl.load(g_ptrs)
            bu = tl.load(u_ptrs)
            accg = tl.dot(a, bg, accg)
            accu = tl.dot(a, bu, accu)
            a_ptrs += BLOCK_K
            g_ptrs += BLOCK_K
            u_ptrs += BLOCK_K

        inter = (accg / (1.0 + tl.exp(-accg))) * accu
        o_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :]
        tl.store(o_ptrs, inter.to(out_ptr.dtype.element_ty), mask=mm[:, None])


# ---------------------------------------------------------------------------
# Stage 2: down projection * routing weight, scatter-accumulated into out
# ---------------------------------------------------------------------------

@triton.jit
def _moe_down(
    inter_ptr, dw_ptr, out_ptr, rw_ptr,
    sorted_ids_ptr, blk_expert_ptr, blk_vend_ptr,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_N: tl.constexpr, NUM_M, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    n_in_group = GROUP_M * NUM_N
    gid = pid // n_in_group
    first_m = gid * GROUP_M
    gsz = tl.minimum(NUM_M - first_m, GROUP_M)
    r = pid % n_in_group
    pid_m = first_m + (r % gsz)
    pid_n = r // gsz

    vend = tl.load(blk_vend_ptr + pid_m)
    bs = pid_m * BLOCK_M
    if bs < vend:
        e = tl.load(blk_expert_ptr + pid_m)

        offs_m = bs + tl.arange(0, BLOCK_M)
        mm = offs_m < vend
        tp = tl.load(sorted_ids_ptr + offs_m, mask=mm, other=0)
        rows = tp // TOPK

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = inter_ptr + offs_m[:, None].to(tl.int64) * I + offs_k[None, :]
        b_ptrs = (dw_ptr + e.to(tl.int64) * (H * I)
                  + offs_n[None, :].to(tl.int64) * I + offs_k[:, None])

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, I // BLOCK_K):
            a = tl.load(a_ptrs, mask=mm[:, None], other=0.0)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        w = tl.load(rw_ptr + tp, mask=mm, other=0.0)
        acc = acc * w[:, None]

        # write to a per-(token,slot) slab; a cheap reduce pass sums the TOPK
        # contributions afterwards. Far cheaper than global fp32 atomics.
        o_ptrs = out_ptr + tp[:, None].to(tl.int64) * H + offs_n[None, :]
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mm[:, None])


@triton.jit
def _reduce_topk(buf_ptr, out_ptr, H: tl.constexpr, TOPK: tl.constexpr,
                 M, BLOCK_H: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    base = pid_m.to(tl.int64) * (TOPK * H) + offs_h
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(buf_ptr + base + k * H).to(tl.float32)
    tl.store(out_ptr + pid_m.to(tl.int64) * H + offs_h,
             acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------

# Every weight byte is re-read once per M-tile that an expert spans, so the
# tile height must grow with the average tokens-per-expert or large batches
# stream the full 1.2 GB weight set several times over.
_TUNED = {
    # BLOCK_M: (gate_up BN, BK, warps, stages), (down BN, BK, warps, stages)
    64:  ((64, 64, 4, 1), (128, 64, 8, 3)),
    128: ((64, 64, 4, 2), (256, 32, 8, 2)),
    256: ((64, 64, 4, 1), (256, 32, 8, 3)),
}
GROUP_M = 8


def _choose_block_m(NP, E):
    avg = NP / E
    for bm in (64, 128):
        if avg <= bm:
            return bm
    return 256


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
) -> torch.Tensor:
    M, H = hidden_states.shape
    E, I, _ = gate_proj_weights.shape
    TOPK = selected_experts.shape[1]
    NP = M * TOPK
    dev = hidden_states.device

    BLOCK_M = _choose_block_m(NP, E)
    (GU_BN, GU_BK, GU_W, GU_S), (DN_BN, DN_BK, DN_W, DN_S) = _TUNED[BLOCK_M]
    NUM_M = (NP + BLOCK_M - 1) // BLOCK_M + E
    EM = NUM_M * BLOCK_M
    EP = triton.next_power_of_2(E)

    sel = selected_experts.reshape(-1)
    rw = routing_weights.reshape(-1)

    counts = torch.zeros(E, dtype=torch.int32, device=dev)
    cend = torch.empty(E, dtype=torch.int32, device=dev)
    vend_e = torch.empty(E, dtype=torch.int32, device=dev)
    cursor = torch.empty(E, dtype=torch.int32, device=dev)
    blk_expert = torch.empty(NUM_M, dtype=torch.int32, device=dev)
    blk_vend = torch.empty(NUM_M, dtype=torch.int32, device=dev)
    sorted_ids = torch.empty(EM, dtype=torch.int32, device=dev)

    CB = 1024
    ngrid = (triton.cdiv(NP, CB),)
    _count_kernel[ngrid](sel, counts, NP, BLOCK=CB, num_warps=4)
    _prefix_kernel[(1,)](counts, cend, vend_e, cursor,
                         E=E, EP=EP, BLOCK_M=BLOCK_M, num_warps=4)
    CH = 64
    _blockmap_kernel[(triton.cdiv(NUM_M, CH),)](
        cend, vend_e, blk_expert, blk_vend, NUM_M,
        E=E, EP=EP, BLOCK_M=BLOCK_M, CH=CH, num_warps=4)
    _scatter_kernel[ngrid](sel, cursor, sorted_ids, NP, BLOCK=CB, num_warps=4)

    inter = torch.empty((EM, I), dtype=torch.bfloat16, device=dev)
    buf = torch.empty((NP, H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    NUM_N1 = I // GU_BN
    _moe_gate_up[(NUM_M * NUM_N1,)](
        hidden_states, gate_proj_weights, up_proj_weights, inter,
        sorted_ids, blk_expert, blk_vend,
        H=H, I=I, TOPK=TOPK,
        BLOCK_M=BLOCK_M, BLOCK_N=GU_BN, BLOCK_K=GU_BK,
        NUM_N=NUM_N1, NUM_M=NUM_M, GROUP_M=GROUP_M,
        num_warps=GU_W, num_stages=GU_S,
    )

    NUM_N2 = H // DN_BN
    _moe_down[(NUM_M * NUM_N2,)](
        inter, down_proj_weights, buf, rw,
        sorted_ids, blk_expert, blk_vend,
        H=H, I=I, TOPK=TOPK,
        BLOCK_M=BLOCK_M, BLOCK_N=DN_BN, BLOCK_K=DN_BK,
        NUM_N=NUM_N2, NUM_M=NUM_M, GROUP_M=GROUP_M,
        num_warps=DN_W, num_stages=DN_S,
    )

    BH = 512
    _reduce_topk[(M, H // BH)](buf, out, H=H, TOPK=TOPK, M=M,
                               BLOCK_H=BH, num_warps=4)
    return out
