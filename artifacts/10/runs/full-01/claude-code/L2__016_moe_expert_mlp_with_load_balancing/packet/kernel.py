import os
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused MoE (Qwen3-Omni-30B-A3B block): router -> softmax -> top8 -> normalize
# -> per-expert SwiGLU MLP -> routing-weighted scatter-add.
#
# Lower bound is reading the expert weights once:
#   (2*64*2560*4096 + 64*4096*2560) * 2 B = 4.03 GB  -> ~0.5 ms at 8 TB/s
# so the design goal is: touch each weight byte as close to once as possible.
# That means BLOCK_M large enough that each expert is few m-blocks, since
# weight traffic = (#m-blocks per expert) x (full expert weight).
#
# Two grouped-GEMM kernels over an expert-sorted, block-padded token list:
#   K1: x[tok] @ Wg^T and x[tok] @ Wu^T (shared x tile) -> SwiGLU -> inter
#   K2: inter @ Wd^T -> scale by routing weight -> atomic scatter-add
# ---------------------------------------------------------------------------


@triton.jit
def _gate_up_swiglu(
    X, WG, WU, OUT,
    SORTED_TOK, EXPERT_IDS,
    K, N,
    stride_xm, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_om, stride_on,
    NUM_PID_M, NUM_PID_N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    # Grouped (L2-friendly) ordering: cover a GROUP_M x NUM_PID_N super-tile
    # before moving on, so both the x tile and the weight tile get reuse.
    num_pid_in_group = GROUP_M * NUM_PID_N
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(NUM_PID_M - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e = tl.load(EXPERT_IDS + pid_m)
    if e < 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(SORTED_TOK + offs_m)
    rowmask = tok >= 0

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + tok[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wbase = e.to(tl.int64) * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    wg_ptrs = WG + wbase
    wu_ptrs = WU + wbase

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs, mask=rowmask[:, None], other=0.0)
        bg = tl.load(wg_ptrs)
        bu = tl.load(wu_ptrs)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk

    # Reproduce the reference's intermediate rounding exactly:
    #   g, u  = bf16(gemm)                      (F.linear outputs)
    #   den   = bf16(1 + exp(-float32(g)))
    #   sg    = bf16(float32(g) / float32(den))
    #   inter = bf16(sg * u)
    g = acc_g.to(tl.bfloat16)
    u = acc_u.to(tl.bfloat16)
    gf = g.to(tl.float32)
    den = (1.0 + tl.exp(-gf)).to(tl.bfloat16).to(tl.float32)
    sg = (gf / den).to(tl.bfloat16)
    inter = (sg.to(tl.float32) * u.to(tl.float32)).to(tl.bfloat16)

    o_ptrs = OUT + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, inter, mask=rowmask[:, None])


@triton.jit
def _down_scatter(
    INTER, WD, OUTF,
    SORTED_TOK, SORTED_W, EXPERT_IDS,
    K, N,
    stride_im, stride_ik,
    stride_we, stride_wn, stride_wk,
    stride_om, stride_on,
    NUM_PID_M, NUM_PID_N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_in_group = GROUP_M * NUM_PID_N
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(NUM_PID_M - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e = tl.load(EXPERT_IDS + pid_m)
    if e < 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(SORTED_TOK + offs_m)
    rowmask = tok >= 0

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    i_ptrs = INTER + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    w_ptrs = WD + e.to(tl.int64) * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(i_ptrs, mask=rowmask[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc)
        i_ptrs += BLOCK_K * stride_ik
        w_ptrs += BLOCK_K * stride_wk

    rw = tl.load(SORTED_W + offs_m, mask=rowmask, other=0.0).to(tl.float32)
    # reference: bf16(gemm) * bf16(routing weight) -> bf16, then index_add_
    val = (acc.to(tl.bfloat16).to(tl.float32) * rw[:, None]).to(tl.bfloat16).to(tl.float32)

    o_ptrs = OUTF + tok[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(o_ptrs, val, mask=rowmask[:, None])


# ---------------------------------------------------------------------------


def plan(sel_i32, tw_f32, BM, num_experts=64, topk=8):
    """Expert-sorted, block-padded token list. No device->host sync."""
    dev = sel_i32.device
    Np = sel_i32.numel()
    flat_e = sel_i32.reshape(-1)

    order = torch.argsort(flat_e, stable=True)
    sorted_e = flat_e[order]

    counts = torch.bincount(flat_e, minlength=num_experts)
    padded = ((counts + (BM - 1)) // BM) * BM
    cum_pad_end = torch.cumsum(padded, 0)
    cum_pad = cum_pad_end - padded
    cum_unpad = torch.cumsum(counts, 0) - counts

    pos = torch.arange(Np, device=dev) - cum_unpad[sorted_e]
    dest = cum_pad[sorted_e] + pos

    nb_max = (Np + BM - 1) // BM + num_experts
    EM = nb_max * BM

    sorted_tok = torch.full((EM,), -1, dtype=torch.int32, device=dev)
    sorted_tok[dest] = torch.div(order, topk, rounding_mode="floor").to(torch.int32)

    sorted_w = torch.zeros((EM,), dtype=torch.float32, device=dev)
    sorted_w[dest] = tw_f32.reshape(-1)[order]

    blk_start = torch.arange(nb_max, device=dev) * BM
    expert_ids = torch.searchsorted(cum_pad_end, blk_start, right=True).to(torch.int32)
    expert_ids = torch.where(expert_ids >= num_experts,
                             torch.full_like(expert_ids, -1), expert_ids)

    return sorted_tok, sorted_w, expert_ids, nb_max, EM


# (BLOCK_M, BN1, BK1, warps1, stages1, GROUP_M1,
#           BN2, BK2, warps2, stages2, GROUP_M2)
DEFAULT_CFG = (128, 128, 64, 8, 2, 1, 128, 64, 8, 2, 1)
CFG_TABLE = {}


def pick_cfg(T):
    for lim in sorted(CFG_TABLE):
        if T <= lim:
            return CFG_TABLE[lim]
    return DEFAULT_CFG


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate_proj_weights: torch.Tensor,
    expert_up_proj_weights: torch.Tensor,
    expert_down_proj_weights: torch.Tensor,
):
    bs, sl, H = hidden_states.shape
    T = bs * sl
    TOPK = 8
    I = expert_gate_proj_weights.shape[1]

    x = hidden_states.reshape(T, H)

    router_logits = torch.mm(x, gate_weight.t())
    rw = torch.softmax(router_logits.float(), dim=1)
    tw, sel = torch.topk(rw, TOPK, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    tw = tw.to(torch.bfloat16).to(torch.float32)

    (BM, BN1, BK1, W1, S1, G1, BN2, BK2, W2, S2, G2) = pick_cfg(T)

    sorted_tok, sorted_w, expert_ids, nb_max, EM = plan(sel.to(torch.int32), tw, BM)

    inter = torch.empty((EM, I), dtype=torch.bfloat16, device=x.device)
    npn1 = I // BN1
    _gate_up_swiglu[(nb_max * npn1,)](
        x, expert_gate_proj_weights, expert_up_proj_weights, inter,
        sorted_tok, expert_ids,
        H, I,
        x.stride(0), x.stride(1),
        expert_gate_proj_weights.stride(0), expert_gate_proj_weights.stride(1),
        expert_gate_proj_weights.stride(2),
        inter.stride(0), inter.stride(1),
        nb_max, npn1,
        BLOCK_M=BM, BLOCK_N=BN1, BLOCK_K=BK1, GROUP_M=G1,
        num_warps=W1, num_stages=S1,
    )

    outf = torch.zeros((T, H), dtype=torch.float32, device=x.device)
    npn2 = H // BN2
    _down_scatter[(nb_max * npn2,)](
        inter, expert_down_proj_weights, outf,
        sorted_tok, sorted_w, expert_ids,
        I, H,
        inter.stride(0), inter.stride(1),
        expert_down_proj_weights.stride(0), expert_down_proj_weights.stride(1),
        expert_down_proj_weights.stride(2),
        outf.stride(0), outf.stride(1),
        nb_max, npn2,
        BLOCK_M=BM, BLOCK_N=BN2, BLOCK_K=BK2, GROUP_M=G2,
        num_warps=W2, num_stages=S2,
    )

    return outf.to(torch.bfloat16).reshape(bs, sl, H), router_logits
