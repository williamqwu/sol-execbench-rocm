import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1:  for each (token,expert) pair row  ->  silu(x @ Wg^T) * (x @ Wu^T)
# ---------------------------------------------------------------------------
@triton.jit
def _moe_gemm1(
    A,          # [T, H]      bf16   hidden states (flat)
    G,          # [E, I, H]   bf16   gate weights
    U,          # [E, I, H]   bf16   up weights
    C,          # [P, I]      bf16   intermediate out
    SID,        # [NPM*BM]    int32  sorted pair ids (sentinel = P)
    EID,        # [NPM]       int32  expert per m-block
    P, H, I,
    s_am, s_we, s_wn, s_cm,
    num_pid_m, num_pid_n,
    TOPK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_in_group = GROUP_M * num_pid_n
    gid = pid // num_in_group
    first_m = gid * GROUP_M
    gsize = min(num_pid_m - first_m, GROUP_M)
    r = pid % num_in_group
    pid_m = first_m + (r % gsize)
    pid_n = r // gsize

    offs_row = pid_m * BM + tl.arange(0, BM)
    first_pair = tl.load(SID + pid_m * BM)
    if first_pair >= P:
        return

    pair = tl.load(SID + offs_row)
    valid = pair < P
    tok = (pair // TOPK).to(tl.int64)
    e = tl.load(EID + pid_m)

    offs_n = (pid_n * BN + tl.arange(0, BN)) % I
    offs_k = tl.arange(0, BK)

    a_ptrs = A + tok[:, None] * s_am + offs_k[None, :]
    wbase = e.to(tl.int64) * s_we + offs_n[None, :].to(tl.int64) * s_wn + offs_k[:, None]
    g_ptrs = G + wbase
    u_ptrs = U + wbase

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(0, tl.cdiv(H, BK)):
        a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
        bg = tl.load(g_ptrs)
        bu = tl.load(u_ptrs)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        a_ptrs += BK
        g_ptrs += BK
        u_ptrs += BK

    out = (acc_g * tl.sigmoid(acc_g)) * acc_u

    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = C + pair[:, None].to(tl.int64) * s_cm + offs_cn[None, :]
    tl.store(c_ptrs, out.to(C.dtype.element_ty),
             mask=valid[:, None] & (offs_cn[None, :] < I))


# ---------------------------------------------------------------------------
# Kernel 2:  y_pair = (inter @ Wd^T) * routing_weight
# ---------------------------------------------------------------------------
@triton.jit
def _moe_gemm2(
    Cin,        # [P, I]      bf16
    D,          # [E, H, I]   bf16
    Y,          # [P, H]      bf16
    RW,         # [P]         fp32
    SID, EID,
    P, H, I,
    s_cm, s_de, s_dn, s_ym,
    num_pid_m, num_pid_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_in_group = GROUP_M * num_pid_n
    gid = pid // num_in_group
    first_m = gid * GROUP_M
    gsize = min(num_pid_m - first_m, GROUP_M)
    r = pid % num_in_group
    pid_m = first_m + (r % gsize)
    pid_n = r // gsize

    offs_row = pid_m * BM + tl.arange(0, BM)
    first_pair = tl.load(SID + pid_m * BM)
    if first_pair >= P:
        return

    pair = tl.load(SID + offs_row)
    valid = pair < P
    e = tl.load(EID + pid_m)

    offs_n = (pid_n * BN + tl.arange(0, BN)) % H
    offs_k = tl.arange(0, BK)

    a_ptrs = Cin + pair[:, None].to(tl.int64) * s_cm + offs_k[None, :]
    d_ptrs = D + e.to(tl.int64) * s_de + offs_n[None, :].to(tl.int64) * s_dn + offs_k[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(I, BK)):
        a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
        b = tl.load(d_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        d_ptrs += BK

    w = tl.load(RW + pair, mask=valid, other=0.0)
    acc = acc * w[:, None]

    offs_yn = pid_n * BN + tl.arange(0, BN)
    y_ptrs = Y + pair[:, None].to(tl.int64) * s_ym + offs_yn[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty),
             mask=valid[:, None] & (offs_yn[None, :] < H))


def _align(expert_flat, P, E, BM, num_blocks):
    """Counting-sort pairs by expert, pad each expert's run to a BM multiple."""
    dev = expert_flat.device
    order = torch.argsort(expert_flat, stable=True)
    exp_sorted = expert_flat[order]
    cnt = torch.bincount(expert_flat, minlength=E)
    pad_cnt = (cnt + (BM - 1)) // BM * BM
    cum = torch.cumsum(cnt, 0)
    cum_pad = torch.cumsum(pad_cnt, 0)
    start = cum - cnt
    start_pad = cum_pad - pad_cnt
    rank = torch.arange(P, device=dev) - start[exp_sorted]
    dst = start_pad[exp_sorted] + rank

    sid = torch.full((num_blocks * BM,), P, dtype=torch.int32, device=dev)
    sid[dst] = order.to(torch.int32)

    blk_start = torch.arange(num_blocks, device=dev) * BM
    eid = torch.searchsorted(cum_pad, blk_start, right=True).clamp_(max=E - 1)
    return sid, eid.to(torch.int32)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    bsz, slen, H = hidden_states.shape
    E, I, _ = gate_proj_weights.shape
    T = bsz * slen
    TOPK = selected_experts.shape[1]
    P = T * TOPK
    dev = hidden_states.device

    a = hidden_states.reshape(T, H)
    rw = routing_weights.reshape(-1).contiguous()
    expert_flat = selected_experts.reshape(-1)

    # NOTE: BM=64 miscompiles in _moe_gemm2 on gfx950 (nondeterministic NaNs /
    # wrong values with num_stages=2); BM=128 is correct on every shape tested.
    BM = globals().get("FORCE_BM", 128)

    num_blocks = triton.cdiv(P, BM) + E
    sid, eid = _align(expert_flat, P, E, BM, num_blocks)

    inter = torch.empty((P, I), dtype=torch.bfloat16, device=dev)

    BN1, BK1 = 128, 64
    npn1 = triton.cdiv(I, BN1)
    grid1 = (num_blocks * npn1,)
    _moe_gemm1[grid1](
        a, gate_proj_weights, up_proj_weights, inter, sid, eid,
        P, H, I,
        a.stride(0), gate_proj_weights.stride(0), gate_proj_weights.stride(1),
        inter.stride(0),
        num_blocks, npn1,
        TOPK=TOPK, BM=BM, BN=BN1, BK=BK1, GROUP_M=8,
        num_warps=4, num_stages=1,
    )

    y = torch.empty((P, H), dtype=torch.bfloat16, device=dev)
    BN2, BK2 = 128, 64
    npn2 = triton.cdiv(H, BN2)
    grid2 = (num_blocks * npn2,)
    _moe_gemm2[grid2](
        inter, down_proj_weights, y, rw, sid, eid,
        P, H, I,
        inter.stride(0), down_proj_weights.stride(0), down_proj_weights.stride(1),
        y.stride(0),
        num_blocks, npn2,
        BM=BM, BN=BN2, BK=BK2, GROUP_M=8,
        num_warps=4, num_stages=2,
    )

    yv = y.view(T, TOPK, H)
    if TOPK == 2:
        out = yv[:, 0] + yv[:, 1]
    else:
        out = yv.sum(dim=1)
    return out.view(bsz, slen, H)
