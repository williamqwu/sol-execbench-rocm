"""Fused MoE expert-batched execution with capacity factor, for MI355X (gfx950).

Strategy
--------
The reference materialises a padded [E, capacity, H] input tensor, runs three
BMMs over it, and gathers the result back. That costs several GB of HBM
round-trips for intermediates that are never needed, and spends ~20% of its
FLOPs on zero-padding rows whose outputs are discarded.

This implementation keeps the same semantics but:
  * never materialises `expert_inputs` -- rows are gathered from
    `hidden_states` inside the GEMM prologue via a slot->token map;
  * fuses gate + up + SwiGLU into one kernel, so `gate_out`/`up_out` never
    reach HBM;
  * skips tiles past an expert's admitted-token count, so padded rows cost
    nothing;
  * fuses the routing-weight scale and the token scatter into the down-GEMM
    epilogue.

Numerics follow the reference's rounding order: the gate and up accumulators
are rounded to bfloat16 before SiLU, SiLU's result is rounded to bfloat16
before the multiply, and the product is rounded to bfloat16 before the down
projection.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Kernel 1: gathered gate/up GEMM + SwiGLU
# --------------------------------------------------------------------------
@triton.jit
def _gate_up_swiglu(
    X, WG, WU, HBUF,
    slot_token, counts,
    H, I, CAP,
    stride_xm, stride_we, stride_wk, stride_hb,
    MT: tl.constexpr, NT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    # m fastest, so blocks sharing a weight tile are launched adjacently
    mt = pid % MT
    rest = pid // MT
    nt_ = rest % NT
    e = rest // NT

    n_admit = tl.load(counts + e)
    m0 = mt * BM
    if m0 >= n_admit:
        return

    offs_m = m0 + tl.arange(0, BM)
    row_mask = offs_m < n_admit
    slots = e * CAP + offs_m
    rows = tl.load(slot_token + slots, mask=row_mask, other=0).to(tl.int32)

    offs_n = nt_ * BN + tl.arange(0, BN)
    n_mask = offs_n < I
    offs_k = tl.arange(0, BK)

    x_ptr = X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :]
    wbase = WG + e.to(tl.int64) * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :]
    ubase = WU + e.to(tl.int64) * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :]

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, tl.cdiv(H, BK)):
        kk = k * BK
        k_mask = (offs_k + kk) < H
        x = tl.load(x_ptr + kk, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        wg = tl.load(wbase + kk * stride_wk, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wu = tl.load(ubase + kk * stride_wk, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc_g = tl.dot(x, wg, acc_g)
        acc_u = tl.dot(x, wu, acc_u)

    # match reference rounding: bmm -> bf16, silu -> bf16, mul -> bf16
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16)
    h = (s * u).to(tl.bfloat16)

    out = HBUF + slots[:, None].to(tl.int64) * stride_hb + offs_n[None, :]
    tl.store(out, h, mask=row_mask[:, None] & n_mask[None, :])


# --------------------------------------------------------------------------
# Kernel 2: down GEMM + routing-weight scale + scatter-add to tokens
# --------------------------------------------------------------------------
@triton.jit
def _down_scatter(
    HBUF, WD, OUT,
    slot_token, slot_w, counts,
    H, I, CAP,
    stride_hb, stride_we, stride_wk, stride_om,
    MT: tl.constexpr, NT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    mt = pid % MT
    rest = pid // MT
    nt_ = rest % NT
    e = rest // NT

    n_admit = tl.load(counts + e)
    m0 = mt * BM
    if m0 >= n_admit:
        return

    offs_m = m0 + tl.arange(0, BM)
    row_mask = offs_m < n_admit
    slots = e * CAP + offs_m
    rows = tl.load(slot_token + slots, mask=row_mask, other=0).to(tl.int32)
    w = tl.load(slot_w + slots, mask=row_mask, other=0.0)

    offs_n = nt_ * BN + tl.arange(0, BN)
    n_mask = offs_n < H
    offs_k = tl.arange(0, BK)

    a_ptr = HBUF + slots[:, None].to(tl.int64) * stride_hb + offs_k[None, :]
    wbase = WD + e.to(tl.int64) * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(I, BK)):
        kk = k * BK
        k_mask = (offs_k + kk) < I
        a = tl.load(a_ptr + kk, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        wd = tl.load(wbase + kk * stride_wk, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, wd, acc)

    # reference: bmm result -> bf16, then bf16 weight * bf16 value
    val = acc.to(tl.bfloat16).to(tl.float32) * w[:, None]

    optr = OUT + rows[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.atomic_add(optr, val, mask=row_mask[:, None] & n_mask[None, :], sem="relaxed")


# --------------------------------------------------------------------------
# Routing metadata (no host synchronisation)
# --------------------------------------------------------------------------
def _routing(selected_experts, routing_weights, E, cap):
    nt, K = selected_experts.shape
    dev = selected_experts.device
    NK = nt * K

    flat_e = selected_experts.reshape(-1)
    sorted_e, sidx = torch.sort(flat_e, stable=True)

    counts = torch.bincount(sorted_e, minlength=E)
    starts = torch.zeros(E, dtype=torch.int64, device=dev)
    torch.cumsum(counts[:-1], 0, out=starts[1:])

    within = torch.arange(NK, device=dev) - starts[sorted_e]
    admitted = within < cap

    dump = E * cap
    slot = torch.where(admitted, sorted_e * cap + within, dump)

    slot_token = torch.zeros(dump + 1, dtype=torch.int32, device=dev)
    slot_token.scatter_(0, slot, torch.div(sidx, K, rounding_mode="floor").to(torch.int32))

    slot_w = torch.zeros(dump + 1, dtype=torch.float32, device=dev)
    slot_w.scatter_(0, slot, routing_weights.reshape(-1).float()[sidx])

    return slot_token, slot_w, counts.clamp_(max=cap).to(torch.int32)


def _cfg(cap, avg_m):
    """Block sizes. Small per-expert M is weight-bandwidth bound; large M is
    compute bound and wants wider tiles."""
    if avg_m <= 96:
        return dict(BM=64, BN=128, BK=64, warps=8, stages=2)
    if avg_m <= 192:
        return dict(BM=128, BN=128, BK=64, warps=8, stages=2)
    return dict(BM=128, BN=256, BK=64, warps=8, stages=2)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, moe_intermediate_size = expert_gate_weights.shape
    K = selected_experts.shape[1]
    dev = hidden_states.device
    H, I, E = hidden_size, moe_intermediate_size, num_experts

    cap = max(int((num_tokens * K / E) * 1.25), 1)

    hidden_states = hidden_states.contiguous()
    slot_token, slot_w, counts = _routing(selected_experts, routing_weights, E, cap)

    avg_m = num_tokens * K / E
    c = _cfg(cap, avg_m)
    BM, BN, BK = c["BM"], c["BN"], c["BK"]

    hbuf = torch.empty(E * cap, I, dtype=torch.bfloat16, device=dev)

    MT = triton.cdiv(cap, BM)
    NT = triton.cdiv(I, BN)
    _gate_up_swiglu[(E * NT * MT,)](
        hidden_states, expert_gate_weights, expert_up_weights, hbuf,
        slot_token, counts,
        H, I, cap,
        hidden_states.stride(0),
        expert_gate_weights.stride(0), expert_gate_weights.stride(1),
        hbuf.stride(0),
        MT=MT, NT=NT, BM=BM, BN=BN, BK=BK,
        num_warps=c["warps"], num_stages=c["stages"],
    )

    out = torch.zeros(num_tokens, H, dtype=torch.float32, device=dev)

    NT2 = triton.cdiv(H, BN)
    _down_scatter[(E * NT2 * MT,)](
        hbuf, expert_down_weights, out,
        slot_token, slot_w, counts,
        H, I, cap,
        hbuf.stride(0),
        expert_down_weights.stride(0), expert_down_weights.stride(1),
        out.stride(0),
        MT=MT, NT=NT2, BM=BM, BN=BN, BK=BK,
        num_warps=c["warps"], num_stages=c["stages"],
    )

    return out.to(torch.bfloat16)
