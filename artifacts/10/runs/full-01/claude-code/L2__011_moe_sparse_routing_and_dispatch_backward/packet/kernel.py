import torch
import triton
import triton.language as tl


@triton.jit
def _router_core(
    rp_ptr, se_ptr, rws_ptr, grw_ptr, gem_ptr, grl_ptr,
    on, mn,
    s_rp_n, s_se_n, s_grw_n, s_gem_e, s_gem_k, s_gem_n, s_grl_n,
    E: tl.constexpr, K: tl.constexpr,
):
    """Shared body: computes grad_router_logits_computed for a block of rows.

    Reproduces the reference's op order exactly (f32 accumulation, bf16 rounding
    only at the final store), but reads grad_expert_mask via a direct strided
    gather instead of materialising the permuted E*K*N float32 tensor.
    """
    oe = tl.arange(0, E)
    ok = tl.arange(0, K)

    rp = tl.load(rp_ptr + on[:, None] * s_rp_n + oe[None, :], mask=mn[:, None], other=0.0)
    rws = tl.load(rws_ptr + on, mask=mn, other=1.0)
    se = tl.load(se_ptr + on[:, None] * s_se_n + ok[None, :], mask=mn[:, None], other=0)
    grw = tl.load(
        grw_ptr + on[:, None] * s_grw_n + ok[None, :], mask=mn[:, None], other=0.0
    ).to(tl.float32)

    # routing_weights_unnorm = gather(routing_probs, 1, selected_experts)
    rwu = tl.load(rp_ptr + on[:, None] * s_rp_n + se, mask=mn[:, None], other=0.0)

    # normalization backward (quotient rule), same order as the reference
    gsum = tl.sum(grw * rwu / rws[:, None], axis=1)
    gu = grw / rws[:, None] - (gsum / rws)[:, None]

    # grad_from_mask[n,k] = grad_expert_mask[se[n,k], k, n]
    gem = tl.load(
        gem_ptr + se * s_gem_e + ok[None, :] * s_gem_k + on[:, None] * s_gem_n,
        mask=mn[:, None], other=0.0,
    ).to(tl.float32)

    # scatter_ (set) then scatter_add_ (add) over the same unique top-k indices
    # collapses to a single scatter of the sum.
    c = gu + gem

    acc = tl.zeros([on.shape[0], E], dtype=tl.float32)
    for k in range(K):
        ks = ok == k
        sk = tl.sum(tl.where(ks[None, :], se, 0), axis=1)
        ck = tl.sum(tl.where(ks[None, :], c, 0.0), axis=1)
        acc += tl.where(oe[None, :] == sk[:, None], ck[:, None], 0.0)

    grl = tl.load(
        grl_ptr + on[:, None] * s_grl_n + oe[None, :], mask=mn[:, None], other=0.0
    ).to(tl.float32)
    grp = acc + grl

    # softmax backward
    dot = tl.sum(grp * rp, axis=1)
    return rp * (grp - dot[:, None])


@triton.jit
def _fused_kernel(
    rp_ptr, se_ptr, rws_ptr, grw_ptr, gem_ptr, grl_ptr, gw_ptr,
    g_ptr, ghs_ptr, N, H,
    s_rp_n, s_se_n, s_grw_n, s_gem_e, s_gem_k, s_gem_n, s_grl_n,
    s_gw_e, s_g_n, s_ghs_n,
    E: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BH: tl.constexpr,
):
    """Router backward fused with grad_hidden_states = g @ gate_weight.

    g stays in registers between the two, so the N*E intermediate is never
    round-tripped through HBM for the first matmul.
    """
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    mn = on < N

    g32 = _router_core(rp_ptr, se_ptr, rws_ptr, grw_ptr, gem_ptr, grl_ptr,
                       on, mn, s_rp_n, s_se_n, s_grw_n,
                       s_gem_e, s_gem_k, s_gem_n, s_grl_n, E=E, K=K)
    g = g32.to(tl.bfloat16)

    # g is still needed by the second matmul. Store it already transposed:
    # torch.mm is materially faster with a contiguous [E, N] LHS than with a
    # transposed view of an [N, E] buffer.
    oe = tl.arange(0, E)
    tl.store(g_ptr + oe[:, None] * s_g_n + on[None, :], tl.trans(g), mask=mn[None, :])

    for h0 in range(0, H, BH):
        oh = h0 + tl.arange(0, BH)
        b = tl.load(gw_ptr + oe[:, None] * s_gw_e + oh[None, :]).to(tl.bfloat16)
        r = tl.dot(g, b)
        tl.store(ghs_ptr + on[:, None] * s_ghs_n + oh[None, :],
                 r.to(tl.bfloat16), mask=mn[:, None])


@triton.jit
def _fused2d_kernel(
    rp_ptr, se_ptr, rws_ptr, grw_ptr, gem_ptr, grl_ptr, gw_ptr,
    g_ptr, ghs_ptr, N, H,
    s_rp_n, s_se_n, s_grw_n, s_gem_e, s_gem_k, s_gem_n, s_grl_n,
    s_gw_e, s_g_n, s_ghs_n,
    E: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BH: tl.constexpr,
):
    """Same fusion, but with the H axis promoted into the grid.

    At small N the 1-D version launches too few programs to fill 256 CUs and is
    latency-bound, not bandwidth-bound. Splitting H across the grid recomputes
    the router per h-block -- pure ALU on data already in cache -- to buy
    occupancy, which is the better trade below ~4K tokens.
    """
    pn = tl.program_id(0)
    ph = tl.program_id(1)
    on = pn * BN + tl.arange(0, BN)
    mn = on < N

    g32 = _router_core(rp_ptr, se_ptr, rws_ptr, grw_ptr, gem_ptr, grl_ptr,
                       on, mn, s_rp_n, s_se_n, s_grw_n,
                       s_gem_e, s_gem_k, s_gem_n, s_grl_n, E=E, K=K)
    g = g32.to(tl.bfloat16)

    oe = tl.arange(0, E)
    if ph == 0:
        tl.store(g_ptr + oe[:, None] * s_g_n + on[None, :], tl.trans(g), mask=mn[None, :])

    oh = ph * BH + tl.arange(0, BH)
    b = tl.load(gw_ptr + oe[:, None] * s_gw_e + oh[None, :]).to(tl.bfloat16)
    r = tl.dot(g, b)
    tl.store(ghs_ptr + on[:, None] * s_ghs_n + oh[None, :],
             r.to(tl.bfloat16), mask=mn[:, None])


@triton.jit
def _router_only_kernel(
    rp_ptr, se_ptr, rws_ptr, grw_ptr, gem_ptr, grl_ptr, g_ptr, N,
    s_rp_n, s_se_n, s_grw_n, s_gem_e, s_gem_k, s_gem_n, s_grl_n, s_g_n,
    E: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    on = pid * BN + tl.arange(0, BN)
    mn = on < N
    g32 = _router_core(rp_ptr, se_ptr, rws_ptr, grw_ptr, gem_ptr, grl_ptr,
                       on, mn, s_rp_n, s_se_n, s_grw_n,
                       s_gem_e, s_gem_k, s_gem_n, s_grl_n, E=E, K=K)
    tl.store(g_ptr + on[:, None] * s_g_n + tl.arange(0, E)[None, :],
             g32.to(g_ptr.dtype.element_ty), mask=mn[:, None])


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    router_logits: torch.Tensor,
    routing_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights_sum: torch.Tensor,
    grad_routing_weights: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_router_logits: torch.Tensor,
):
    N, H = hidden_states.shape
    E = gate_weight.shape[0]
    K = selected_experts.shape[1]
    dev = hidden_states.device

    common = (
        routing_probs, selected_experts, routing_weights_sum,
        grad_routing_weights, grad_expert_mask, grad_router_logits,
    )
    strides = (
        routing_probs.stride(0), selected_experts.stride(0),
        grad_routing_weights.stride(0),
        grad_expert_mask.stride(0), grad_expert_mask.stride(1),
        grad_expert_mask.stride(2), grad_router_logits.stride(0),
    )

    # gate_weight must be row-major contiguous for the fused tl.dot path.
    use_fused = N <= 16384 and gate_weight.stride(1) == 1 and H % 128 == 0

    if use_fused:
        # gT is [E, N] contiguous -- see the store in the fused kernels.
        g = torch.empty((E, N), dtype=hidden_states.dtype, device=dev)
        grad_hidden_states = torch.empty((N, H), dtype=hidden_states.dtype, device=dev)
        args = (
            *common, gate_weight, g, grad_hidden_states, N, H,
            *strides, gate_weight.stride(0), g.stride(0),
            grad_hidden_states.stride(0),
        )
        # Crossover measured on MI355X: below ~6K tokens the 1-D grid cannot
        # fill 256 CUs and the 2-D split wins; above it the redundant router
        # recompute dominates and the 1-D grid wins.
        if N <= 6144:
            if N <= 2560:
                BN, BH, nw = 32, 128, 4
            else:
                BN, BH, nw = 64, 256, 8
            BH = min(BH, H)
            _fused2d_kernel[(triton.cdiv(N, BN), H // BH)](
                *args, E=E, K=K, BN=BN, BH=BH, num_warps=nw,
            )
        else:
            BN, BH, nw = (32, 256, 4) if N <= 8192 else (64, 128, 4)
            BH = min(BH, H)
            _fused_kernel[(triton.cdiv(N, BN),)](
                *args, E=E, K=K, BN=BN, BH=BH, num_warps=nw,
            )
    else:
        g = torch.empty((N, E), dtype=hidden_states.dtype, device=dev)
        BN, nw = (8, 8) if N <= 2048 else (32, 4)
        _router_only_kernel[(triton.cdiv(N, BN),)](
            *common, g, N, *strides, g.stride(0),
            E=E, K=K, BN=BN, num_warps=nw,
        )
        grad_hidden_states = torch.matmul(g, gate_weight)
        return grad_hidden_states, torch.matmul(g.t(), hidden_states)

    grad_gate_weight = torch.matmul(g, hidden_states)
    return grad_hidden_states, grad_gate_weight
