import torch
import triton
import triton.language as tl


@triton.jit
def _moe_pre_kernel(
    routing_probs_ptr,
    selected_experts_ptr,
    routing_weights_sum_ptr,
    grad_routing_weights_ptr,
    grad_expert_mask_ptr,
    grad_router_logits_ptr,
    out_ptr,
    N, E, K,
    stride_rp_n, stride_rp_e,
    stride_se_n, stride_se_k,
    stride_rws_n,
    stride_grw_n, stride_grw_k,
    stride_gem_e, stride_gem_k, stride_gem_n,
    stride_grl_n, stride_grl_e,
    stride_out_n, stride_out_e,
    BLOCK_N: tl.constexpr,
    E_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    e_offsets = tl.arange(0, E_BLOCK)  # [E]
    k_offsets = tl.arange(0, K_BLOCK)  # [K]

    rp_ptrs = routing_probs_ptr + n_offsets[:, None] * stride_rp_n + e_offsets[None, :] * stride_rp_e
    rp = tl.load(rp_ptrs, mask=n_mask[:, None], other=0.0)

    grl_ptrs = grad_router_logits_ptr + n_offsets[:, None] * stride_grl_n + e_offsets[None, :] * stride_grl_e
    grp = tl.load(grl_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    rws = tl.load(routing_weights_sum_ptr + n_offsets * stride_rws_n, mask=n_mask, other=1.0)
    inv_sum = 1.0 / rws

    grw_ptrs = grad_routing_weights_ptr + n_offsets[:, None] * stride_grw_n + k_offsets[None, :] * stride_grw_k
    grw = tl.load(grw_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    se_ptrs = selected_experts_ptr + n_offsets[:, None] * stride_se_n + k_offsets[None, :] * stride_se_k
    se = tl.load(se_ptrs, mask=n_mask[:, None], other=0)

    rwu_ptrs = routing_probs_ptr + n_offsets[:, None] * stride_rp_n + se * stride_rp_e
    rwu = tl.load(rwu_ptrs, mask=n_mask[:, None], other=0.0)

    grad_sum = tl.sum(grw * rwu * inv_sum[:, None], axis=1)
    grw_unnorm = (grw - grad_sum[:, None]) * inv_sum[:, None]

    # Expert-mask: grad_from_mask[n,k] = grad_expert_mask[se[n,k], k, n]
    gem_ptrs = (grad_expert_mask_ptr
                + se * stride_gem_e
                + k_offsets[None, :] * stride_gem_k
                + n_offsets[:, None] * stride_gem_n)
    gfm = tl.load(gem_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    # Scatter-add both contributions: for each expert e, sum over k where se==e.
    # Build [BLOCK_N, E] accumulation via broadcasted mask.
    scatter_val = grw_unnorm + gfm  # [BLOCK_N, K]
    # se_expert: [BLOCK_N, K], e_offsets: [1, E]
    mask = (se == e_offsets[None, :])  # [BLOCK_N, K, E]? need careful broadcast
    # Use sum over k: for each (n, e), sum scatter_val[n,k] where se[n,k]==e
    contrib = tl.where(se[:, :, None] == e_offsets[None, None, :], scatter_val[:, :, None], 0.0)
    scatter = tl.sum(contrib, axis=1)  # [BLOCK_N, E]
    grp = grp + scatter

    # Softmax backward
    dot = tl.sum(grp * rp, axis=1)
    grad_logits = rp * (grp - dot[:, None])

    out_ptrs = out_ptr + n_offsets[:, None] * stride_out_n + e_offsets[None, :] * stride_out_e
    tl.store(out_ptrs, grad_logits.to(tl.bfloat16), mask=n_mask[:, None])


def _fused_pre_triton(
    routing_probs, selected_experts, routing_weights_sum,
    grad_routing_weights, grad_expert_mask, grad_router_logits,
    out,
):
    N, E = routing_probs.shape
    K = selected_experts.shape[1]
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_N),)
    _moe_pre_kernel[grid](
        routing_probs, selected_experts, routing_weights_sum,
        grad_routing_weights, grad_expert_mask, grad_router_logits,
        out,
        N, E, K,
        routing_probs.stride(0), routing_probs.stride(1),
        selected_experts.stride(0), selected_experts.stride(1),
        routing_weights_sum.stride(0),
        grad_routing_weights.stride(0), grad_routing_weights.stride(1),
        grad_expert_mask.stride(0), grad_expert_mask.stride(1), grad_expert_mask.stride(2),
        grad_router_logits.stride(0), grad_router_logits.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, E_BLOCK=128, K_BLOCK=8,
    )


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
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = routing_probs.shape
    out = torch.empty(num_tokens, num_experts, dtype=hidden_states.dtype, device=hidden_states.device)
    _fused_pre_triton(
        routing_probs, selected_experts, routing_weights_sum,
        grad_routing_weights, grad_expert_mask, grad_router_logits,
        out,
    )
    grad_hidden_states = torch.matmul(out, gate_weight)
    grad_gate_weight = torch.matmul(hidden_states.t(), out).t()
    return grad_hidden_states, grad_gate_weight
