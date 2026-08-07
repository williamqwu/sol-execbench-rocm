import torch


@torch.compile(dynamic=True)
def _fused_pre(
    routing_probs,
    selected_experts,
    routing_weights_sum,
    grad_routing_weights,
    grad_expert_mask,
    grad_router_logits,
    arange_tokens,
    arange_topk,
):
    grad_routing_probs = grad_router_logits.float()

    grad_routing_weights_f32 = grad_routing_weights.float()
    routing_weights_unnorm = torch.gather(routing_probs, 1, selected_experts)
    inv_sum = 1.0 / routing_weights_sum
    grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm * inv_sum).sum(-1, keepdim=True)
    grad_routing_weights_unnorm = (grad_routing_weights_f32 - grad_sum) * inv_sum
    grad_routing_probs = grad_routing_probs.scatter_add(1, selected_experts, grad_routing_weights_unnorm)

    # Expert-mask path: direct indexing instead of permute+materialize.
    # grad_from_mask[t, k] = grad_expert_mask[selected_experts[t,k], k, t]
    grad_from_mask = grad_expert_mask[selected_experts, arange_topk, arange_tokens].float()
    grad_routing_probs = grad_routing_probs.scatter_add(1, selected_experts, grad_from_mask)

    dot_product = (grad_routing_probs * routing_probs).sum(1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    return grad_router_logits_computed


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
    input_dtype = hidden_states.dtype
    num_tokens = hidden_states.shape[0]
    top_k = selected_experts.shape[1]

    arange_tokens = torch.arange(num_tokens, device=hidden_states.device).view(num_tokens, 1)
    arange_topk = torch.arange(top_k, device=hidden_states.device).view(1, top_k)

    grad_router_logits_computed = _fused_pre(
        routing_probs,
        selected_experts,
        routing_weights_sum,
        grad_routing_weights,
        grad_expert_mask,
        grad_router_logits,
        arange_tokens,
        arange_topk,
    )
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)

    grad_hidden_states = torch.matmul(grad_router_logits_computed, gate_weight)
    grad_gate_weight = torch.matmul(grad_router_logits_computed.t(), hidden_states)

    return grad_hidden_states, grad_gate_weight
