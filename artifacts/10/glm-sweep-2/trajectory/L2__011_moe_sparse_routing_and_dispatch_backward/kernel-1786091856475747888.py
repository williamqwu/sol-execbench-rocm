import torch


@torch.compile(dynamic=True)
def _full(
    hidden_states,
    gate_weight,
    routing_probs,
    selected_experts,
    routing_weights_sum,
    grad_routing_weights,
    grad_expert_mask,
    grad_router_logits,
    arange_t,
    arange_k,
):
    input_dtype = hidden_states.dtype

    grad_routing_probs = grad_router_logits.float()
    grad_routing_weights_f32 = grad_routing_weights.float()
    routing_weights_unnorm = torch.gather(routing_probs, 1, selected_experts)
    inv_sum = 1.0 / routing_weights_sum
    grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm * inv_sum).sum(-1, keepdim=True)
    grad_routing_weights_unnorm = (grad_routing_weights_f32 - grad_sum) * inv_sum
    grad_routing_probs = grad_routing_probs.scatter_add(1, selected_experts, grad_routing_weights_unnorm)

    # Expert-mask path via flat indexing on the contiguous tensor:
    # grad_from_mask[t,k] = grad_expert_mask[selected_experts[t,k], k, t]
    num_tokens = hidden_states.shape[0]
    top_k = selected_experts.shape[1]
    flat_idx = (selected_experts * (top_k * num_tokens) + arange_k * num_tokens + arange_t).reshape(-1)
    grad_from_mask = grad_expert_mask.reshape(-1).float()[flat_idx].reshape(num_tokens, top_k)
    grad_routing_probs = grad_routing_probs.scatter_add(1, selected_experts, grad_from_mask)

    dot_product = (grad_routing_probs * routing_probs).sum(1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)

    grad_hidden_states = torch.matmul(grad_router_logits_computed, gate_weight)
    grad_gate_weight = torch.matmul(grad_router_logits_computed.t(), hidden_states)
    return grad_hidden_states, grad_gate_weight


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
    num_tokens = hidden_states.shape[0]
    top_k = selected_experts.shape[1]
    arange_t = torch.arange(num_tokens, device=hidden_states.device).view(num_tokens, 1)
    arange_k = torch.arange(top_k, device=hidden_states.device).view(1, top_k)
    return _full(
        hidden_states,
        gate_weight,
        routing_probs,
        selected_experts,
        routing_weights_sum,
        grad_routing_weights,
        grad_expert_mask,
        grad_router_logits,
        arange_t,
        arange_k,
    )
