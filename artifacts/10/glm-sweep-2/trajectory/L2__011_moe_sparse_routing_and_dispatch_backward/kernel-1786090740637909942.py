import torch


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
    """
    Backward pass for MoE sparse routing.
    """
    input_dtype = hidden_states.dtype

    # --- Build grad_routing_probs (float32) in one accumulator ---
    # Start from grad_router_logits (the direct path), then scatter-add the
    # two sparse gradient paths on top. This collapses
    #   zero + scatter_ + scatter_add_ + add   ->   base + scatter_add
    grad_routing_probs = grad_router_logits.float()

    # Path 1: normalization backward
    grad_routing_weights_f32 = grad_routing_weights.float()
    routing_weights_unnorm = torch.gather(routing_probs, 1, selected_experts)
    inv_sum = 1.0 / routing_weights_sum
    grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm * inv_sum).sum(-1, keepdim=True)
    grad_routing_weights_unnorm = (grad_routing_weights_f32 - grad_sum) * inv_sum
    grad_routing_probs.scatter_add_(1, selected_experts, grad_routing_weights_unnorm)

    # Path 2: expert-mask auxiliary path
    # grad_expert_mask: (num_experts, top_k, num_tokens) -> (num_tokens, top_k, num_experts)
    grad_expert_mask_permuted = grad_expert_mask.permute(2, 1, 0).float()
    grad_from_mask = torch.gather(
        grad_expert_mask_permuted, 2, selected_experts.unsqueeze(2)
    ).squeeze(2)
    grad_routing_probs.scatter_add_(1, selected_experts, grad_from_mask)

    # --- Softmax backward ---
    dot_product = (grad_routing_probs * routing_probs).sum(1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)

    # --- Linear projection (two matmuls) ---
    grad_hidden_states = torch.matmul(grad_router_logits_computed, gate_weight)
    grad_gate_weight = torch.matmul(grad_router_logits_computed.t(), hidden_states)

    return grad_hidden_states, grad_gate_weight
