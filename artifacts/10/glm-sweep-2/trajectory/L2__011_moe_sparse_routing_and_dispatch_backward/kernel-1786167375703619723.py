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
    num_tokens = hidden_states.shape[0]
    num_experts = gate_weight.shape[0]
    top_k = selected_experts.shape[1]
    input_dtype = hidden_states.dtype
    device = hidden_states.device

    # --- Normalization backward (quotient rule) ---
    grad_routing_weights_f32 = grad_routing_weights.float()
    routing_weights_unnorm = torch.gather(routing_probs, dim=1, index=selected_experts)

    grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm / routing_weights_sum).sum(
        dim=-1, keepdim=True
    )
    grad_routing_weights_unnorm = (
        grad_routing_weights_f32 / routing_weights_sum
        - grad_sum / routing_weights_sum
    )

    # --- Expert mask backward via direct indexing (avoids huge permute+float) ---
    # grad_expert_mask: [num_experts, top_k, num_tokens]
    # We want grad_from_mask[t, k] = grad_expert_mask[selected_experts[t,k], k, t]
    t_idx = torch.arange(num_tokens, device=device).unsqueeze(1)
    k_idx = torch.arange(top_k, device=device).unsqueeze(0)
    grad_from_mask = grad_expert_mask[selected_experts, k_idx, t_idx].float()

    # --- Combine and scatter into grad_routing_probs ---
    combined = grad_routing_weights_unnorm + grad_from_mask
    grad_routing_probs = grad_router_logits.float()
    grad_routing_probs.scatter_add_(dim=1, index=selected_experts, src=combined)

    # --- Softmax backward ---
    dot_product = (grad_routing_probs * routing_probs).sum(dim=1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)

    # --- Linear projection backward ---
    grad_hidden_states = torch.matmul(grad_router_logits_computed, gate_weight)
    grad_gate_weight = torch.matmul(grad_router_logits_computed.t(), hidden_states)

    return grad_hidden_states, grad_gate_weight
