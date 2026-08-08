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

    # --- Expert mask backward ---
    # grad_expert_mask: [num_experts, top_k, num_tokens] -> permute to [num_tokens, top_k, num_experts] (view, no copy)
    # Gather on bf16 (avoids materializing huge float32 tensor), then convert small result
    grad_expert_mask_permuted = grad_expert_mask.permute(2, 1, 0)  # [num_tokens, top_k, num_experts] bf16 view
    grad_from_mask = torch.gather(
        grad_expert_mask_permuted,
        dim=2,
        index=selected_experts.unsqueeze(2),
    ).squeeze(2).float()  # [num_tokens, top_k] float32

    # --- Combine scatter sources and scatter once into grad_routing_probs ---
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
