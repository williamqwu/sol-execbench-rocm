import torch


def get_inputs(
    axes_and_scalars: dict[str, int], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MoE routing backward pass."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_experts = axes_and_scalars["num_experts"]
    top_k = axes_and_scalars["top_k"]
    
    # Hidden states from forward pass
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Gate weight
    gate_weight = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    
    # Router logits from forward pass
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device)
    
    # Routing probs (softmax of router_logits)
    routing_probs = torch.softmax(router_logits.float(), dim=1)
    
    # Selected experts (top-k indices)
    _, selected_experts = torch.topk(routing_probs, top_k, dim=-1)
    
    # Routing weights sum (for normalization backward)
    routing_weights, _ = torch.topk(routing_probs, top_k, dim=-1)
    routing_weights_sum = routing_weights.sum(dim=-1, keepdim=True)
    
    # Gradients
    grad_routing_weights = torch.randn(num_tokens, top_k, dtype=torch.bfloat16, device=device) * 0.1
    grad_expert_mask = torch.randn(num_experts, top_k, num_tokens, dtype=torch.bfloat16, device=device) * 0.01
    grad_router_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device) * 0.1
    
    return {
        "hidden_states": hidden_states,
        "gate_weight": gate_weight,
        "router_logits": router_logits,
        "routing_probs": routing_probs,
        "selected_experts": selected_experts,
        "routing_weights_sum": routing_weights_sum,
        "grad_routing_weights": grad_routing_weights,
        "grad_expert_mask": grad_expert_mask,
        "grad_router_logits": grad_router_logits,
    }


@torch.compile(dynamic=True)
def _run_compiled(
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
    
    Computes gradients through:
    1. Normalization (quotient rule)
    2. Top-k selection (straight-through estimator)
    3. Softmax
    4. Linear projection
    """
    num_tokens = hidden_states.shape[0]
    num_experts = gate_weight.shape[0]
    top_k = selected_experts.shape[1]
    input_dtype = hidden_states.dtype
    
    # Initialize with the dense direct-gradient path.
    grad_routing_probs = grad_router_logits.float()
    
    # Process grad_routing_weights through normalization backward
    grad_routing_weights_f32 = grad_routing_weights.float()
    
    # Gradient through normalization: d(w_i / sum(w)) / d(w_i)
    # Gather the routing weights before normalization
    routing_weights_unnorm = torch.gather(routing_probs, dim=1, index=selected_experts)  # [num_tokens, top_k]
    
    # Compute gradient w.r.t. unnormalized weights using quotient rule
    inv_sum = routing_weights_sum.reciprocal()
    weighted_sum = (grad_routing_weights_f32 * routing_weights_unnorm).sum(
        dim=-1, keepdim=True
    )
    grad_routing_weights_unnorm = (
        grad_routing_weights_f32 - weighted_sum * inv_sum
    ) * inv_sum
    
    # Load mask gradients directly from their [expert, k, token] layout.
    token_idx = torch.arange(num_tokens, device=selected_experts.device)[:, None]
    k_idx = torch.arange(top_k, device=selected_experts.device)[None, :]
    grad_from_mask = grad_expert_mask[selected_experts, k_idx, token_idx].float()
    
    # Combine both sparse paths and scatter once. Top-k indices are unique.
    grad_routing_probs.scatter_add_(
        dim=1,
        index=selected_experts,
        src=grad_routing_weights_unnorm + grad_from_mask
    )
    
    # Gradient through softmax
    # d(softmax)/d(logits) = softmax * (grad - sum(grad * softmax))
    dot_product = (grad_routing_probs * routing_probs).sum(dim=1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    
    # Convert back to input dtype
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)
    
    # Gradient through linear projection
    # grad_hidden_states = grad_router_logits @ gate_weight
    # Shape: (num_tokens, num_experts) @ (num_experts, hidden_size)
    grad_hidden_states = torch.matmul(
        grad_router_logits_computed,
        gate_weight
    )
    
    # grad_gate_weight = grad_router_logits.T @ hidden_states
    # Shape: (num_experts, num_tokens) @ (num_tokens, hidden_size)
    grad_gate_weight = torch.matmul(
        grad_router_logits_computed.t(),
        hidden_states
    )
    
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
    return _run_compiled(
        hidden_states, gate_weight, router_logits, routing_probs,
        selected_experts, routing_weights_sum, grad_routing_weights,
        grad_expert_mask, grad_router_logits,
    )
