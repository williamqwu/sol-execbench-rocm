import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    Sparse MoE expert computation with top-k routing.
    
    1. Compute routing scores via sigmoid activation
    2. Apply group-wise expert selection (n_group=1, topk_group=1 simplifies to global topk)
    3. Select top-8 experts per token
    4. Compute SwiGLU for each selected expert
    5. Combine with shared expert output
    """
    # Constants
    n_routed_experts = 128
    num_experts_per_tok = 8
    n_group = 1
    topk_group = 1
    norm_topk_prob = True
    
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    
    # Router logits: [num_tokens, n_routed_experts]
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    
    # Sigmoid activation for scores
    scores = router_logits.sigmoid_()
    
    # Apply score correction bias
    scores_for_choice = scores.add_(e_score_correction_bias.unsqueeze(0))
    
    # Select top-k experts
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )
    
    # Normalize weights if required
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights.div_(denominator)
    
    # Apply routing scaling factor
    topk_weights.mul_(routed_scaling_factor)
    topk_weights = topk_weights.to(hidden_states.dtype)
    
    # Compute routed expert outputs
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    
    # Group the routes once so each expert consumes one contiguous range.  This
    # avoids scanning the complete [tokens, top-k] table for every expert.
    flat_experts = topk_indices.reshape(-1)
    route_order = torch.argsort(flat_experts)
    expert_counts = torch.bincount(flat_experts, minlength=n_routed_experts).cpu().tolist()
    flat_token_indices = route_order >> 3
    sorted_expert_weights = topk_weights.reshape(-1)[route_order]

    # Process each expert
    route_start = 0
    for expert_idx in range(n_routed_experts):
        route_end = route_start + expert_counts[expert_idx]
        if route_end > route_start:
            token_indices = flat_token_indices[route_start:route_end]
            # Get routing weights for this expert
            expert_weights = sorted_expert_weights[route_start:route_end]
            
            # Get input tokens for this expert
            expert_input = hidden_states[token_indices]
            
            # Expert MLP computation: SwiGLU activation
            gate_output = F.linear(expert_input, expert_gate_weights[expert_idx])
            up_output = F.linear(expert_input, expert_up_weights[expert_idx])
            intermediate = F.silu(gate_output, inplace=True).mul_(up_output)
            expert_output = F.linear(intermediate, expert_down_weights[expert_idx])
            
            # Apply routing weights
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Accumulate to final output
            final_hidden_states.index_add_(0, token_indices, weighted_output)
        route_start = route_end
    
    # Compute shared expert output
    shared_gate_output = F.linear(hidden_states, shared_gate_weight)
    shared_up_output = F.linear(hidden_states, shared_up_weight)
    shared_intermediate = F.silu(shared_gate_output, inplace=True).mul_(shared_up_output)
    shared_output = F.linear(shared_intermediate, shared_down_weight)

    # Combine routed and shared experts
    output = shared_output.add_(final_hidden_states)
    
    return output
