import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
    shared_gate_proj_weight: torch.Tensor,
    shared_up_proj_weight: torch.Tensor,
    shared_down_proj_weight: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """
    Complete MoE layer forward pass.
    
    1. Router: Sigmoid-based hierarchical group selection for top-8 experts from 160
    2. Routed Experts: 160 experts with SwiGLU activation
    3. Weighted Aggregation: Normalized routing weights scaled by routed_scaling_factor
    4. Shared Expert: Single expert processing all tokens
    5. Output: routed_output + shared_output
    """
    # Constants
    n_routed_experts = 160
    num_experts_per_tok = 8
    n_group = 1
    topk_group = 1
    
    batch_seq_len = hidden_states.shape[0]
    
    # Step 1: Compute routing
    # Router forward: [batch_seq_len, n_routed_experts]
    router_logits = F.linear(hidden_states, router_weight).float()
    scores = torch.sigmoid(router_logits)  # [batch_seq_len, 160]
    
    # Apply expert score correction bias
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    
    # n_group == topk_group == 1, so the sole group is always selected and
    # group scoring/masking is an identity operation.
    
    # Select top-k experts within selected groups
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )
    
    # Normalize weights if configured
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    
    # Apply routing scaling factor
    topk_weights = topk_weights * routed_scaling_factor
    topk_weights_bf16 = topk_weights.to(hidden_states.dtype)
    
    # Step 2: Compute routed expert outputs
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    
    # Process each expert
    for expert_idx in range(n_routed_experts):
        token_indices, weight_indices = torch.where(topk_indices == expert_idx)
        
        if token_indices.numel() > 0:
            # Gather tokens for this expert
            expert_input = hidden_states[token_indices]  # [num_tokens, hidden_size]
            expert_weights_sel = topk_weights_bf16[token_indices, weight_indices]  # [num_tokens]

            # Expert forward: SwiGLU activation
            gate_up_output = F.linear(
                expert_input,
                torch.cat(
                    (expert_gate_projs[expert_idx], expert_up_projs[expert_idx]),
                    dim=0,
                ),
            )
            gate_output, up_output = gate_up_output.chunk(2, dim=-1)
            intermediate = F.silu(gate_output, inplace=True).mul_(up_output)
            expert_output = F.linear(intermediate, expert_down_projs[expert_idx])

            # Apply routing weights
            weighted_output = expert_output * expert_weights_sel.unsqueeze(-1)

            # Scatter-add back to final output
            final_hidden_states.index_add_(0, token_indices, weighted_output)
    
    routed_output = final_hidden_states.to(hidden_states.dtype)
    
    # Step 3: Compute shared expert output
    gate_up_output = F.linear(
        hidden_states,
        torch.cat((shared_gate_proj_weight, shared_up_proj_weight), dim=0),
    )
    gate_output, up_output = gate_up_output.chunk(2, dim=-1)
    intermediate = F.silu(gate_output, inplace=True).mul_(up_output)
    shared_output = F.linear(intermediate, shared_down_proj_weight)
    
    # Step 4: Combine outputs
    output = routed_output + shared_output

    return output
