import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def _route(hidden_states, gate_weight, correction_bias, norm_min):
    logits = torch.matmul(hidden_states.float(), gate_weight.t())
    weights = F.softmax(logits, dim=1, dtype=torch.float32) + correction_bias
    selected_weights, selected_experts = torch.topk(weights, 8, dim=-1)
    selected_weights = selected_weights / torch.clamp(
        selected_weights.sum(dim=-1, keepdim=True), min=norm_min
    )
    return logits, selected_experts, selected_weights.to(hidden_states.dtype)

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    norm_min: float,
):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_experts = 64
    top_k = 8
    
    # Flatten batch and sequence dimensions
    hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [batch*seq_len, hidden_size]
    
    # === Shared Experts (process all tokens) ===
    shared_gate_out = torch.matmul(hidden_states_flat, shared_gate_proj.t())
    shared_silu = F.silu(shared_gate_out)
    shared_up_out = torch.matmul(hidden_states_flat, shared_up_proj.t())
    shared_intermediate = shared_silu * shared_up_out
    shared_output = torch.matmul(shared_intermediate, shared_down_proj.t())
    
    # === Gating and Routing (float32 for stability) ===
    # Compute router logits
    router_logits, selected_experts, routing_weights_normalized = _route(
        hidden_states_flat, gate_weight, e_score_correction_bias, norm_min
    )
    
    # === Token Dispatch and Expert Computation ===
    final_hidden_states = shared_output.clone()
    
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).squeeze(-1)
    for expert_idx in expert_hit:
        expert_idx_int = expert_idx.item()
        idx, top_x = torch.where(expert_mask[expert_idx_int].squeeze(0))
        
        if top_x.numel() == 0:
            continue
        
        # Gather tokens for this expert
        current_state = hidden_states_flat[top_x]  # [num_tokens_for_expert, hidden_size]
        
        # Expert MLP: gate_proj + silu + up_proj + down_proj
        gate_out = torch.matmul(current_state, expert_gate_proj[expert_idx_int].t())
        silu_out = F.silu(gate_out)
        up_out = torch.matmul(current_state, expert_up_proj[expert_idx_int].t())
        intermediate = silu_out * up_out
        expert_output = torch.matmul(intermediate, expert_down_proj[expert_idx_int].t())
        
        # Apply routing weights
        weighted_output = expert_output * routing_weights_normalized[top_x, idx, None]  # [num_tokens, hidden_size]
        
        # Accumulate to final output (scatter-add operation)
        final_hidden_states.index_add_(0, top_x, weighted_output)
    
    # Reshape back to [batch_size, seq_len, hidden_size]
    output = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    
    return output, router_logits
