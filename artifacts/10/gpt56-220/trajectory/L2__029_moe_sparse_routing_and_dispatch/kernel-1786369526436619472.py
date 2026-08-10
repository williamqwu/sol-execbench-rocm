import torch
import torch.nn.functional as F

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
    shared_intermediate = shared_silu.mul_(shared_up_out)
    shared_output = torch.matmul(shared_intermediate, shared_down_proj.t())
    
    # === Gating and Routing (float32 for stability) ===
    # Compute router logits
    router_logits = torch.matmul(hidden_states_flat.float(), gate_weight.t())
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights.add_(e_score_correction_bias)
    routing_weights_selected, selected_experts = torch.topk(
        routing_weights, top_k, dim=-1, sorted=False
    )
    routing_weights_selected.div_(torch.clamp(
        routing_weights_selected.sum(dim=-1, keepdim=True), min=norm_min
    ))
    routing_weights_normalized = routing_weights_selected.to(hidden_states.dtype)
    
    # === Token Dispatch and Expert Computation ===
    final_hidden_states = shared_output
    
    expert_ids = torch.arange(num_experts, device=selected_experts.device)[:, None, None]
    expert_mask = selected_experts.unsqueeze(0) == expert_ids
    expert_hit = expert_mask.flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
    for expert_idx_int in expert_hit.tolist():
        top_x, idx = torch.where(expert_mask[expert_idx_int])
        
        # Gather tokens for this expert
        current_state = hidden_states_flat[top_x]
        
        # Expert MLP: gate_proj + silu + up_proj + down_proj
        gate_out = torch.matmul(current_state, expert_gate_proj[expert_idx_int].t())
        silu_out = F.silu(gate_out)
        up_out = torch.matmul(current_state, expert_up_proj[expert_idx_int].t())
        intermediate = silu_out.mul_(up_out)
        expert_output = torch.matmul(intermediate, expert_down_proj[expert_idx_int].t())
        
        # Apply routing weights
        weighted_output = expert_output * routing_weights_normalized[top_x, idx, None]
        
        # Accumulate to final output (scatter-add operation)
        final_hidden_states.index_add_(0, top_x, weighted_output)
    
    # Reshape back to [batch_size, seq_len, hidden_size]
    output = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    
    return output, router_logits
