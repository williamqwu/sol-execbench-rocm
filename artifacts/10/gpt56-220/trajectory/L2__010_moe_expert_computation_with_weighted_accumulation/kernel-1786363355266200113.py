import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Hidden states
    hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    # Routing weights (normalized per token)
    routing_weights_raw = torch.rand(batch_seq_len, num_experts_per_tok, dtype=torch.float32, device=device)
    routing_weights = routing_weights_raw / routing_weights_raw.sum(dim=-1, keepdim=True)
    
    # Selected experts - must be valid indices in [0, num_experts)
    # Each token selects num_experts_per_tok unique experts
    selected_experts = torch.zeros(batch_seq_len, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(batch_seq_len):
        perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        selected_experts[i] = perm
    
    # Expert weights
    gate_proj_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    up_proj_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    down_proj_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=torch.bfloat16, device=device) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "routing_weights": routing_weights,
        "selected_experts": selected_experts,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Complete MoE expert computation with token dispatch and weighted accumulation.
    
    Args:
        hidden_states: Input tokens [batch_seq_len, hidden_size]
        routing_weights: Normalized routing weights [batch_seq_len, num_experts_per_tok]
        selected_experts: Selected expert indices [batch_seq_len, num_experts_per_tok]
        gate_proj_weights: Gate projection weights [num_experts, moe_intermediate_size, hidden_size]
        up_proj_weights: Up projection weights [num_experts, moe_intermediate_size, hidden_size]
        down_proj_weights: Down projection weights [num_experts, hidden_size, moe_intermediate_size]
    
    Returns:
        final_hidden_states: Weighted expert outputs [batch_seq_len, hidden_size]
    """
    batch_seq_len, hidden_dim = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]
    
    # Initialize output accumulator
    final_hidden_states = torch.zeros(
        (batch_seq_len, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    
    # Process experts directly from the compact top-k representation.
    for expert_idx in range(num_experts):
        
        # Get expert weights
        gate_w = gate_proj_weights[expert_idx]  # [intermediate, hidden]
        up_w = up_proj_weights[expert_idx]      # [intermediate, hidden]
        down_w = down_proj_weights[expert_idx]  # [hidden, intermediate]
        
        # Find which tokens are assigned to this expert
        top_x, idx = torch.where(selected_experts == expert_idx)
        
        if top_x.numel() == 0:
            continue
        
        # Gather tokens for this expert
        current_state = hidden_states[top_x]  # [num_tokens, hidden]
        
        # SwiGLU computation: down(silu(gate(x)) * up(x))
        gate_up_w = torch.cat((gate_w, up_w), dim=0)
        gate_output, up_output = torch.matmul(current_state, gate_up_w.t()).chunk(2, dim=-1)
        gate_activated = gate_output / (1.0 + torch.exp(-gate_output))  # SiLU
        intermediate = gate_activated * up_output
        expert_output = torch.matmul(intermediate.to(torch.bfloat16), down_w.t())  # [num_tokens, hidden]
        
        # Weight by routing weights
        weighted_output = expert_output * routing_weights[top_x, idx, None]
        
        # Scatter-add back to final output
        final_hidden_states.index_add_(0, top_x, weighted_output.to(hidden_states.dtype))
    
    return final_hidden_states
