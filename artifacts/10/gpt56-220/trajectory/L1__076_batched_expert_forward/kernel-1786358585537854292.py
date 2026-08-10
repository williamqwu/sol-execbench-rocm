import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """
    Batched MoE expert forward pass.
    
    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        routing_weights: (batch_size * seq_len, num_experts)
        gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
        gate_up_proj_bias: (num_experts, 2 * expert_dim)
        down_proj: (num_experts, expert_dim, hidden_size)
        down_proj_bias: (num_experts, hidden_size)
        alpha: scalar for custom activation
        limit: clamping limit
    
    Returns:
        output: (batch_size, seq_len, hidden_size)
    """
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    num_experts = gate_up_proj.shape[0]
    expert_dim = down_proj.shape[1]
    
    # Flatten batch and sequence dimensions
    # Shape: (batch_size * seq_len, hidden_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    
    # Repeat inputs for all experts
    # Shape: (num_experts * batch_size * seq_len, hidden_size)
    # All experts consume the same tokens.  A zero-stride batch expansion lets
    # bmm broadcast that matrix without first materializing 128 copies.
    hidden_batched = hidden_flat.unsqueeze(0).expand(num_experts, -1, -1)
    
    # Batched gate_up projection
    # bmm: (num_experts, tokens, hidden_size) x (num_experts, hidden_size, 2*expert_dim)
    # Output shape: (num_experts, tokens, 2*expert_dim)
    gate_up = torch.bmm(hidden_batched, gate_up_proj)
    gate_up.add_(gate_up_proj_bias.unsqueeze(1))
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    
    # Batched down projection
    # bmm: (num_experts, tokens, expert_dim) x (num_experts, expert_dim, hidden_size)
    # Output shape: (num_experts, tokens, hidden_size)
    expert_outputs = torch.bmm(gated_output, down_proj)
    expert_outputs.add_(down_proj_bias.unsqueeze(1))
    
    # Reshape for routing weight application
    # Shape: (num_experts, batch_size, seq_len, hidden_size)
    expert_outputs = expert_outputs.view(num_experts, batch_size, seq_len, hidden_size)
    
    # Apply routing weights and sum across experts
    # routing_weights shape: (batch_size * seq_len, num_experts)
    # Reshape to: (num_experts, batch_size, seq_len, 1)
    routing_weights_reshaped = routing_weights.transpose(0, 1).view(
        num_experts, batch_size, seq_len, 1
    )
    
    # Weighted sum across experts
    # Shape: (batch_size, seq_len, hidden_size)
    expert_outputs.mul_(routing_weights_reshaped)
    output = expert_outputs.sum(dim=0)
    
    return output
