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
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    num_experts = gate_up_proj.shape[0]
    expert_dim = down_proj.shape[1]

    # Flatten batch+seq. Use expand (view, no copy) + matmul broadcast
    # instead of repeat (6GB copy). All experts share the same LHS.
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    hidden_batched = hidden_flat.unsqueeze(0).expand(num_experts, -1, -1)

    gate_up = torch.matmul(hidden_batched, gate_up_proj)
    gate_up = gate_up + gate_up_proj_bias.unsqueeze(1)

    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu

    expert_outputs = torch.bmm(gated_output, down_proj)
    expert_outputs = expert_outputs + down_proj_bias.unsqueeze(1)

    expert_outputs = expert_outputs.view(num_experts, batch_size, seq_len, hidden_size)
    routing_weights_reshaped = routing_weights.transpose(0, 1).view(
        num_experts, batch_size, seq_len, 1
    )
    output = (expert_outputs * routing_weights_reshaped).sum(dim=0)
    return output
