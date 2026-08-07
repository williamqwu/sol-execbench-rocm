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
    batch_size, seq_len, hidden_size = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_size)

    # Broadcast matmul avoids materializing the (E*T, H) `repeat` tensor.
    gate_up = torch.matmul(hidden_flat, gate_up_proj)
    gate_up = gate_up + gate_up_proj_bias.unsqueeze(1)

    gate = gate_up[..., ::2].clamp(max=limit)
    up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
    gated_output = (up + 1) * (gate * torch.sigmoid(gate * alpha))

    expert_outputs = torch.bmm(gated_output, down_proj)
    expert_outputs = expert_outputs + down_proj_bias.unsqueeze(1)
    expert_outputs = expert_outputs.view(-1, batch_size, seq_len, hidden_size)

    rw = routing_weights.transpose(0, 1).view(-1, batch_size, seq_len, 1)
    return (expert_outputs * rw).sum(dim=0)
