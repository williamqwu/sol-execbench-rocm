import torch

_RUN = None

def _make():
    @torch.compile(fullgraph=False, dynamic=True)
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

        hidden_flat = hidden_states.reshape(-1, hidden_size)
        hidden_batched = hidden_flat.unsqueeze(0).expand(num_experts, -1, hidden_size)

        gate_up = torch.bmm(hidden_batched, gate_up_proj)
        gate_up = gate_up + gate_up_proj_bias.unsqueeze(1)

        gate = gate_up[..., ::2].clamp(max=limit)
        up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
        gated_output = (up + 1) * (gate * torch.sigmoid(gate * alpha))

        expert_outputs = torch.bmm(gated_output, down_proj)
        expert_outputs = expert_outputs + down_proj_bias.unsqueeze(1)

        expert_outputs = expert_outputs.view(num_experts, batch_size, seq_len, hidden_size)
        routing_weights_reshaped = routing_weights.transpose(0, 1).view(
            num_experts, batch_size, seq_len, 1
        )
        output = (expert_outputs * routing_weights_reshaped).sum(dim=0)
        return output
    return run

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
    global _RUN
    if _RUN is None:
        _RUN = _make()
    return _RUN(hidden_states, routing_weights, gate_up_proj, gate_up_proj_bias,
                down_proj, down_proj_bias, alpha, limit)
