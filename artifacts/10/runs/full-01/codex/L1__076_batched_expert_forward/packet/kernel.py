import torch


@torch.no_grad()
def _forward_flat(
    hidden_flat: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    num_experts = gate_up_proj.shape[0]

    # A zero batch stride lets rocBLAS reuse the same token matrix for every
    # expert without materializing the reference's repeated input.
    hidden_batched = hidden_flat.unsqueeze(0).expand(num_experts, -1, -1)
    gate_up = torch.baddbmm(
        gate_up_proj_bias.unsqueeze(1), hidden_batched, gate_up_proj
    )

    gate = gate_up[..., ::2].clamp(max=limit)
    up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
    gated_output = (up + 1) * (gate * torch.sigmoid(gate * alpha))

    expert_outputs = torch.baddbmm(
        down_proj_bias.unsqueeze(1), gated_output, down_proj
    )
    return (expert_outputs * routing_weights.T.unsqueeze(-1)).sum(dim=0)


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
    shape = hidden_states.shape
    output = _forward_flat(
        hidden_states.reshape(-1, shape[-1]),
        routing_weights,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        alpha,
        limit,
    )
    return output.view(shape)
