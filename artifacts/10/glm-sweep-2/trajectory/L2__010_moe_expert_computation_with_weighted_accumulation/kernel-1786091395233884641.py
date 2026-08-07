import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
) -> torch.Tensor:
    batch_seq_len, hidden_dim = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]

    final_hidden_states = torch.zeros(
        (batch_seq_len, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()

        gate_w = gate_proj_weights[expert_idx]
        up_w = up_proj_weights[expert_idx]
        down_w = down_proj_weights[expert_idx]

        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        if top_x.numel() == 0:
            continue

        # gate/up in fp32 (large reduction dim 2048 -> bf16 output too lossy)
        current_state = hidden_states[top_x].to(torch.float32)
        gate_output = torch.matmul(current_state, gate_w.t().to(torch.float32))
        gate_activated = F.silu(gate_output)
        up_output = torch.matmul(current_state, up_w.t().to(torch.float32))
        intermediate = gate_activated * up_output
        # down projection: bf16 inputs (reduction dim 768, bf16 output is within tolerance)
        expert_output = torch.matmul(intermediate.to(torch.bfloat16), down_w.t())
        weighted_output = expert_output.to(torch.float32) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, weighted_output.to(hidden_states.dtype))

    return final_hidden_states
