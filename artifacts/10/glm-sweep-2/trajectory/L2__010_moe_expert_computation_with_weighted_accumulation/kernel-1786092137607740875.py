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

    # Preprocess weights once: transpose to [hidden, intermediate]/[intermediate,hidden]
    # and cast to fp32 so per-expert matmuls need no cast/transpose/copy.
    # gate/up weights are [num_experts, intermediate, hidden] -> want [num_experts, hidden, intermediate]
    gate_w = gate_proj_weights.transpose(1, 2).to(torch.float32)  # [E, hidden, intermediate]
    up_w = up_proj_weights.transpose(1, 2).to(torch.float32)
    # down weights are [num_experts, hidden, intermediate] -> want [E, intermediate, hidden]
    down_w = down_proj_weights.transpose(1, 2).to(torch.float32)  # [E, intermediate, hidden]

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()

        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        if top_x.numel() == 0:
            continue

        current_state = hidden_states[top_x].to(torch.float32)  # [n, hidden]

        gate_output = torch.matmul(current_state, gate_w[expert_idx])      # [n, intermediate]
        gate_activated = F.silu(gate_output)
        up_output = torch.matmul(current_state, up_w[expert_idx])          # [n, intermediate]
        intermediate = gate_activated * up_output
        expert_output = torch.matmul(intermediate, down_w[expert_idx])     # [n, hidden]

        weighted_output = expert_output * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, weighted_output.to(hidden_states.dtype))

    return final_hidden_states
