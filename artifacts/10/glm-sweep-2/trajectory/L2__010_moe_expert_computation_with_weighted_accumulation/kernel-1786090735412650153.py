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

    # expert_mask: [num_experts, num_experts_per_tok, batch_seq_len]
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

        current_state = hidden_states[top_x]  # bf16

        # SwiGLU in bf16 via MFMA
        gate_output = torch.matmul(current_state, gate_w.t())  # [n, intermediate]
        gate_activated = F.silu(gate_output)
        up_output = torch.matmul(current_state, up_w.t())
        intermediate = gate_activated * up_output
        expert_output = torch.matmul(intermediate, down_w.t())  # [n, hidden]

        weighted_output = expert_output * routing_weights[top_x, idx, None].to(hidden_states.dtype)
        final_hidden_states.index_add_(0, top_x, weighted_output)

    return final_hidden_states
