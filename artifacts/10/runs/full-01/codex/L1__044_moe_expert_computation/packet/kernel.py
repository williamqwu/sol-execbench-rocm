import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states,
    expert_ids,
    expert_weights,
    gate_proj_weights,
    up_proj_weights,
    down_proj_weights,
):
    num_experts = gate_proj_weights.shape[0]
    flat_experts = expert_ids.reshape(-1)
    _, order = torch.sort(flat_experts)
    token_indices = torch.div(order, expert_ids.shape[1], rounding_mode="floor")
    x = hidden_states[token_indices]

    # _grouped_mm expects the inclusive end offset of every expert group.
    offsets = torch.bincount(flat_experts, minlength=num_experts).cumsum(0).to(torch.int32)
    gate = torch._grouped_mm(x, gate_proj_weights.transpose(1, 2), offsets)
    up = torch._grouped_mm(x, up_proj_weights.transpose(1, 2), offsets)
    intermediate = F.silu(gate) * up
    y = torch._grouped_mm(intermediate, down_proj_weights.transpose(1, 2), offsets)

    routing = expert_weights.reshape(-1)[order, None]
    y.mul_(routing)
    output = torch.zeros_like(hidden_states)
    output.index_add_(0, token_indices, y)
    return output
