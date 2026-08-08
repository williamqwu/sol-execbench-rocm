import torch
import torch.nn.functional as F


# Fixed constants for gated GLU activation
ALPHA = 1.702
LIMIT = 7.0


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
) -> torch.Tensor:
    alpha = ALPHA
    limit = LIMIT
    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_up_proj.shape[0]
    top_k = router_indices.shape[1]
    device = hidden_states.device
    token_flat = torch.arange(num_tokens, device=device).unsqueeze(-1).expand(-1, top_k).reshape(-1)
    expert_flat = router_indices.reshape(-1)
    sorted_expert, sort_idx = torch.sort(expert_flat, stable=True)
    sorted_tokens = token_flat[sort_idx]
    expert_counts = torch.bincount(sorted_expert, minlength=num_experts)
    max_count = int(expert_counts.max().item())
    num_assign = sorted_tokens.numel()
    arange_assign = torch.arange(num_assign, device=device, dtype=torch.int64)
    cum_starts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cum_starts[1:] = expert_counts.cumsum(0)
    group_id = torch.searchsorted(cum_starts[1:], arange_assign, right=True)
    pos_in_group = arange_assign - cum_starts[group_id]
    current_state = hidden_states[sorted_tokens]
    padded_state = torch.zeros(num_experts, max_count, hidden_size, device=device, dtype=hidden_states.dtype)
    padded_state[group_id, pos_in_group] = current_state
    gate_up = torch.bmm(padded_state, gate_up_proj)
    gate_up = gate_up + gate_up_proj_bias.unsqueeze(1)
    gate = gate_up[..., 0::2].clamp(min=None, max=limit)
    up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    expert_output = torch.bmm(gated_output, down_proj)
    expert_output = expert_output + down_proj_bias.unsqueeze(1)
    weighted = expert_output[group_id, pos_in_group] * routing_weights[sorted_tokens, sorted_expert, None]
    output = torch.zeros_like(hidden_states)
    output.index_add_(0, sorted_tokens, weighted)
    return output
