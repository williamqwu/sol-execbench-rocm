import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len

    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.bfloat16, device=device
    )

    routing_weights = torch.rand(
        batch_seq_len, num_experts_per_tok,
        dtype=torch.float32, device=device
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    selected_experts = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(batch_seq_len)
    ]).to(torch.int64)

    gate_proj_weights = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02

    up_proj_weights = torch.randn(
        num_experts, intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02

    down_proj_weights = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device
    ) * 0.02

    return {
        "hidden_states": hidden_states,
        "routing_weights": routing_weights,
        "selected_experts": selected_experts,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
    }


@torch.no_grad()
def _compute(
    x: torch.Tensor,
    gate_up_w_t: torch.Tensor,
    down_w_t: torch.Tensor,
    token_weights: torch.Tensor,
    intermediate_size: int,
):
    # Fused gate+up: [E, M, 2*intermediate]
    gate_up_out = torch.bmm(x, gate_up_w_t).to(torch.float32)
    gate_out = gate_up_out[:, :, :intermediate_size]
    up_out = gate_up_out[:, :, intermediate_size:]

    gate_activated = F.silu(gate_out)
    intermediate = gate_activated * up_out

    expert_output = torch.bmm(intermediate.to(torch.bfloat16), down_w_t)

    weighted_output = expert_output.to(torch.float32) * token_weights.unsqueeze(-1)
    return weighted_output


_compute_compiled = torch.compile(_compute, dynamic=True)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]
    num_experts_per_tok = selected_experts.shape[1]
    intermediate_size = gate_proj_weights.shape[1]

    num_tokens = batch_size * seq_len
    device = hidden_states.device
    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    total_assignments = num_tokens * num_experts_per_tok
    flat_expert = selected_experts.reshape(-1)

    sorted_expert, sort_order = torch.sort(flat_expert, stable=True)

    row_ids = torch.arange(total_assignments, device=device)
    flat_token = row_ids // num_experts_per_tok
    flat_slot = row_ids % num_experts_per_tok
    sorted_token = flat_token[sort_order]
    sorted_slot = flat_slot[sort_order]

    expert_counts = torch.bincount(flat_expert, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, device=device, dtype=torch.int64)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    max_count = int(expert_counts.max().item())

    pos_within = torch.arange(total_assignments, device=device) - expert_offsets[sorted_expert]
    flat_idx = sorted_expert * max_count + pos_within

    padded_tokens = torch.zeros(num_experts, max_count, dtype=torch.int64, device=device)
    padded_slots = torch.zeros(num_experts, max_count, dtype=torch.int64, device=device)
    valid_mask = torch.zeros(num_experts, max_count, dtype=torch.bool, device=device)

    padded_tokens.view(-1)[flat_idx] = sorted_token
    padded_slots.view(-1)[flat_idx] = sorted_slot
    valid_mask.view(-1)[flat_idx] = True

    x = hidden_states_flat[padded_tokens]

    # Fuse gate and up weight transposes: [E, hidden, 2*intermediate]
    gate_up_w_t = torch.cat(
        [gate_proj_weights.transpose(-1, -2), up_proj_weights.transpose(-1, -2)],
        dim=-1
    )
    down_w_t = down_proj_weights.transpose(-1, -2)

    token_weights = routing_weights[padded_tokens, padded_slots] * valid_mask

    weighted_output = _compute_compiled(
        x, gate_up_w_t, down_w_t, token_weights, intermediate_size
    )

    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=torch.float32,
        device=device
    )
    final_hidden_states.index_add_(
        0,
        padded_tokens.reshape(-1),
        weighted_output.reshape(-1, hidden_dim)
    )

    final_hidden_states = final_hidden_states.to(torch.bfloat16).reshape(batch_size, seq_len, hidden_dim)
    return final_hidden_states
