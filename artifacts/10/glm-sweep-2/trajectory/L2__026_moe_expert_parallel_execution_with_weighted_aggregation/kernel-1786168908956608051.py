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

    num_tokens = batch_size * seq_len
    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    # Flatten selected_experts and routing_weights so each (token, top_k) pair
    # is a row. Then sort by expert id to bin tokens per expert in one pass.
    # flat_expert: [num_tokens * top_k]
    flat_expert = selected_experts.reshape(-1)  # [num_tokens * top_k]
    # token index for each row: arange(num_tokens) repeated per top_k slot
    # row r -> token = r // top_k, slot = r % top_k
    flat_token = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(num_experts_per_tok)

    # Sort by expert id
    sorted_expert, sort_order = torch.sort(flat_expert, stable=True)
    sorted_token = flat_token[sort_order]  # token index per sorted row

    # slot index for each row: r % top_k
    row_ids = torch.arange(num_tokens * num_experts_per_tok, device=hidden_states.device)
    flat_slot = row_ids % num_experts_per_tok
    sorted_slot = flat_slot[sort_order]

    # Boundaries: count how many rows per expert
    expert_counts = torch.bincount(flat_expert, minlength=num_experts)
    # offsets via cumsum
    expert_offsets = torch.zeros(num_experts + 1, device=hidden_states.device, dtype=torch.int64)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device
    )

    for expert_idx in range(num_experts):
        start = expert_offsets[expert_idx].item()
        end = expert_offsets[expert_idx + 1].item()
        n = end - start
        if n == 0:
            continue

        top_x = sorted_token[start:end]      # [n] token indices
        idx = sorted_slot[start:end]          # [n] slot (top-k position)

        current_state = hidden_states_flat[top_x]  # [n, hidden] bf16

        gate_w = gate_proj_weights[expert_idx]  # [intermediate, hidden] bf16
        up_w = up_proj_weights[expert_idx]
        down_w = down_proj_weights[expert_idx]

        gate_out = (current_state @ gate_w.t()).to(torch.float32)
        up_out = (current_state @ up_w.t()).to(torch.float32)

        gate_activated = F.silu(gate_out)
        intermediate = gate_activated * up_out

        expert_output = intermediate.to(torch.bfloat16) @ down_w.t()

        token_weights = routing_weights[top_x, idx].unsqueeze(1)  # [n, 1] fp32
        weighted_output = expert_output.to(torch.float32) * token_weights

        final_hidden_states.index_add_(
            0,
            top_x,
            weighted_output.to(torch.bfloat16)
        )

    final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
    return final_hidden_states
