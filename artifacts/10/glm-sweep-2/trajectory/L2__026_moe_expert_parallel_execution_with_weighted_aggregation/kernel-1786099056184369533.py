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


@torch.compile(dynamic=True, mode="reduce-overhead")
def _batched_mlp(padded_state, gate_w_t, up_w_t, down_w_t):
    gate_out = torch.bmm(padded_state, gate_w_t)
    up_out = torch.bmm(padded_state, up_w_t)
    intermediate = F.silu(gate_out) * up_out
    return torch.bmm(intermediate, down_w_t)


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

    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    num_tokens = batch_size * seq_len

    flat_expert = selected_experts.reshape(-1)
    flat_token = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(1)
        .expand(-1, num_experts_per_tok)
        .reshape(-1)
    )
    flat_k = (
        torch.arange(num_experts_per_tok, device=hidden_states.device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
        .reshape(-1)
    )

    sorted_order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[sorted_order]
    sorted_token = flat_token[sorted_order]
    sorted_k = flat_k[sorted_order]

    counts = torch.bincount(sorted_expert, minlength=num_experts)
    offsets = torch.cumsum(counts, dim=0)
    starts = torch.zeros_like(offsets)
    starts[1:] = offsets[:-1]

    max_tokens = int(counts.max().item())

    expert_starts_per_assign = starts[sorted_expert]
    pos_in_expert_sorted = torch.arange(
        num_tokens * num_experts_per_tok, device=hidden_states.device
    ) - expert_starts_per_assign

    E = num_experts
    padded_state = torch.zeros(
        (E, max_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device
    )
    padded_state[sorted_expert, pos_in_expert_sorted] = hidden_states_flat[sorted_token]

    gate_w_t = gate_proj_weights.transpose(1, 2)
    up_w_t = up_proj_weights.transpose(1, 2)
    down_w_t = down_proj_weights.transpose(1, 2)

    padded_out = _batched_mlp(padded_state, gate_w_t, up_w_t, down_w_t)

    expert_output_per = padded_out[sorted_expert, pos_in_expert_sorted]
    token_weights = routing_weights[sorted_token, sorted_k].unsqueeze(1).to(torch.bfloat16)
    weighted_output = expert_output_per * token_weights

    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device
    )
    final_hidden_states.index_add_(0, sorted_token, weighted_output)

    final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
    return final_hidden_states
