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


@torch.compile(dynamic=True)
def _expert_mlp(current_state, gate_w, up_w, down_w):
    gate_out = F.linear(current_state, gate_w)
    up_out = F.linear(current_state, up_w)
    gate_activated = F.silu(gate_out.to(torch.float32))
    intermediate = (gate_activated * up_out.to(torch.float32)).to(torch.bfloat16)
    return F.linear(intermediate, down_w)


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

    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device
    )

    counts_cpu = counts.cpu().tolist()
    starts_cpu = starts.cpu().tolist()

    for expert_idx in range(num_experts):
        n = counts_cpu[expert_idx]
        if n == 0:
            continue
        s = starts_cpu[expert_idx]
        top_x = sorted_token[s:s + n]
        idx = sorted_k[s:s + n]

        current_state = hidden_states_flat[top_x]

        gate_w = gate_proj_weights[expert_idx]
        up_w = up_proj_weights[expert_idx]
        down_w = down_proj_weights[expert_idx]

        expert_output = _expert_mlp(current_state, gate_w, up_w, down_w)

        token_weights = routing_weights[top_x, idx].unsqueeze(1).to(torch.bfloat16)
        weighted_output = expert_output * token_weights

        final_hidden_states.index_add_(0, top_x, weighted_output)

    final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
    return final_hidden_states
