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

    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    expert_mask = F.one_hot(
        selected_experts,
        num_classes=num_experts
    ).permute(2, 1, 0)  # [num_experts, top_k, batch*seq]

    final_hidden_states = torch.zeros(
        (batch_size * seq_len, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device
    )

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])

        if top_x.shape[0] == 0:
            continue

        current_state = hidden_states_flat[top_x]  # [num_tokens, hidden] bf16

        gate_w = gate_proj_weights[expert_idx]  # [intermediate, hidden] bf16
        up_w = up_proj_weights[expert_idx]
        down_w = down_proj_weights[expert_idx]

        # bf16 matmuls (fp32 accumulate internally), upcast results for activation
        gate_out = (current_state @ gate_w.t()).to(torch.float32)
        up_out = (current_state @ up_w.t()).to(torch.float32)

        gate_activated = F.silu(gate_out)
        intermediate = gate_activated * up_out

        # down proj in bf16
        expert_output = intermediate.to(torch.bfloat16) @ down_w.t()

        token_weights = routing_weights[top_x, idx].unsqueeze(1)  # [num_tokens, 1] fp32
        weighted_output = expert_output.to(torch.float32) * token_weights

        final_hidden_states.index_add_(
            0,
            top_x,
            weighted_output.to(torch.bfloat16)
        )

    final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
    return final_hidden_states
