import torch
import torch.nn.functional as F


_DETERMINISTIC = True


# Fixed constants for gated GLU activation
ALPHA = 1.702
LIMIT = 7.0


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for sparse expert dispatch and combine."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    num_local_experts = axes_and_scalars["num_local_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device)
    router_indices = torch.randint(
        0, num_local_experts, (num_tokens, num_experts_per_tok), dtype=torch.int64, device=device
    )
    routing_logits = torch.randn(num_tokens, num_local_experts, dtype=torch.float32, device=device)
    routing_weights = F.softmax(routing_logits, dim=-1)

    std = 0.02
    gate_up_proj = torch.randn(num_local_experts, hidden_size, 2 * intermediate_size, dtype=torch.float32, device=device) * std
    gate_up_proj_bias = torch.zeros(num_local_experts, 2 * intermediate_size, dtype=torch.float32, device=device)
    down_proj = torch.randn(num_local_experts, intermediate_size, hidden_size, dtype=torch.float32, device=device) * std
    down_proj_bias = torch.zeros(num_local_experts, hidden_size, dtype=torch.float32, device=device)

    return {
        "hidden_states": hidden_states,
        "router_indices": router_indices,
        "routing_weights": routing_weights,
        "gate_up_proj": gate_up_proj,
        "gate_up_proj_bias": gate_up_proj_bias,
        "down_proj": down_proj,
        "down_proj_bias": down_proj_bias,
    }


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
    """Sparse expert dispatch via grouped (padded) GEMM over all experts."""
    _prev_det = torch.are_deterministic_algorithms_enabled()
    if _DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        return _run_impl(
            hidden_states, router_indices, routing_weights,
            gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias)
    finally:
        torch.use_deterministic_algorithms(_prev_det)


def _run_impl(
    hidden_states, router_indices, routing_weights,
    gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias,
):
    alpha = ALPHA
    limit = LIMIT

    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_up_proj.shape[0]
    top_k = router_indices.shape[1]
    device = hidden_states.device

    # Flatten (token, expert) assignments.
    flat_expert = router_indices.reshape(-1)                      # [N*K]
    flat_token = torch.arange(num_tokens, device=device).repeat_interleave(top_k)  # [N*K]

    counts = torch.bincount(flat_expert, minlength=num_experts)    # [E]
    max_count = int(counts.max().item())

    # Group expanded rows by expert (stable sort).
    perm = torch.argsort(flat_expert, stable=True)                # [N*K]
    sorted_expert = flat_expert[perm]
    sorted_token = flat_token[perm]

    offsets = torch.zeros(num_experts + 1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, 0)
    padded_token = torch.full((num_experts, max_count), -1, device=device, dtype=torch.long)
    pos = torch.arange(perm.numel(), device=device) - offsets[sorted_expert]
    padded_token[sorted_expert, pos] = sorted_token

    valid = padded_token >= 0
    safe_token = padded_token.clamp(min=0)
    grouped_states = hidden_states[safe_token]                    # [E, max_count, H]
    grouped_states = grouped_states * valid[:, :, None]

    # Grouped gate-up projection.
    gate_up = torch.bmm(grouped_states, gate_up_proj)             # [E, max_count, 2I]
    gate_up = gate_up + gate_up_proj_bias.unsqueeze(1)

    gate = gate_up[..., 0::2]
    up = gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    gated_output = gated_output * valid[:, :, None]

    # Grouped down projection.
    expert_output = torch.bmm(gated_output, down_proj)            # [E, max_count, H]
    expert_output = expert_output + down_proj_bias.unsqueeze(1)
    expert_output = expert_output * valid[:, :, None]

    # Apply routing weights and scatter-add back to tokens.
    expert_ids = torch.arange(num_experts, device=device).unsqueeze(1).expand(-1, max_count)
    rw = routing_weights[safe_token, expert_ids]                  # [E, max_count]
    weighted = expert_output * rw[:, :, None]
    weighted = weighted * valid[:, :, None]

    output = torch.zeros_like(hidden_states)
    flat_weighted = weighted.reshape(-1, hidden_size)
    flat_token_flat = padded_token.reshape(-1)
    validmask_flat = valid.reshape(-1)
    flat_weighted = flat_weighted[validmask_flat]
    flat_token_flat = flat_token_flat[validmask_flat]
    output.index_add_(0, flat_token_flat, flat_weighted)
    return output
