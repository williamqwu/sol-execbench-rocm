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
    """
    Sparse expert dispatch and weighted combination for MoE training.

    Uses sort-by-expert + grouped GEMM to process all experts in two batched
    matrix multiplies instead of a Python loop with per-expert launches.
    """
    alpha = ALPHA
    limit = LIMIT

    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_up_proj.shape[0]
    top_k = router_indices.shape[1]
    device = hidden_states.device

    # --- Dispatch: flatten all (token, slot) assignments and sort by expert ---
    # token_flat: [num_tokens*top_k] token index, expert_flat: [num_tokens*top_k] expert index
    token_flat = torch.arange(num_tokens, device=device).unsqueeze(-1).expand(-1, top_k).reshape(-1)
    expert_flat = router_indices.reshape(-1)  # [num_tokens*top_k]

    # Sort assignments by expert id so all tokens for a given expert are contiguous.
    sorted_expert, sort_idx = torch.sort(expert_flat, stable=True)
    sorted_tokens = token_flat[sort_idx]

    # Build cumulative end-offsets per expert for grouped_mm: offs[i] = number of
    # assignments with expert <= i.  grouped_mm uses [offs[i-1], offs[i]).
    # Count assignments per expert.
    expert_counts = torch.bincount(sorted_expert, minlength=num_experts)
    # cumulative end offsets, length == num_experts
    offs = torch.cumsum(expert_counts, dim=0).to(torch.int32)

    # Only experts that actually received tokens participate; but grouped_mm
    # requires the weight tensor batch dim to equal offs.size(0). We keep all
    # experts and rely on zero-size groups being skipped. Handle experts with
    # zero count: grouped_mm needs strictly increasing offsets. Remove empties.
    nonzero = expert_counts > 0
    if not nonzero.all():
        keep = nonzero.nonzero(as_tuple=False).squeeze(-1)
        offs = offs[keep]
        # slice weight tensors to active experts; offsets remain cumulative across
        # the full sorted stream, which is what grouped_mm expects.
        gate_up_w = gate_up_proj[keep]
        gate_up_b = gate_up_proj_bias[keep]
        down_w = down_proj[keep]
        down_b = down_proj_bias[keep]
        # expert id per kept assignment for weight lookup
        kept_expert = sorted_expert
    else:
        gate_up_w = gate_up_proj
        gate_up_b = gate_up_proj_bias
        down_w = down_proj
        down_b = down_proj_bias
        kept_expert = sorted_expert

    num_assign = sorted_tokens.numel()

    # Gather hidden states for all assignments: [num_assign, hidden_size]
    current_state = hidden_states[sorted_tokens]

    # --- Gate-up projection via grouped GEMM ---
    # weight layout [E, K, N] with K=hidden_size, N=2*intermediate_size
    gate_up = torch._grouped_mm(current_state, gate_up_w, offs=offs)
    # add per-expert bias
    # bias is [active_E, 2*intermediate_size]; expand per assignment
    gate_up = gate_up + gate_up_b[kept_expert]

    # Split gate/up: original interleaves [g0,u0,g1,u1,...]; slicing ::2 and 1::2
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]

    # Clamp
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)

    # Gated GLU
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu

    # --- Down projection via grouped GEMM ---
    # down_w [E, intermediate_size, hidden_size] -> [E, K=inter, N=hidden]
    expert_output = torch._grouped_mm(gated_output, down_w, offs=offs)
    expert_output = expert_output + down_b[kept_expert]

    # Apply routing weights
    weighted_output = expert_output * routing_weights[sorted_tokens, sorted_expert, None]

    # Scatter-add accumulate back to output
    output = torch.zeros_like(hidden_states)
    output.index_add_(0, sorted_tokens, weighted_output)

    return output
