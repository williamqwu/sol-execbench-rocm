import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
    shared_gate_proj_weight: torch.Tensor,
    shared_up_proj_weight: torch.Tensor,
    shared_down_proj_weight: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    n_routed_experts = 160
    num_experts_per_tok = 8

    H = hidden_states.shape[1]
    I = expert_gate_projs.shape[1]
    N_tok = hidden_states.shape[0]
    E = n_routed_experts

    # ---- Step 1: Routing (n_group=1, topk_group=1 => group selection is no-op) ----
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    scores = torch.sigmoid(router_logits)
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor

    # ---- Step 2: Routed expert outputs via padded bmm ----
    # Sort token-expert assignments by expert for contiguous per-expert groups.
    flat_tok = torch.arange(N_tok, device=hidden_states.device).unsqueeze(-1).expand(-1, num_experts_per_tok)
    flat_expert = topk_indices.reshape(-1)
    flat_tok = flat_tok.reshape(-1)
    flat_weight = topk_weights.reshape(-1)

    sort_order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[sort_order]
    sorted_tok = flat_tok[sort_order]
    sorted_weight = flat_weight[sort_order]

    expert_counts = torch.bincount(sorted_expert, minlength=E)
    max_count = int(expert_counts.max().item())

    # Build padded per-expert token blocks: [E, max_count, H]
    expert_input = hidden_states[sorted_tok]  # [N*8, H]
    # Create padded buffer and a scatter index.
    padded_input = expert_input.new_zeros(E * max_count, H)
    # offset for each assignment = expert_start + position_within_expert
    expert_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=hidden_states.device, dtype=expert_counts.dtype), expert_counts[:-1]]), dim=0
    )
    within = torch.arange(N_tok * num_experts_per_tok, device=hidden_states.device) - expert_starts[sorted_expert]
    pad_idx = sorted_expert * max_count + within
    padded_input[pad_idx] = expert_input
    padded_input = padded_input.view(E, max_count, H)

    pad_weights = sorted_weight.new_zeros(E * max_count)
    pad_weights[pad_idx] = sorted_weight
    pad_weights = pad_weights.view(E, max_count)
    # mask for padding
    pad_mask = pad_weights != 0  # True where real

    # bmm: [E, max_count, H] x [E, H, I] -> [E, max_count, I]
    gate_w = expert_gate_projs.transpose(1, 2)  # [E, H, I]
    up_w = expert_up_projs.transpose(1, 2)
    gate_out = torch.bmm(padded_input, gate_w)
    up_out = torch.bmm(padded_input, up_w)
    intermediate = F.silu(gate_out) * up_out  # [E, max_count, I]

    down_w = expert_down_projs.transpose(1, 2)  # [E, I, H]
    expert_output = torch.bmm(intermediate, down_w)  # [E, max_count, H]

    # apply weights, zero out padding
    weighted = expert_output * pad_weights.unsqueeze(-1)
    weighted = weighted * pad_mask.unsqueeze(-1)
    weighted = weighted.view(E * max_count, H)

    # gather back: only real entries
    out_flat = weighted[pad_idx]  # [N*8, H]
    final_hidden = torch.zeros(N_tok, H, device=hidden_states.device, dtype=out_flat.dtype)
    final_hidden.index_add_(0, sorted_tok, out_flat)
    routed_output = final_hidden.to(hidden_states.dtype)

    # ---- Step 3: Shared expert ----
    gate_output = F.linear(hidden_states, shared_gate_proj_weight)
    up_output = F.linear(hidden_states, shared_up_proj_weight)
    intermediate = F.silu(gate_output) * up_output
    shared_output = F.linear(intermediate, shared_down_proj_weight)

    # ---- Step 4: Combine ----
    output = routed_output + shared_output
    return output
