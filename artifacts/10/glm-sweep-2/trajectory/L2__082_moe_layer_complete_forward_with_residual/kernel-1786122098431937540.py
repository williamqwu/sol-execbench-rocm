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
    # ---- Step 1: Routing ----
    # n_group=1, topk_group=1 => hierarchical group selection is a no-op (one
    # group containing all experts is always selected). Reduce to: sigmoid +
    # bias, then topk over all experts.
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

    # ---- Step 2: Routed expert outputs via grouped GEMM ----
    # Flatten token/expert assignments and sort by expert id so tokens for the
    # same expert are contiguous (required by _grouped_mm offsets).
    N_tok = hidden_states.shape[0]
    flat_tok = torch.arange(N_tok, device=hidden_states.device).unsqueeze(-1).expand(-1, num_experts_per_tok)
    flat_expert = topk_indices.reshape(-1)
    flat_tok = flat_tok.reshape(-1)
    flat_weight = topk_weights.reshape(-1)

    sort_order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[sort_order]
    sorted_tok = flat_tok[sort_order]
    sorted_weight = flat_weight[sort_order]

    # Gather input tokens in sorted order
    expert_input = hidden_states[sorted_tok]  # [N_tok*8, H]

    # Per-expert end offsets (cumulative), length n_routed_experts.
    # For each expert, count how many tokens assigned.
    expert_counts = torch.bincount(sorted_expert, minlength=n_routed_experts)
    # cumulative end offsets
    expert_offsets = torch.cumsum(expert_counts, dim=0).to(torch.int32)

    # Transpose weights to [E, K, N] layout for _grouped_mm.
    # gate: weight [E, 1536, H] -> [E, H, 1536] (a @ b = [toks,H]@[H,1536])
    gate_w = expert_gate_projs.transpose(1, 2)
    up_w = expert_up_projs.transpose(1, 2)
    # down: weight [E, H, 1536] -> [E, 1536, H] (a @ b = [toks,1536]@[1536,H])
    down_w = expert_down_projs.transpose(1, 2)

    gate_out = torch._grouped_mm(expert_input, gate_w, expert_offsets, None, None)
    up_out = torch._grouped_mm(expert_input, up_w, expert_offsets, None, None)
    intermediate = F.silu(gate_out) * up_out
    expert_output = torch._grouped_mm(intermediate, down_w, expert_offsets, None, None)

    # Apply routing weights
    weighted_output = expert_output * sorted_weight.unsqueeze(-1)

    # Scatter-add back to final output (tokens may be assigned to multiple experts)
    final_hidden = torch.zeros(N_tok, H, device=hidden_states.device, dtype=weighted_output.dtype)
    final_hidden.index_add_(0, sorted_tok, weighted_output)
    routed_output = final_hidden.to(hidden_states.dtype)

    # ---- Step 3: Shared expert ----
    gate_output = F.linear(hidden_states, shared_gate_proj_weight)
    up_output = F.linear(hidden_states, shared_up_proj_weight)
    intermediate = F.silu(gate_output) * up_output
    shared_output = F.linear(intermediate, shared_down_proj_weight)

    # ---- Step 4: Combine ----
    output = routed_output + shared_output
    return output
