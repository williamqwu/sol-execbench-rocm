import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    n_routed_experts = 128
    num_experts_per_tok = 8
    num_tokens = hidden_states.shape[0]

    router_logits = F.linear(hidden_states, router_weight)
    scores = torch.sigmoid(router_logits.float())
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)

    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )
    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor
    topk_weights = topk_weights.to(hidden_states.dtype)

    flat_indices = topk_indices.flatten()
    flat_weights = topk_weights.flatten()
    flat_tokens = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(-1)
        .expand(-1, num_experts_per_tok)
        .flatten()
    )

    order = torch.argsort(flat_indices, stable=True)
    sorted_experts = flat_indices[order]
    sorted_tokens = flat_tokens[order]
    sorted_weights = flat_weights[order]

    expert_counts = torch.bincount(flat_indices, minlength=n_routed_experts)
    cumsum = torch.cumsum(expert_counts, dim=0)
    starts = torch.cat([torch.zeros(1, device=expert_counts.device, dtype=cumsum.dtype), cumsum[:-1]])

    sorted_input = hidden_states[sorted_tokens]

    final_hidden_states = torch.zeros_like(hidden_states)

    active_experts = torch.nonzero(expert_counts, as_tuple=False).flatten()
    for expert_idx in active_experts.tolist():
        s = starts[expert_idx].item()
        e = cumsum[expert_idx].item()
        if e <= s:
            continue
        expert_input = sorted_input[s:e]
        ew = sorted_weights[s:e].unsqueeze(-1)

        # Fuse gate + up: concat weights -> single matmul
        gate_up_w = torch.cat([expert_gate_weights[expert_idx], expert_up_weights[expert_idx]], dim=0)
        gu = F.linear(expert_input, gate_up_w)
        moi = expert_gate_weights.shape[1]
        gate_output = F.silu(gu[:, :moi])
        up_output = gu[:, moi:]
        intermediate = gate_output * up_output
        expert_output = F.linear(intermediate, expert_down_weights[expert_idx])

        weighted_output = expert_output * ew
        toks = sorted_tokens[s:e]
        final_hidden_states.index_add_(0, toks, weighted_output)

    shared_gate_output = F.silu(F.linear(hidden_states, shared_gate_weight))
    shared_up_output = F.linear(hidden_states, shared_up_weight)
    shared_intermediate = shared_gate_output * shared_up_output
    shared_output = F.linear(shared_intermediate, shared_down_weight)

    output = final_hidden_states + shared_output
    return output
