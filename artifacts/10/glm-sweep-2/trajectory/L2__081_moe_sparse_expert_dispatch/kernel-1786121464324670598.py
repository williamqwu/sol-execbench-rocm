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
    """
    Sparse MoE expert computation with top-k routing.
    n_group=1, topk_group=1 => group selection is a no-op; topk is global.
    Only iterate over experts that actually receive tokens.
    """
    n_routed_experts = 128
    num_experts_per_tok = 8

    num_tokens = hidden_states.shape[0]

    # Router logits in bf16 (fp32 not needed for routing decision).
    router_logits = F.linear(hidden_states, router_weight)
    scores = torch.sigmoid(router_logits.float())
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)

    # n_group=1, topk_group=1 => score_mask is all ones; skip masking.
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False
    )

    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Determine which experts actually receive tokens.
    expert_counts = torch.bincount(topk_indices.flatten(), minlength=n_routed_experts)
    active_experts = torch.nonzero(expert_counts, as_tuple=False).flatten()

    final_hidden_states = torch.zeros_like(hidden_states)

    # expert_mask: [n_routed_experts, num_tokens, num_experts_per_tok]
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 0, 1)

    for expert_idx in active_experts.tolist():
        mask = expert_mask[expert_idx]
        token_indices, weight_indices = torch.where(mask)

        expert_weights = topk_weights[token_indices, weight_indices]
        expert_input = hidden_states[token_indices]

        gate_output = F.silu(F.linear(expert_input, expert_gate_weights[expert_idx]))
        up_output = F.linear(expert_input, expert_up_weights[expert_idx])
        intermediate = gate_output * up_output
        expert_output = F.linear(intermediate, expert_down_weights[expert_idx])

        weighted_output = expert_output * expert_weights.unsqueeze(-1)
        final_hidden_states.index_add_(0, token_indices, weighted_output)

    # Shared expert
    shared_gate_output = F.silu(F.linear(hidden_states, shared_gate_weight))
    shared_up_output = F.linear(hidden_states, shared_up_weight)
    shared_intermediate = shared_gate_output * shared_up_output
    shared_output = F.linear(shared_intermediate, shared_down_weight)

    output = final_hidden_states + shared_output
    return output
