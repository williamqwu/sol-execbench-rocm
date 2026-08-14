import torch


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """Exact reference arithmetic with lazy per-routed-expert dequantization."""
    T = routing_logits.shape[0]
    H = 7168
    I = 2048
    BLOCK = 128

    # Input dequantization, including the reference's transposed scale layout.
    a_scale = hidden_states_scale.float().permute(1, 0).contiguous()
    a_scale = (
        a_scale.unsqueeze(-1)
        .repeat(1, 1, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    activations = hidden_states.float() * a_scale

    # DeepSeek no-aux routing. Keep the explicit sigmoid expression and each
    # topk/scatter operation identical to the specification to preserve ties.
    scores_unbiased = 1.0 / (1.0 + torch.exp(-routing_logits.float()))
    scores_biased = scores_unbiased + routing_bias.float().reshape(-1)
    grouped = scores_biased.view(T, 8, 32)
    top2 = torch.topk(
        grouped, k=2, dim=2, largest=True, sorted=False
    ).values
    group_scores = top2.sum(dim=2)
    group_idx = torch.topk(
        group_scores, k=4, dim=1, largest=True, sorted=False
    ).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, 8, 32).reshape(T, 256)
    pruned = scores_biased.masked_fill(
        score_mask == 0, torch.finfo(torch.float32).min
    )
    topk_idx = torch.topk(
        pruned, k=8, dim=1, largest=True, sorted=False
    ).indices

    selected_experts = torch.zeros_like(scores_unbiased)
    selected_experts.scatter_(1, topk_idx, 1.0)
    routing_weights = scores_unbiased * selected_experts
    routing_weights = routing_weights / (
        routing_weights.sum(dim=1, keepdim=True) + 1e-20
    )
    routing_weights = routing_weights * routed_scaling_factor

    output = torch.zeros(
        (T, H), dtype=torch.float32, device=hidden_states.device
    )
    local_start = int(local_expert_offset)

    for local_expert in range(32):
        global_expert = local_start + local_expert
        if global_expert < 0 or global_expert >= 256:
            continue

        token_mask = (topk_idx == global_expert).any(dim=1)
        if not token_mask.any():
            continue
        token_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(1)

        # These are exactly the corresponding slices of the reference's full
        # repeat_interleave expansions, materialized only for routed experts.
        scale13 = torch.repeat_interleave(
            gemm1_weights_scale[local_expert].float(), BLOCK, dim=0
        )
        scale13 = torch.repeat_interleave(scale13, BLOCK, dim=1)
        weight13 = gemm1_weights[local_expert].float() * scale13

        gemm1 = activations.index_select(0, token_idx).matmul(weight13.t())
        x1 = gemm1[:, :I]
        x2 = gemm1[:, I:]
        intermediate = (x2 / (1.0 + torch.exp(-x2))) * x1

        scale2 = torch.repeat_interleave(
            gemm2_weights_scale[local_expert].float(), BLOCK, dim=0
        )
        scale2 = torch.repeat_interleave(scale2, BLOCK, dim=1)
        weight2 = gemm2_weights[local_expert].float() * scale2
        expert_output = intermediate.matmul(weight2.t())

        token_weights = routing_weights.index_select(0, token_idx)[
            :, global_expert
        ]
        output.index_add_(
            0, token_idx, expert_output * token_weights.unsqueeze(1)
        )

    return output.to(torch.bfloat16)
