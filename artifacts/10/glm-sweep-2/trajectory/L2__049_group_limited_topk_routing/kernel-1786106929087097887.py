import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    experts_per_group = num_experts // n_group  # 32

    num_tokens = hidden_states.shape[0]

    # bf16 GEMM with fp32 accumulation (no upcast)
    logits = F.linear(hidden_states, weight)

    # sigmoid in fp32 for numerical stability of the routing
    scores = torch.sigmoid(logits.to(torch.float32))  # [num_tokens, 256]

    scores_for_routing = scores + expert_bias.to(torch.float32)

    # group scores: top-2 within each group of 32, summed -> [num_tokens, 8]
    group_scores_reshaped = scores_for_routing.view(num_tokens, n_group, experts_per_group)
    top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=-1)

    _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)

    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, experts_per_group)
        .reshape(num_tokens, num_experts)
    )

    neg_inf = torch.finfo(torch.float32).min
    masked_scores = scores_for_routing.masked_fill(score_mask == 0, neg_inf)

    _, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)

    selected_scores = torch.gather(scores, dim=1, index=topk_idx)

    topk_weight = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor

    return topk_idx, topk_weight
