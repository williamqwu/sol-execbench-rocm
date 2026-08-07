import torch

@torch.no_grad()
def _run_impl(scores: torch.Tensor):
    num_experts = 256
    n_group = 8
    topk_group = 4
    experts_per_group = num_experts // n_group  # 32

    num_tokens = scores.size(0)

    # Reshape scores into groups: (num_tokens, 8, 32)
    group_scores_reshaped = scores.view(num_tokens, n_group, experts_per_group)

    # Compute top-2 scores per group and sum them: (num_tokens, 8, 2) -> (num_tokens, 8)
    top2_per_group = torch.topk(group_scores_reshaped, k=2, dim=-1)[0]  # (num_tokens, 8, 2)
    group_scores = top2_per_group.sum(dim=-1)  # (num_tokens, 8)

    # Select top 4 groups based on aggregated scores
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]

    # Create binary mask for selected groups: (num_tokens, 8)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Expand group mask to expert-level mask
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, experts_per_group)
        .reshape(num_tokens, num_experts)
    )

    # Apply mask
    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))

    return masked_scores, group_mask

_run_compiled = torch.compile(_run_impl, dynamic=True)


@torch.no_grad()
def run(scores: torch.Tensor):
    return _run_compiled(scores)
