import torch


@torch.no_grad()
def run(scores: torch.Tensor):
    rows = scores.shape[0]
    grouped = scores.view(rows, 8, 32)
    first, first_idx = grouped.max(dim=-1)
    without_first = grouped.scatter(2, first_idx.unsqueeze(-1), float('-inf'))
    second = without_first.max(dim=-1).values
    group_scores = first + second
    group_idx = torch.topk(group_scores, 4, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    expert_mask = group_mask[:, :, None].expand(rows, 8, 32).reshape(rows, 256)
    return scores.masked_fill(~expert_mask.bool(), float('-inf')), group_mask
