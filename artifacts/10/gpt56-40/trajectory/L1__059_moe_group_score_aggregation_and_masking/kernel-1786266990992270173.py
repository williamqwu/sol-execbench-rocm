import torch


def _impl(scores):
    rows = scores.shape[0]
    grouped = scores.view(rows, 8, 32)
    group_scores = torch.topk(grouped, 2, dim=-1).values.sum(-1)
    indices = torch.topk(group_scores, 4, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores).scatter(1, indices, 1.0)
    masked = torch.where(
        group_mask[:, :, None].expand(rows, 8, 32).reshape(rows, 256).bool(),
        scores,
        float('-inf'),
    )
    return masked, group_mask


_compiled = torch.compile(_impl, mode="max-autotune-no-cudagraphs", dynamic=True)


@torch.no_grad()
def run(scores: torch.Tensor):
    return _compiled(scores)
