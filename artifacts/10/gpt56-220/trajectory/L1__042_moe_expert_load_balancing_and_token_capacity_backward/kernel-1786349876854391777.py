import torch


@torch.no_grad()
def run(
    grad_tokens_per_expert: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_load_balance_loss: torch.Tensor,
    topk_idx: torch.Tensor,
    expert_mask: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    training: torch.Tensor,
):
    n_tokens = topk_idx.shape[0]
    loss_grad = grad_load_balance_loss * training / (n_tokens * 8)
    return grad_expert_mask + grad_tokens_per_expert.unsqueeze(0) + loss_grad
