import torch


@torch.compile(fullgraph=True, dynamic=True)
def _fused(grad_tpe, grad_mask, grad_loss, training):
    loss_grad = grad_loss * training / (grad_mask.shape[0] * 8)
    return grad_mask + grad_tpe.unsqueeze(0) + loss_grad


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
    return _fused(
        grad_tokens_per_expert,
        grad_expert_mask,
        grad_load_balance_loss,
        training,
    )
