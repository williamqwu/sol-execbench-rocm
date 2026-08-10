import torch


@torch.no_grad()
def run(
    grad_topk_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_weights_normalized: torch.Tensor,
    denominator: torch.Tensor,
    routed_scaling_factor: float,
):
    grad_normalized = grad_topk_weights * routed_scaling_factor
    grad_sum = (grad_normalized * topk_weights_normalized).sum(dim=-1, keepdim=True)
    grad_unnorm = (grad_normalized - grad_sum) / denominator
    grad_scores = torch.zeros_like(scores)
    grad_scores.scatter_add_(1, topk_indices, grad_unnorm)
    grad_logits = grad_scores * scores * (1.0 - scores)
    return grad_logits @ weight, grad_logits.t() @ hidden_states
