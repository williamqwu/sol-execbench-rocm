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
    selected_scores = scores.gather(1, topk_indices)
    selected_logits_grad = grad_unnorm * selected_scores * (1.0 - selected_scores)
    grad_logits = torch.zeros_like(scores)
    grad_logits.scatter_add_(1, topk_indices, selected_logits_grad)
    return grad_logits @ weight, grad_logits.t() @ hidden_states
