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
    N = hidden_states.shape[0]
    n_routed_experts = scores.shape[1]

    grad_topk_weights_normalized = grad_topk_weights * routed_scaling_factor
    grad_sum = (grad_topk_weights_normalized * topk_weights_normalized).sum(dim=-1, keepdim=True)
    grad_topk_weights_unnorm = (grad_topk_weights_normalized - grad_sum) / denominator

    grad_scores = torch.zeros(
        N, n_routed_experts, dtype=grad_topk_weights_unnorm.dtype, device=grad_topk_weights_unnorm.device
    )
    grad_scores.scatter_add_(1, topk_indices, grad_topk_weights_unnorm)
    grad_router_logits = grad_scores * scores * (1.0 - scores)

    grad_hidden_states = torch.matmul(grad_router_logits, weight)
    grad_weight = torch.matmul(grad_router_logits.t(), hidden_states)
    return grad_hidden_states, grad_weight
