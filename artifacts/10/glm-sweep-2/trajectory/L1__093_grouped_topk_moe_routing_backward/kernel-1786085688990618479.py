import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for backward pass testing."""
    N = axes_and_scalars["N"]
    hidden_size = 5120
    n_routed_experts = 160
    top_k = 8
    routed_scaling_factor = 2.5

    grad_topk_weights = torch.randn(N, top_k, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(N, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n_routed_experts, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    router_logits = torch.randn(N, n_routed_experts, dtype=torch.bfloat16, device=device)
    scores = torch.sigmoid(router_logits)
    topk_indices = torch.stack([
        torch.randperm(n_routed_experts, device=device)[:top_k]
        for _ in range(N)
    ]).to(torch.int64)
    topk_weights = scores.gather(1, topk_indices)
    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights_normalized = topk_weights / denominator

    return {
        "grad_topk_weights": grad_topk_weights,
        "hidden_states": hidden_states,
        "weight": weight,
        "scores": scores,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "topk_weights_normalized": topk_weights_normalized,
        "denominator": denominator,
        "routed_scaling_factor": routed_scaling_factor,
    }


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
    grad_topk_weights_normalized = grad_topk_weights * routed_scaling_factor
    grad_sum = (grad_topk_weights_normalized * topk_weights_normalized).sum(dim=-1, keepdim=True)
    grad_topk_weights_unnorm = (grad_topk_weights_normalized - grad_sum) / denominator

    grad_scores = torch.zeros(scores.shape[0], scores.shape[1], dtype=grad_topk_weights_unnorm.dtype,
                              device=grad_topk_weights_unnorm.device)
    grad_scores.scatter_add_(1, topk_indices, grad_topk_weights_unnorm)

    grad_router_logits = grad_scores * scores * (1.0 - scores)

    grad_hidden_states = torch.matmul(grad_router_logits, weight)
    grad_weight = torch.matmul(grad_router_logits.t(), hidden_states)

    return grad_hidden_states, grad_weight
