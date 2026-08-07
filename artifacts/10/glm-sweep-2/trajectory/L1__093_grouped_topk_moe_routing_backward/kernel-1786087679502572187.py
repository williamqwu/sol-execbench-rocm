import torch
import triton
import triton.language as tl


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


@triton.jit
def _fused_pre_matmul_kernel(
    grad_topk_weights_ptr,
    topk_weights_normalized_ptr,
    denominator_ptr,
    topk_indices_ptr,
    scores_ptr,
    grad_router_logits_ptr,
    n_routed_experts: tl.constexpr,
    top_k: tl.constexpr,
    scaling: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, top_k)
    offs = pid * top_k + cols

    grad_topk_weights = tl.load(grad_topk_weights_ptr + offs)
    topk_weights_norm = tl.load(topk_weights_normalized_ptr + offs)
    denominator = tl.load(denominator_ptr + pid)

    grad_normalized = (grad_topk_weights * scaling).to(tl.bfloat16)
    prod = (grad_normalized * topk_weights_norm).to(tl.bfloat16)
    grad_sum = tl.sum(prod.to(tl.float32), axis=0).to(tl.bfloat16)
    diff = (grad_normalized - grad_sum).to(tl.bfloat16)
    grad_topk_weights_unnorm = (diff / denominator).to(tl.bfloat16)

    indices = tl.load(topk_indices_ptr + offs)
    score_offs = pid * n_routed_experts + indices
    scores_at = tl.load(scores_ptr + score_offs)

    gs_times_s = (grad_topk_weights_unnorm * scores_at).to(tl.bfloat16)
    one_minus = (1.0 - scores_at).to(tl.bfloat16)
    grad_router = (gs_times_s * one_minus).to(tl.bfloat16)

    store_offs = pid * n_routed_experts + indices
    tl.store(grad_router_logits_ptr + store_offs, grad_router)


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
    N = scores.shape[0]
    n_routed_experts = scores.shape[1]
    top_k = topk_indices.shape[1]

    grad_router_logits = torch.zeros(
        (N, n_routed_experts), dtype=scores.dtype, device=scores.device
    )

    _fused_pre_matmul_kernel[(N,)](
        grad_topk_weights,
        topk_weights_normalized,
        denominator,
        topk_indices,
        scores,
        grad_router_logits,
        n_routed_experts=n_routed_experts,
        top_k=top_k,
        scaling=float(routed_scaling_factor),
        num_warps=1,
    )

    grad_hidden_states = torch.matmul(grad_router_logits, weight)
    grad_weight = torch.matmul(grad_router_logits.t(), hidden_states)

    return grad_hidden_states, grad_weight
