import torch
import triton
import triton.language as tl


@triton.jit
def _routing_grad_kernel(gu_ptr, scores_ptr, indices_ptr, out_ptr, E: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, BLOCK)
    mask = e < E
    s = tl.load(scores_ptr + row * E + e, mask=mask, other=0.0)
    gs = tl.zeros((BLOCK,), tl.bfloat16)
    for i in tl.static_range(8):
        idx_i = tl.load(indices_ptr + row * 8 + i)
        gu_i = tl.load(gu_ptr + row * 8 + i)
        hit = e == idx_i
        gs = tl.where(hit, gu_i, gs)
    one_minus = (1.0 - s.to(tl.float32)).to(tl.bfloat16)
    first = (gs.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    out = (first.to(tl.float32) * one_minus.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row * E + e, out, mask=mask)


def _routing_grad(gu, scores, indices):
    out = torch.empty_like(scores)
    _routing_grad_kernel[(scores.shape[0],)](
        gu, scores, indices, out,
        E=scores.shape[1], BLOCK=256, num_warps=4, waves_per_eu=16)
    return out


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
    grad_logits = _routing_grad(grad_unnorm, scores, topk_indices)
    return grad_logits @ weight, grad_logits.t() @ hidden_states
