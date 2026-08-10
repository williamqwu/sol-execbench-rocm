import torch
import triton
import triton.language as tl


@triton.jit
def _routing_grad_kernel(grad_ptr, scores_ptr, indices_ptr, norm_ptr, denom_ptr,
                         out_ptr, scale: tl.constexpr, E: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, 8)
    go = tl.load(grad_ptr + row * 8 + k).to(tl.float32)
    nw = tl.load(norm_ptr + row * 8 + k).to(tl.float32)
    # Every named tensor in the eager reference is materialized as BF16.
    gn = (go * scale).to(tl.bfloat16)
    prod = (gn * nw).to(tl.bfloat16)
    gsum = tl.sum(prod.to(tl.float32), axis=0).to(tl.bfloat16)
    den = tl.load(denom_ptr + row).to(tl.float32)
    gu = ((gn.to(tl.float32) - gsum.to(tl.float32)) / den).to(tl.bfloat16)
    idx = tl.load(indices_ptr + row * 8 + k)

    e = tl.arange(0, BLOCK)
    mask = e < E
    s = tl.load(scores_ptr + row * E + e, mask=mask, other=0.0)
    selected = e[:, None] == idx[None, :]
    gs = tl.sum(tl.where(selected, gu[None, :], 0.0).to(tl.float32), axis=1).to(tl.bfloat16)
    one_minus = (1.0 - s.to(tl.float32)).to(tl.bfloat16)
    first = (gs.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    out = (first.to(tl.float32) * one_minus.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row * E + e, out, mask=mask)


def _routing_grad(grad, scores, indices, norm, denom, scale):
    out = torch.empty_like(scores)
    _routing_grad_kernel[(scores.shape[0],)](
        grad, scores, indices, norm, denom, out,
        scale=scale, E=scores.shape[1], BLOCK=256, num_warps=4)
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
    grad_logits = _routing_grad(
        grad_topk_weights, scores, topk_indices,
        topk_weights_normalized, denominator, routed_scaling_factor)
    return grad_logits @ weight, grad_logits.t() @ hidden_states
