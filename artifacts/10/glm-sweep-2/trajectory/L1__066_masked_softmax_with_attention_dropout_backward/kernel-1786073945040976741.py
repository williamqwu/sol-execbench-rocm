import torch

@torch.compile(dynamic=True)
def _fwd(grad_output, p_attn, mask, dropout_mask, p_dropout):
    if p_dropout > 0.0:
        grad_softmax_output = grad_output * dropout_mask.to(grad_output.dtype) / (1.0 - p_dropout)
    else:
        grad_softmax_output = grad_output
    sum_term = (p_attn * grad_softmax_output).sum(dim=-1, keepdim=True)
    grad_softmax_input = p_attn * (grad_softmax_output - sum_term)
    grad_scores = grad_softmax_input.masked_fill(~mask, 0.0)
    return grad_scores


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    return _fwd(grad_output, p_attn, mask, dropout_mask, p_dropout)
