import torch


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    if p_dropout > 0.0:
        grad = grad_output * dropout_mask.float() / (1.0 - p_dropout)
    else:
        grad = grad_output
    result = torch.ops.aten._softmax_backward_data(grad, p_attn, -1, torch.float32)
    return result.masked_fill(~mask, 0.0)
