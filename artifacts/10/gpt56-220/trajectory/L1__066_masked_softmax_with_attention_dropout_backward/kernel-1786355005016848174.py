import torch


@torch.compile(fullgraph=True, dynamic=True)
def _backward(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    grad = grad_output * dropout_mask * scale
    sum_term = (p_attn * grad).sum(dim=-1, keepdim=True)
    return torch.where(mask, p_attn * (grad - sum_term), 0.0)


@torch.compile(fullgraph=True, dynamic=True)
def _backward_no_dropout(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    sum_term = (p_attn * grad_output).sum(dim=-1, keepdim=True)
    return torch.where(mask, p_attn * (grad_output - sum_term), 0.0)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    if p_dropout > 0.0:
        return _backward(
            grad_output, p_attn, mask, dropout_mask, 1.0 / (1.0 - p_dropout)
        )
    return _backward_no_dropout(grad_output, p_attn, mask)
