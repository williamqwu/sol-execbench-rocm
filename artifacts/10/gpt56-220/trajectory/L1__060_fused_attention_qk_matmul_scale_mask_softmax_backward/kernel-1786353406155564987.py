import torch


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_weights: torch.Tensor,
    scaling: float,
):
    grad = torch.ops.aten._softmax_backward_data(
        grad_output.float(), attn_weights.float(), -1, torch.float32
    ).to(query.dtype)
    grad = grad * scaling
    return torch.matmul(grad, key), torch.matmul(grad.transpose(-2, -1), query)
