import torch


@torch.compile(fullgraph=True, mode="max-autotune")
def _backward(grad_output, reshaped, weight):
    hidden_size = grad_output.shape[-1]
    grad_output_2d = grad_output.reshape(-1, hidden_size)
    reshaped_2d = reshaped.reshape(-1, hidden_size)
    return grad_output_2d.mm(weight), grad_output_2d.t().mm(reshaped_2d)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_size = grad_output.shape
    grad_reshaped, grad_weight = _backward(grad_output, reshaped, weight)
    grad_attn_output = (
        grad_reshaped.reshape(batch_size, seq_len, 32, 64)
        .transpose(1, 2)
    )
    return grad_attn_output, grad_weight
