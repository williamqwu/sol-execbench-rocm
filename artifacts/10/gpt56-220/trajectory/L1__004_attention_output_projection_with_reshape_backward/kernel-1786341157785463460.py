import torch


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_size = grad_output.shape
    grad_output_2d = grad_output.reshape(-1, hidden_size)
    reshaped_2d = reshaped.reshape(-1, hidden_size)

    grad_weight = grad_output_2d.t().mm(reshaped_2d)
    grad_reshaped = grad_output_2d.mm(weight)
    grad_attn_output = (
        grad_reshaped.reshape(batch_size, seq_len, 32, 64)
        .transpose(1, 2)
        .contiguous()
    )
    return grad_attn_output, grad_weight
