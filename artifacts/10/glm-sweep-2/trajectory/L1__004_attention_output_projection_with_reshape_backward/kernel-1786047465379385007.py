import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    head_dim = 64

    grad_output_2d = grad_output.reshape(-1, hidden_size)
    reshaped_2d = reshaped.reshape(-1, hidden_size)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        grad_weight = grad_output_2d.t().mm(reshaped_2d)
    grad_reshaped_2d = grad_output_2d.mm(weight)
    torch.cuda.current_stream().wait_stream(s)

    grad_transposed = grad_reshaped_2d.reshape(batch_size, seq_len, num_heads, head_dim)
    grad_attn_output = grad_transposed.transpose(1, 2)

    return grad_attn_output, grad_weight
