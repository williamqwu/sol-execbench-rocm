import torch

_compiled = None

def _run(grad_output, reshaped, weight):
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    head_dim = 64

    grad_output_2d = grad_output.reshape(-1, hidden_size)
    reshaped_2d = reshaped.reshape(-1, hidden_size)

    grad_weight = grad_output_2d.t().mm(reshaped_2d)
    grad_reshaped_2d = grad_output_2d.mm(weight)

    grad_transposed = grad_reshaped_2d.reshape(batch_size, seq_len, num_heads, head_dim)
    grad_attn_output = grad_transposed.transpose(1, 2)

    return grad_attn_output, grad_weight

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_run, dynamic=True, mode="max-autotune-no-cudagraphs")
    return _compiled(grad_output, reshaped, weight)
