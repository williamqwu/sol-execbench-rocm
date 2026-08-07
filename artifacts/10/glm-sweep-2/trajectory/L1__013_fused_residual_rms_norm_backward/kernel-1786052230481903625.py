import torch

@torch.compile(dynamic=True, fullgraph=True)
def _compiled(grad_output, normalized, rstd, weight):
    grad_output_f32 = grad_output.to(torch.float32)
    grad_weight = (grad_output_f32 * normalized).sum(dim=(0, 1))
    grad_normalized = grad_output_f32 * weight
    mean_grad_norm = (grad_normalized * normalized).mean(dim=-1, keepdim=True)
    grad_x = rstd * (grad_normalized - mean_grad_norm * normalized)
    grad_x_bf16 = grad_x.to(torch.bfloat16)
    return grad_x_bf16, grad_x_bf16.clone(), grad_weight


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, normalized: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor):
    grad_hidden_states, grad_residual, grad_weight = _compiled(grad_output, normalized, rstd, weight)
    return grad_hidden_states, grad_residual, grad_weight
