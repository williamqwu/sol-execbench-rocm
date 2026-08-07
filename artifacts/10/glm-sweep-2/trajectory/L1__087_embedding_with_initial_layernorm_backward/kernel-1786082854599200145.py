import torch

@torch.no_grad()
def _run_impl(
    grad_output: torch.Tensor,
    input_ids: torch.Tensor,
    hidden_states_fp32: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
):
    vocab_size = 65536
    hidden_size = 4096

    grad_output_fp32 = grad_output.to(torch.float32)
    hidden_states_normalized = hidden_states_fp32 * rstd

    grad_norm_weight = (grad_output_fp32 * hidden_states_normalized).sum(dim=(0, 1))

    grad_hidden_states_normalized = grad_output_fp32 * norm_weight.to(torch.float32)
    mean_grad_normalized = (grad_hidden_states_normalized * hidden_states_normalized).mean(dim=-1, keepdim=True)
    grad_hidden_states_fp32 = rstd * (grad_hidden_states_normalized - mean_grad_normalized * hidden_states_normalized)

    grad_embed_weight = torch.zeros(vocab_size, hidden_size, dtype=torch.float32, device=grad_output.device)
    input_ids_flat = input_ids.view(-1)
    grad_hidden_states_flat = grad_hidden_states_fp32.view(-1, hidden_size)
    grad_embed_weight.index_add_(0, input_ids_flat, grad_hidden_states_flat)
    grad_embed_weight = grad_embed_weight.to(torch.bfloat16)
    grad_norm_weight = grad_norm_weight.to(torch.bfloat16)

    return grad_embed_weight, grad_norm_weight


_compiled = torch.compile(_run_impl, mode="max-autotune")


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    input_ids: torch.Tensor,
    hidden_states_fp32: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
):
    return _compiled(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight)
