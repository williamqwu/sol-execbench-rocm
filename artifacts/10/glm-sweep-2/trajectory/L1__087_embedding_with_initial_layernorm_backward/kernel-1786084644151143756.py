import torch

HIDDEN_SIZE = 4096
VOCAB_SIZE = 65536
_THRESH = 10000

_buf_fp32 = None


def _get_buf(device):
    global _buf_fp32
    if _buf_fp32 is None or _buf_fp32.device != device:
        _buf_fp32 = torch.empty(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float32, device=device)
    return _buf_fp32


@torch.no_grad()
def _elementwise(grad_output, hidden_states_fp32, rstd, norm_weight):
    grad_output_fp32 = grad_output.to(torch.float32)
    hidden_states_normalized = hidden_states_fp32 * rstd
    grad_norm_weight = (grad_output_fp32 * hidden_states_normalized).sum(dim=(0, 1))
    grad_hidden_states_normalized = grad_output_fp32 * norm_weight.to(torch.float32)
    mean_grad_normalized = (grad_hidden_states_normalized * hidden_states_normalized).mean(dim=-1, keepdim=True)
    grad_hidden_states_fp32 = rstd * (grad_hidden_states_normalized - mean_grad_normalized * hidden_states_normalized)
    return grad_hidden_states_fp32, grad_norm_weight


_elementwise_c = torch.compile(_elementwise, mode="max-autotune-no-cudagraphs")


@torch.no_grad()
def _run_large(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight):
    grad_output_fp32 = grad_output.to(torch.float32)
    hidden_states_normalized = hidden_states_fp32 * rstd
    grad_norm_weight = (grad_output_fp32 * hidden_states_normalized).sum(dim=(0, 1))
    grad_hidden_states_normalized = grad_output_fp32 * norm_weight.to(torch.float32)
    mean_grad_normalized = (grad_hidden_states_normalized * hidden_states_normalized).mean(dim=-1, keepdim=True)
    grad_hidden_states_fp32 = rstd * (grad_hidden_states_normalized - mean_grad_normalized * hidden_states_normalized)
    grad_embed_weight = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float32, device=grad_output.device)
    grad_embed_weight.index_add_(0, input_ids.reshape(-1), grad_hidden_states_fp32.reshape(-1, HIDDEN_SIZE))
    grad_embed_weight = grad_embed_weight.to(torch.bfloat16)
    grad_norm_weight = grad_norm_weight.to(torch.bfloat16)
    return grad_embed_weight, grad_norm_weight


_run_large_c = torch.compile(_run_large, mode="max-autotune-no-cudagraphs")


@torch.no_grad()
def run(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight):
    n_tokens = input_ids.numel()
    if n_tokens >= _THRESH:
        return _run_large_c(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight)

    grad_hidden_states_fp32, grad_norm_weight = _elementwise_c(
        grad_output, hidden_states_fp32, rstd, norm_weight
    )
    input_ids_flat = input_ids.reshape(-1)
    grad_hidden_states_flat = grad_hidden_states_fp32.reshape(-1, HIDDEN_SIZE)

    buf = _get_buf(grad_output.device)
    grad_embed_weight = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=grad_output.device)
    buf[input_ids_flat] = 0
    buf.index_add_(0, input_ids_flat, grad_hidden_states_flat)
    grad_embed_weight[input_ids_flat] = buf[input_ids_flat].to(torch.bfloat16)

    grad_norm_weight = grad_norm_weight.to(torch.bfloat16)
    return grad_embed_weight, grad_norm_weight
