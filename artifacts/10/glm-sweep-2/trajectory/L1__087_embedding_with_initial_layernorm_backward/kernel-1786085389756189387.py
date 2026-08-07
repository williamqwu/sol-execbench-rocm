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
def _run_small(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight, buf):
    grad_output_fp32 = grad_output.to(torch.float32)
    hidden_states_normalized = hidden_states_fp32 * rstd
    grad_norm_weight = (grad_output_fp32 * hidden_states_normalized).sum(dim=(0, 1))
    grad_hidden_states_normalized = grad_output_fp32 * norm_weight.to(torch.float32)
    mean_grad_normalized = (grad_hidden_states_normalized * hidden_states_normalized).mean(dim=-1, keepdim=True)
    grad_hidden_states_fp32 = rstd * (grad_hidden_states_normalized - mean_grad_normalized * hidden_states_normalized)

    ids = input_ids.reshape(-1)
    ghf = grad_hidden_states_fp32.reshape(-1, HIDDEN_SIZE)
    grad_embed_weight = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=grad_output.device)
    buf[ids] = 0
    buf.index_add_(0, ids, ghf)
    grad_embed_weight[ids] = buf[ids].to(torch.bfloat16)

    return grad_embed_weight, grad_norm_weight.to(torch.bfloat16)


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


_run_small_c = torch.compile(_run_small, mode="max-autotune-no-cudagraphs", dynamic=True)
_run_large_c = torch.compile(_run_large, mode="max-autotune-no-cudagraphs", dynamic=True)


@torch.no_grad()
def run(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight):
    n_tokens = input_ids.numel()
    if n_tokens < _THRESH:
        buf = _get_buf(grad_output.device)
        return _run_small_c(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight, buf)
    return _run_large_c(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight)
