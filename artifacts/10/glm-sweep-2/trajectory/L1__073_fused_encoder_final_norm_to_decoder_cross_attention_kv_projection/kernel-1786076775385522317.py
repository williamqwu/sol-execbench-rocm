import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def _rmsnorm(x, weight, eps):
    # x: fp16 [N, H]
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight * xf).to(input_dtype)

@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, hidden_size = encoder_hidden_states.shape
    num_kv_heads = 2
    head_dim = 64

    x2d = encoder_hidden_states.reshape(-1, hidden_size)
    normalized = _rmsnorm(x2d, norm_weight, eps)

    keys_flat = F.linear(normalized, k_proj_weight, bias=None)
    values_flat = F.linear(normalized, v_proj_weight, bias=None)

    keys = keys_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    keys = keys.transpose(1, 2).contiguous()
    values = values_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    values = values.transpose(1, 2).contiguous()
    return keys, values
