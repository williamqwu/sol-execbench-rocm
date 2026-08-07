import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def _rmsnorm(x, weight, eps):
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
    kv_hidden = num_kv_heads * head_dim  # 128

    x2d = encoder_hidden_states.reshape(-1, hidden_size)
    normalized = _rmsnorm(x2d, norm_weight, eps)

    # Fuse the two GEMMs into one: stack weights -> [256, 1024]
    w_kv = torch.cat([k_proj_weight, v_proj_weight], dim=0)  # [256, 1024]
    kv_flat = F.linear(normalized, w_kv, bias=None)          # [N, 256]

    keys_flat = kv_flat[:, :kv_hidden]
    values_flat = kv_flat[:, kv_hidden:]

    keys = keys_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    keys = keys.transpose(1, 2).contiguous()
    values = values_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    values = values.transpose(1, 2).contiguous()
    return keys, values
