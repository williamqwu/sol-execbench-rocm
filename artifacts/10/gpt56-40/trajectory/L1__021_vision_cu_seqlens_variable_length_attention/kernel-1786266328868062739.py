import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, cu_seqlens, cos, sin, qkv_weight, qkv_bias,
        proj_weight, proj_bias):
    n = hidden_states.shape[0]
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias).view(n, 3, 16, 80)
    q, k, v = qkv.unbind(1)
    c = cos[:, None, :]
    s = sin[:, None, :]
    # Avoid materializing the two concatenated rotated tensors.
    q0, q1 = q[..., :40], q[..., 40:]
    k0, k1 = k[..., :40], k[..., 40:]
    q = torch.cat((q0*c[..., :40] - q1*s[..., :40],
                   q1*c[..., 40:] + q0*s[..., 40:]), -1)
    k = torch.cat((k0*c[..., :40] - k1*s[..., :40],
                   k1*c[..., 40:] + k0*s[..., 40:]), -1)
    cu = cu_seqlens.to(torch.int32)
    out = torch.ops.aten._flash_attention_forward(
        q, k, v, cu, cu, n, n, 0.0, False, False,
        scale=80 ** -0.5)[0]
    return F.linear(out.reshape(n, 1280), proj_weight, proj_bias)
