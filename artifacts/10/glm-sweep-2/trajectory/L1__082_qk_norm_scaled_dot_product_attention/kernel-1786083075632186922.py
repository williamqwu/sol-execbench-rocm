import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr, out_ptr, scale,
               D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + pid * D + offs, mask=mask, other=0.0)
    m = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    normalized = (x - m) / rstd
    out = normalized * w + b
    out = out * scale
    tl.store(out_ptr + pid * D + offs, out, mask=mask)


def _triton_ln(x, mean, rstd, w, b, scale):
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    out = torch.empty_like(x2)
    N = x2.shape[0]
    BLOCK = triton.next_power_of_2(D)
    _ln_kernel[(N,)](x2, mean.reshape(-1), rstd.reshape(-1), w, b, out,
                     scale, D=D, BLOCK=BLOCK)
    return out.reshape_as(x)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, dim = hidden_states.shape
    num_heads = 24
    head_dim = 64
    scale = head_dim ** -0.5

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Fuse mean+var for q and k in one pass each (concatenated along batch)
    qk = torch.cat([q, k], dim=0)
    qk_mean = qk.mean(dim=-1, keepdim=True)
    qk_rstd = torch.sqrt(qk.var(dim=-1, unbiased=False, keepdim=True) + eps)
    q_mean, k_mean = qk_mean[:batch_size], qk_mean[batch_size:]
    q_rstd, k_rstd = qk_rstd[:batch_size], qk_rstd[batch_size:]

    q = _triton_ln(q, q_mean, q_rstd, q_norm_weight, q_norm_bias, scale)
    k = _triton_ln(k, k_mean, k_rstd, k_norm_weight, k_norm_bias, 1.0)

    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    return output
