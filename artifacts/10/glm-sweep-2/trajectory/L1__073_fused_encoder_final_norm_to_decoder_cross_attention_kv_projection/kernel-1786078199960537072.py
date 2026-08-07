import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    n_rows, H: tl.constexpr,
    EPS: tl.constexpr,
    stride_xr, stride_xc,
    stride_or, stride_oc,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    x = tl.load(X_ptr + row * stride_xr + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)

    # variance
    mean_sq = tl.sum(x * x, axis=0) / H
    inv_rms = 1.0 / tl.sqrt(mean_sq + EPS)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (w * x) * inv_rms
    tl.store(OUT_ptr + row * stride_or + cols * stride_oc, out.to(tl.float16), mask=mask)


@torch.no_grad()
def _rmsnorm(x2d, weight, eps):
    n_rows, h = x2d.shape
    out = torch.empty_like(x2d)
    BLOCK = triton.next_power_of_2(h)
    grid = (n_rows,)
    _rmsnorm_kernel[grid](
        x2d, weight, out, n_rows, h, eps,
        x2d.stride(0), x2d.stride(1),
        out.stride(0), out.stride(1),
        BLOCK=BLOCK, num_warps=8,
    )
    return out


@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    bs, sl, h = encoder_hidden_states.shape
    num_kv_heads = 2
    head_dim = 64
    kv_hidden = num_kv_heads * head_dim

    x2d = encoder_hidden_states.reshape(-1, h).contiguous()
    normalized = _rmsnorm(x2d, norm_weight, eps)

    w_kv = torch.cat([k_proj_weight, v_proj_weight], dim=0)
    kv_flat = F.linear(normalized, w_kv, bias=None)

    keys = kv_flat[:, :kv_hidden].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    values = kv_flat[:, kv_hidden:].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    return keys, values
