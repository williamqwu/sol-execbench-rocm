import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _inv_rms_kernel(
    X_ptr, OUT_ptr,
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
    mean_sq = tl.sum(x * x, axis=0) / H
    inv_rms = 1.0 / tl.sqrt(mean_sq + EPS)
    tl.store(OUT_ptr + row * stride_or + cols * stride_oc, inv_rms, mask=cols == 0)


@torch.compile(dynamic=True)
def _fused(x, norm_weight, k_proj_weight, v_proj_weight, inv_rms):
    bs, sl, h = x.shape
    num_kv_heads = 2
    head_dim = 64
    kv_hidden = num_kv_heads * head_dim

    x2d = x.reshape(-1, h)
    w_kv = torch.cat([k_proj_weight, v_proj_weight], dim=0)
    w_folded = norm_weight[None, :] * w_kv

    kv_flat = ((x2d @ w_folded.t()).to(torch.float32) * inv_rms).to(x2d.dtype)

    keys = kv_flat[:, :kv_hidden].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    values = kv_flat[:, kv_hidden:].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    return keys, values


@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    bs, sl, h = encoder_hidden_states.shape
    x2d = encoder_hidden_states.reshape(-1, h).contiguous()
    n_rows = x2d.shape[0]

    inv_rms = torch.empty((n_rows, 1), device=x2d.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(h)
    _inv_rms_kernel[(n_rows,)](
        x2d, inv_rms, n_rows, h, eps,
        x2d.stride(0), x2d.stride(1),
        inv_rms.stride(0), inv_rms.stride(1),
        BLOCK=BLOCK, num_warps=4,
    )

    return _fused(encoder_hidden_states, norm_weight, k_proj_weight, v_proj_weight, inv_rms)
