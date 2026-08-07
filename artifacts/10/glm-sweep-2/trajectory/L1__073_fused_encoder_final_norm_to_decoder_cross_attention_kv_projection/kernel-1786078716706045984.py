import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_norm_mm_kernel(
    X_ptr, NW_ptr, W_ptr, OUT_ptr,
    n_rows, H: tl.constexpr, KV: tl.constexpr,
    EPS: tl.constexpr,
    stride_xr, stride_xc,
    stride_wr, stride_wc,
    stride_or, stride_oc,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_M

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < n_rows
    mask_n_col = offs_n < KV

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    nw = tl.load(NW_ptr + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < H, other=0.0).to(tl.float32)

    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xr + offs_k[None, :] * stride_xc,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        sq += tl.sum(x * x, axis=1)
        xn = nw[None, :] * x
        w = tl.load(
            W_ptr + offs_n[:, None] * stride_wr + offs_k[None, :] * stride_wc,
            mask=mask_n_col[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        acc += tl.dot(xn, tl.trans(w))

    inv_rms = 1.0 / tl.sqrt(sq / H + EPS)
    out = acc * inv_rms[:, None]
    tl.store(
        OUT_ptr + offs_m[:, None] * stride_or + offs_n[None, :] * stride_oc,
        out.to(tl.float16),
        mask=mask_m[:, None] & mask_n_col[None, :],
    )


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
    kv_hidden = num_kv_heads * head_dim  # 128

    x2d = encoder_hidden_states.reshape(-1, h).contiguous()
    n_rows = x2d.shape[0]

    w_kv = torch.cat([k_proj_weight, v_proj_weight], dim=0).contiguous()  # [256, 1024]

    out = torch.empty((n_rows, 2 * kv_hidden), device=x2d.device, dtype=torch.float16)

    BLOCK_M = 16
    BLOCK_K = triton.next_power_of_2(h)  # 1024
    BLOCK_N = 256
    grid = (triton.cdiv(n_rows, BLOCK_M),)
    _fused_norm_mm_kernel[grid](
        x2d, norm_weight, w_kv, out,
        n_rows, h, 2 * kv_hidden, eps,
        x2d.stride(0), x2d.stride(1),
        w_kv.stride(0), w_kv.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )

    keys = out[:, :kv_hidden].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    values = out[:, kv_hidden:].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    return keys, values
