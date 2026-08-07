import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_norm_kv_kernel(
    X_ptr, NW_ptr, WK_ptr, WV_ptr, Kout_ptr, Vout_ptr,
    n_rows, H: tl.constexpr, KV: tl.constexpr,
    EPS: tl.constexpr,
    stride_xr, stride_xc,
    stride_kr, stride_kc,
    stride_vr, stride_vc,
    stride_outr, stride_outc,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    # ---- RMSNorm reduction over H ----
    # load whole row in chunks of BLOCK_H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for off in range(0, H, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < H
        x = tl.load(X_ptr + row * stride_xr + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    # reduce acc to scalar
    var = tl.sum(acc, axis=0) / H
    inv_rms = 1.0 / tl.sqrt(var + EPS)

    # ---- Compute outputs: out_kv = sum_h (nw[h]*x[h]/rms) * W[kv,h] ----
    # We accumulate into KV-length vectors.
    out_k = tl.zeros((KV,), dtype=tl.float32)
    out_v = tl.zeros((KV,), dtype=tl.float32)

    # tile over H dimension
    for off in range(0, H, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        mask = cols < H
        x = tl.load(X_ptr + row * stride_xr + cols * stride_xc, mask=mask, other=0.0).to(tl.float32)
        nw = tl.load(NW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        xn = (nw * x) * inv_rms  # (BLOCK_H,) fp32 normalized

        # WK: (KV, H)  -> load column block [KV, BLOCK_H]
        # indices: k in [0,KV), c in cols
        kv_idx = tl.arange(0, KV)
        c_idx = cols  # (BLOCK_H,)
        # WK_ptr[k, c] at WK_ptr + k*stride_kr + c*stride_kc
        w_k = tl.load(
            WK_ptr + kv_idx[:, None] * stride_kr + c_idx[None, :] * stride_kc,
            mask=mask[None, :], other=0.0
        ).to(tl.float32)
        w_v = tl.load(
            WV_ptr + kv_idx[:, None] * stride_vr + c_idx[None, :] * stride_vc,
            mask=mask[None, :], other=0.0
        ).to(tl.float32)

        out_k += tl.sum(w_k * xn[None, :], axis=1)
        out_v += tl.sum(w_v * xn[None, :], axis=1)

    # store
    kv_idx = tl.arange(0, KV)
    tl.store(Kout_ptr + row * stride_outr + kv_idx * stride_outc, out_k.to(tl.float16))
    tl.store(Vout_ptr + row * stride_outr + kv_idx * stride_outc, out_v.to(tl.float16))


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

    x2d = encoder_hidden_states.reshape(-1, hidden_size).contiguous()
    n_rows = x2d.shape[0]
    assert hidden_size == 1024
    assert kv_hidden == 128

    keys_flat = torch.empty((n_rows, kv_hidden), device=x2d.device, dtype=torch.float16)
    values_flat = torch.empty((n_rows, kv_hidden), device=x2d.device, dtype=torch.float16)

    BLOCK_H = 1024
    grid = (n_rows,)
    _fused_norm_kv_kernel[grid](
        x2d, norm_weight, k_proj_weight, v_proj_weight, keys_flat, values_flat,
        n_rows, hidden_size, kv_hidden,
        eps,
        x2d.stride(0), x2d.stride(1),
        k_proj_weight.stride(0), k_proj_weight.stride(1),
        v_proj_weight.stride(0), v_proj_weight.stride(1),
        keys_flat.stride(0), keys_flat.stride(1),
        BLOCK_H=BLOCK_H,
        num_warps=4,
    )

    keys = keys_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    keys = keys.transpose(1, 2).contiguous()
    values = values_flat.view(batch_size, seq_len, num_kv_heads, head_dim)
    values = values.transpose(1, 2).contiguous()
    return keys, values
