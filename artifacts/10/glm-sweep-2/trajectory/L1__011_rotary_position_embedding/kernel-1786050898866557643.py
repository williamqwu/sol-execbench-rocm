import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    pos_ptr,          # [B, S] int64
    freq_ptr,         # [half] float32
    out_ptr,          # [B, S, head_dim, 2] bf16
    scaling,
    n_bs,
    S,
    pos_stride_b,
    pos_stride_s,
    out_stride_b,
    out_stride_s,
    out_stride_d,
    out_stride_c,
    BLOCK_S: tl.constexpr,
    HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, HALF)

    b = offs_s // S
    s = offs_s % S
    mask_bs = offs_s < n_bs

    pos = tl.load(pos_ptr + b * pos_stride_b + s * pos_stride_s,
                  mask=mask_bs, other=0.0).to(tl.float32)
    inv_freq = tl.load(freq_ptr + offs_h)

    freqs = pos[:, None] * inv_freq[None, :]
    c = tl.math.cos(freqs) * scaling
    sn = tl.math.sin(freqs) * scaling

    base = b * out_stride_b + s * out_stride_s
    d0 = offs_h * out_stride_d
    d1 = (offs_h + HALF) * out_stride_d

    # cos, first half
    tl.store(out_ptr + base[:, None] + d0[None, :], c,
             mask=mask_bs[:, None])
    # cos, second half
    tl.store(out_ptr + base[:, None] + d1[None, :], c,
             mask=mask_bs[:, None])
    # sin, first half
    tl.store(out_ptr + base[:, None] + d0[None, :] + out_stride_c, sn,
             mask=mask_bs[:, None])
    # sin, second half
    tl.store(out_ptr + base[:, None] + d1[None, :] + out_stride_c, sn,
             mask=mask_bs[:, None])


@torch.no_grad()
def run(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    B, S = position_ids.shape
    half = inv_freq.shape[0]
    head_dim = half * 2
    out = torch.empty((B, S, head_dim, 2), dtype=torch.bfloat16,
                      device=position_ids.device)

    BLOCK_S = 32
    grid = (triton.cdiv(B * S, BLOCK_S),)
    _rope_kernel[grid](
        position_ids, inv_freq, out,
        attention_scaling,
        B * S, S,
        position_ids.stride(0), position_ids.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_S=BLOCK_S, HALF=half,
        num_warps=2,
    )
    return out
