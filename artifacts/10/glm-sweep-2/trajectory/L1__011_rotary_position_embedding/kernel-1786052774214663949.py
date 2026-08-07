import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    pos_ptr, freq_ptr, out_ptr, scaling,
    S,
    pos_stride_b, pos_stride_s,
    out_stride_b, out_stride_s,
    BLOCK_S: tl.constexpr, HALF: tl.constexpr, HEAD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, HEAD)
    mask_s = offs_s < S
    pos = tl.load(pos_ptr + pid_b * pos_stride_b + offs_s * pos_stride_s,
                  mask=mask_s, other=0.0).to(tl.float32)
    d_mod = offs_d % HALF
    inv_freq_full = tl.load(freq_ptr + d_mod)
    freqs = pos[:, None] * inv_freq_full[None, :]
    c = tl.math.cos(freqs) * scaling
    sn = tl.math.sin(freqs) * scaling
    base = pid_b * out_stride_b + offs_s * out_stride_s
    off_d0 = offs_d * 2
    off_d1 = offs_d * 2 + 1
    m = mask_s[:, None]
    tl.store(out_ptr + base[:, None] + off_d0[None, :], c, mask=m)
    tl.store(out_ptr + base[:, None] + off_d1[None, ], sn, mask=m)


@torch.no_grad()
def run(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    B, S = position_ids.shape
    half = inv_freq.shape[0]
    head = half * 2
    out = torch.empty((B, S, head, 2), dtype=torch.bfloat16,
                      device=position_ids.device)
    BLOCK_S = 8
    grid = (B, triton.cdiv(S, BLOCK_S))
    _rope_kernel[grid](
        position_ids, inv_freq, out, attention_scaling,
        S,
        position_ids.stride(0), position_ids.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_S=BLOCK_S, HALF=half, HEAD=head, num_warps=4,
    )
    return out
