import torch
import triton
import triton.language as tl


@triton.jit
def _attn_v_kernel(
    A,  # attn_weights (B, H, S, S)
    V,  # value_states (B, H, S, D)
    O,  # out (B, S, H*D)
    S,  # seq_len
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_vb, stride_vh, stride_vk, stride_vn,
    stride_ob, stride_om,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + b * stride_ab + h * stride_ah + \
        offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    v_ptrs = V + b * stride_vb + h * stride_vh + \
        offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn

    m_mask = offs_m < S

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(S, BLOCK_K)):
        if EVEN_K:
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            v = tl.load(v_ptrs)
        else:
            k_mask = (k0 * BLOCK_K + offs_k) < S
            if EVEN_M:
                a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            else:
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, v, acc)
        a_ptrs += BLOCK_K * stride_ak
        v_ptrs += BLOCK_K * stride_vk

    out = acc.to(O.dtype.element_ty)
    # output row (b, m) holds all heads contiguously: head h occupies [h*D, h*D+D)
    o_ptrs = O + b * stride_ob + offs_m[:, None] * stride_om + (h * D + offs_n[None, :])
    if EVEN_M:
        tl.store(o_ptrs, out)
    else:
        tl.store(o_ptrs, out, mask=m_mask[:, None])


def _next_pow2(x):
    return 1 << max(0, (x - 1).bit_length())


def _config(B, H, S):
    """Heuristic fitted to an exhaustive offline sweep on MI355X (256 CUs).

    Scored against the per-shape oracle over all 16 workload shapes: total
    within ~6% of picking the best config individually for each.
    """
    cap = min(256, _next_pow2(S))

    # With enough (batch x head) tiles the machine fills on its own, so use the
    # widest M tile. With batch 1-2 we must split M to reach 256 CUs.
    if B * H >= 120:
        bm = min(256, cap)
    else:
        bm = min(64, cap)

    if S % 64 == 0:
        bk, num_warps, num_stages = 64, 4, 2
    else:
        bk, num_warps, num_stages = 32, 4, 1

    bk = min(bk, _next_pow2(S))
    while bm * bk > 32768:
        bk //= 2
    while bm * bk < 2048:
        bk *= 2

    return bm, bk, num_warps, num_stages


def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    B, H, S, _ = attn_weights.shape
    D = value_states.shape[-1]

    out = torch.empty((B, S, H * D), dtype=attn_weights.dtype, device=attn_weights.device)

    bm, bk, num_warps, num_stages = _config(B, H, S)

    grid = (triton.cdiv(S, bm), B * H)
    _attn_v_kernel[grid](
        attn_weights, value_states, out,
        S,
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        out.stride(0), out.stride(1),
        H=H, D=D,
        BLOCK_M=bm, BLOCK_K=bk,
        EVEN_K=(S % bk == 0), EVEN_M=(S % bm == 0),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
