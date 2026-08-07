"""Fused attention score-value matmul for MI355X (gfx950).

reference semantics:
    attn_output = attention_weights @ value        # [B,H,Q,K] @ [B,H,K,D] -> [B,H,Q,D]
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output.reshape(B, Q, H*D)

torch's bf16 matmul accumulates in fp32 and rounds the result to bf16 once, at
the end.  The transpose+contiguous that follows is a pure data movement: it
re-reads the whole [B,H,Q,D] result and writes it back out permuted.

So the reference touches, per call:
    read  A          B*H*Q*K*2
    read  V          B*H*K*D*2
    write tmp        B*H*Q*D*2
    read  tmp        B*H*Q*D*2
    write out        B*H*Q*D*2

This kernel writes the fp32 accumulator straight to its final transposed
address, which removes the tmp round-trip entirely (2 * B*Q*H*D*2 bytes) and
one kernel launch.  The rounding behaviour is unchanged: fp32 accumulate, a
single narrowing to bf16 on store, exactly as the reference does.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for block_q in (64, 128, 256):
        for block_k in (64, 128):
            for num_warps in (4, 8):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_Q": block_q, "BLOCK_K": block_k},
                        num_warps=num_warps,
                        num_stages=2,
                    )
                )
    return cfgs


@triton.autotune(
    configs=_configs(),
    key=["Q", "K", "NBH"],
)
@triton.heuristics(
    {
        "EVEN_Q": lambda a: a["Q"] % a["BLOCK_Q"] == 0,
        "EVEN_K": lambda a: a["K"] % a["BLOCK_K"] == 0,
    }
)
@triton.jit
def _av_fused_kernel(
    A_ptr,  # [B, H, Q, K]  bf16
    V_ptr,  # [B, H, K, D]  bf16
    O_ptr,  # [B, Q, H*D]   bf16
    Q,
    K,
    NBH,  # B*H, autotune key only
    stride_ab,
    stride_ah,
    stride_aq,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_ob,
    stride_oq,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_Q: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # program_id(0) is the fastest-varying dimension, so consecutive
    # workgroups share (b, h) and therefore share V.  That keeps the V tile
    # resident in cache across the Q-blocks of one head instead of re-fetching
    # it from HBM for each.
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D)

    a_ptrs = (
        A_ptr
        + b * stride_ab
        + h * stride_ah
        + offs_q[:, None] * stride_aq
        + offs_k[None, :]
    )
    v_ptrs = (
        V_ptr
        + b * stride_vb
        + h * stride_vh
        + offs_k[:, None] * stride_vk
        + offs_d[None, :]
    )

    q_mask = offs_q < Q

    acc = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        if EVEN_K:
            if EVEN_Q:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=q_mask[:, None], other=0.0)
            v = tl.load(v_ptrs)
        else:
            k_mask = (k0 + offs_k) < K
            if EVEN_Q:
                a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            else:
                a = tl.load(
                    a_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0
                )
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        # fp32 accumulator, bf16 MFMA operands -- same as torch's bf16 matmul.
        acc = tl.dot(a, v, acc)

        a_ptrs += BLOCK_K
        v_ptrs += BLOCK_K * stride_vk

    # Store straight into the [B, Q, H*D] layout: this *is* the transpose.
    o_ptrs = (
        O_ptr
        + b * stride_ob
        + offs_q[:, None] * stride_oq
        + (h * D + offs_d)[None, :]
    )
    out = acc.to(O_ptr.dtype.element_ty)
    if EVEN_Q:
        tl.store(o_ptrs, out)
    else:
        tl.store(o_ptrs, out, mask=q_mask[:, None])


def _reference_fallback(attention_weights: torch.Tensor, value: torch.Tensor):
    b = attention_weights.shape[0]
    q = attention_weights.shape[2]
    h = attention_weights.shape[1]
    d = value.shape[-1]
    o = torch.matmul(attention_weights, value)
    return o.transpose(1, 2).contiguous().reshape(b, q, h * d)


@torch.no_grad()
def run(
    attention_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    B, H, Q, K = attention_weights.shape
    D = value.shape[-1]

    # The kernel indexes the reduction axis of A and the D axis of V with unit
    # stride; anything else takes the torch path rather than reading garbage.
    if (
        attention_weights.stride(3) != 1
        or value.stride(3) != 1
        or D not in (16, 32, 64, 128)
        or K == 0
        or Q == 0
        or attention_weights.dtype != value.dtype
        or not attention_weights.is_cuda
    ):
        return _reference_fallback(attention_weights, value)

    out = torch.empty(
        (B, Q, H * D), dtype=attention_weights.dtype, device=attention_weights.device
    )

    grid = lambda meta: (triton.cdiv(Q, meta["BLOCK_Q"]), B * H)  # noqa: E731

    _av_fused_kernel[grid](
        attention_weights,
        value,
        out,
        Q,
        K,
        B * H,
        attention_weights.stride(0),
        attention_weights.stride(1),
        attention_weights.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        out.stride(0),
        out.stride(1),
        H=H,
        D=D,
    )
    return out
