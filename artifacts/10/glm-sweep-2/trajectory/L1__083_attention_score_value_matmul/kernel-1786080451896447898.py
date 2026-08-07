import torch
import triton
import triton.language as tl

NUM_HEADS = 20
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM


@triton.jit
def _attn_kv_kernel(
    # attention_weights: [B, H, Q, K]  -> strides
    aw_ptr,
    # value: [B, H, K, D] -> strides
    v_ptr,
    # output: [B, Q, H*D] -> strides
    o_ptr,
    B, Q, K,
    stride_aw_b, stride_aw_h, stride_aw_q, stride_aw_k,
    stride_v_b, stride_v_h, stride_v_k, stride_v_d,
    stride_o_b, stride_o_q, stride_o_hd,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = offs_q < Q

    # output offsets: output[b, q, h*D + d]
    o_offs = (
        pid_b * stride_o_b
        + offs_q[:, None] * stride_o_q
        + (pid_h * HEAD_DIM + offs_d[None, :]) * stride_o_hd
    )

    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # aw[b, h, q, k]
        aw_offs = (
            pid_b * stride_aw_b
            + pid_h * stride_aw_h
            + offs_q[:, None] * stride_aw_q
            + offs_k[None, :] * stride_aw_k
        )
        aw = tl.load(aw_offs, mask=q_mask[:, None] & k_mask[None, :], other=0.0)

        # v[b, h, k, d]
        v_offs = (
            pid_b * stride_v_b
            + pid_h * stride_v_h
            + offs_k[:, None] * stride_v_k
            + offs_d[None, :] * stride_v_d
        )
        v = tl.load(v_offs, mask=k_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0)

        acc += tl.dot(aw, v)

    o_mask = q_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
    tl.store(o_offs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


@torch.no_grad()
def run(attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    B, H, Q, K = attention_weights.shape
    assert H == NUM_HEADS
    D = value.shape[3]
    assert D == HEAD_DIM

    output = torch.empty((B, Q, HIDDEN), dtype=attention_weights.dtype, device=attention_weights.device)

    BLOCK_Q = 64
    BLOCK_K = 64
    BLOCK_D = 64

    grid = (B, H, triton.cdiv(Q, BLOCK_Q))

    _attn_kv_kernel[grid](
        attention_weights, value, output,
        B, Q, K,
        attention_weights.stride(0), attention_weights.stride(1), attention_weights.stride(2), attention_weights.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        NUM_HEADS=NUM_HEADS, HEAD_DIM=HEAD_DIM,
        num_warps=4, num_stages=3,
    )
    return output
