import torch
import triton
import triton.language as tl

NUM_HEADS = 20
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM


@triton.jit
def _fused_kernel(
    aw_ptr, v_ptr, o_ptr,
    B, Q, K,
    saw_b, saw_h, saw_q, saw_k,
    sv_b, sv_h, sv_k, sv_d,
    so_b, so_q, so_hd,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_q < Q

    o_offs = (
        pid_b * so_b
        + offs_q[:, None] * so_q
        + (pid_h * HEAD_DIM + offs_d[None, :]) * so_hd
    )

    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        aw_offs = (
            pid_b * saw_b + pid_h * saw_h
            + offs_q[:, None] * saw_q + offs_k[None, :] * saw_k
        )
        aw = tl.load(aw_ptr + aw_offs, mask=q_mask[:, None] & k_mask[None, :], other=0.0)

        v_offs = (
            pid_b * sv_b + pid_h * sv_h
            + offs_k[:, None] * sv_k + offs_d[None, :] * sv_d
        )
        v = tl.load(v_ptr + v_offs, mask=k_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0)

        acc += tl.dot(aw, v)

    tl.store(o_ptr + o_offs, acc.to(o_ptr.dtype.element_ty),
             mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _transpose_kernel(
    in_ptr, o_ptr,
    B, Q,
    si_b, si_h, si_q, si_d,
    so_b, so_q, so_hd,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_HEADS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_q < Q

    i_offs = (
        pid_b * si_b + pid_h * si_h
        + offs_q[:, None] * si_q + offs_d[None, :] * si_d
    )
    x = tl.load(in_ptr + i_offs, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0)

    o_offs = (
        pid_b * so_b + offs_q[:, None] * so_q
        + (pid_h * HEAD_DIM + offs_d[None, :]) * so_hd
    )
    tl.store(o_ptr + o_offs, x, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM))


@torch.no_grad()
def run(attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    B, H, Q, K = attention_weights.shape

    output = torch.empty((B, Q, HIDDEN), dtype=attention_weights.dtype, device=attention_weights.device)

    # Large compute-bound self-attention: hipBLASLt GEMM + fused Triton transpose.
    flops = B * H * Q * K * HEAD_DIM
    if flops >= 8e9:
        mat = torch.matmul(attention_weights, value)  # [B, H, Q, D]
        BLOCK_Q = 64
        grid = (B, H, triton.cdiv(Q, BLOCK_Q))
        _transpose_kernel[grid](
            mat, output, B, Q,
            mat.stride(0), mat.stride(1), mat.stride(2), mat.stride(3),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_Q=BLOCK_Q, BLOCK_D=HEAD_DIM, HEAD_DIM=HEAD_DIM, NUM_HEADS=NUM_HEADS,
            num_warps=4,
        )
        return output

    # Fused matmul writing directly into [B, Q, H*D].
    # Cross-attention (small K): wider Q tile + more warps for bandwidth.
    if K <= 128:
        BLOCK_Q, BLOCK_K, num_warps, num_stages = 128, 128, 8, 2
    else:
        BLOCK_Q, BLOCK_K, num_warps, num_stages = 64, 64, 4, 2

    grid = (B, H, triton.cdiv(Q, BLOCK_Q))
    _fused_kernel[grid](
        attention_weights, value, output, B, Q, K,
        attention_weights.stride(0), attention_weights.stride(1),
        attention_weights.stride(2), attention_weights.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=HEAD_DIM, HEAD_DIM=HEAD_DIM,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output
