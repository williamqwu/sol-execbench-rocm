import torch
import triton
import triton.language as tl


@triton.jit
def fp8_blockwise_gemm_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, output_ptr,
    # Pointers to scales
    scale_x_ptr, scale_w_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    stride_sx_m, stride_sx_kb,
    stride_sw_nb, stride_sw_kb,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FP8 Blockwise-scaled GEMM kernel.

    Computes: output[M, N] = dequant(x[M, K]) @ dequant(w[N, K]).T

    Scale layout:
    - scale_x: [M, K//128] - BlockWise1x128
    - scale_w: [N//128, K//128] - BlockWise128x128
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for output block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for the output in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load x block [BLOCK_M, BLOCK_K] in FP8
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w block [BLOCK_N, BLOCK_K] in FP8
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Convert FP8 to float32
        x_f32 = x_fp8.to(tl.float32)
        w_f32 = w_fp8.to(tl.float32)

        # Load and apply scales for x (BlockWise1x128)
        # For each element at position (m, k), the scale is at (m, k // 128)
        offs_k_block = offs_k // 128
        scale_x_ptrs = scale_x_ptr + offs_m[:, None] * stride_sx_m + offs_k_block[None, :] * stride_sx_kb
        scale_x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        scale_x = tl.load(scale_x_ptrs, mask=scale_x_mask, other=1.0)
        x_f32 = x_f32 * scale_x

        # Load and apply scales for w (BlockWise128x128)
        # For each element at position (n, k), the scale is at (n // 128, k // 128)
        offs_n_block = offs_n // 128
        scale_w_ptrs = scale_w_ptr + offs_n_block[:, None] * stride_sw_nb + offs_k_block[None, :] * stride_sw_kb
        scale_w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        scale_w = tl.load(scale_w_ptrs, mask=scale_w_mask, other=1.0)
        w_f32 = w_f32 * scale_w

        # Perform matmul: acc += x_f32 @ w_f32.T
        acc += tl.dot(x_f32, tl.trans(w_f32))

    # Convert accumulator to bfloat16 and store
    output = acc.to(tl.bfloat16)

    # Store output
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, output, mask=out_mask)


def run(
    x: torch.Tensor,
    w: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
):
    """
    FP8 output projection GEMM with BlockWise1x128 activation scaling and BlockWise128x128 weight scaling.

    Args:
        x: Quantized input tensor [M, K] in FP8
        w: Quantized weight tensor [N, K] in FP8
        scale_x: BlockWise1x128 scales for input [M, K//128]
        scale_w: BlockWise128x128 scales for weights [N//128, K//128]

    Returns:
        Output tensor [M, N] in BF16
    """
    M, K = x.shape
    N, _ = w.shape

    # Allocate output
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    # Use working block sizes with tuning for num_warps and num_stages
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    fp8_blockwise_gemm_kernel[grid](
        x, w, output,
        scale_x, scale_w,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        output.stride(0), output.stride(1),
        scale_x.stride(0), scale_x.stride(1),
        scale_w.stride(0), scale_w.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
