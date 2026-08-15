import torch


def quantize_blockwise_1x128(x, scales):
    """Quantize with BlockWise1x128 scaling."""
    M, K = x.shape
    BLOCK_K = 128
    num_blocks = K // BLOCK_K
    E4M3_MAX = 448.0

    x_fp32 = x.to(torch.float32).view(M, num_blocks, BLOCK_K)
    x_scaled = (x_fp32 / scales.unsqueeze(2)).clamp(-E4M3_MAX, E4M3_MAX)
    return x_scaled.view(M, K).to(torch.float8_e4m3fn)


def quantize_blockwise_128x128(w, scales):
    """Quantize weight with BlockWise128x128 scaling."""
    N, K = w.shape
    BLOCK = 128
    num_k_blocks = K // BLOCK
    num_n_blocks = N // BLOCK
    E4M3_MAX = 448.0

    # Transpose to (K, N) for block scaling
    w_t = w.T.to(torch.float32)
    w_reshaped = w_t.view(num_k_blocks, BLOCK, num_n_blocks, BLOCK)
    w_scaled = (w_reshaped / scales.unsqueeze(1).unsqueeze(3)).clamp(-E4M3_MAX, E4M3_MAX)
    return w_scaled.view(K, N).T.to(torch.float8_e4m3fn)


def dequantize_blockwise_1x128(qx, scales):
    """Dequantize with BlockWise1x128 scaling."""
    M, K = qx.shape
    BLOCK_K = 128
    num_blocks = K // BLOCK_K

    qx_f32 = qx.to(torch.float32).view(M, num_blocks, BLOCK_K)
    return (qx_f32 * scales.unsqueeze(2)).view(M, K)


def dequantize_blockwise_128x128(qw, scales):
    """Dequantize weight with BlockWise128x128 scaling."""
    N, K = qw.shape
    BLOCK = 128
    num_k_blocks = K // BLOCK
    num_n_blocks = N // BLOCK

    qw_t = qw.T.to(torch.float32)
    qw_reshaped = qw_t.view(num_k_blocks, BLOCK, num_n_blocks, BLOCK)
    return (qw_reshaped * scales.unsqueeze(1).unsqueeze(3)).view(K, N).T


# Compile individual functions for better reuse across workloads
quantize_blockwise_1x128 = torch.compile(quantize_blockwise_1x128)
quantize_blockwise_128x128 = torch.compile(quantize_blockwise_128x128)
dequantize_blockwise_1x128 = torch.compile(dequantize_blockwise_1x128)
dequantize_blockwise_128x128 = torch.compile(dequantize_blockwise_128x128)


def run(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    scale_attn: torch.Tensor,
    scale_weight: torch.Tensor,
):
    """
    FP8 vision attention output projection with blockwise scaling.

    Uses compiled helper functions for quantization/dequantization
    to enable better kernel fusion and reuse.
    """
    # Quantize inputs
    qx = quantize_blockwise_1x128(attn_output, scale_attn)
    qw = quantize_blockwise_128x128(weight, scale_weight)

    # Dequantize
    qx_dequant = dequantize_blockwise_1x128(qx, scale_attn)
    qw_dequant = dequantize_blockwise_128x128(qw, scale_weight)

    # GEMM and convert to BF16
    output = torch.matmul(qx_dequant, qw_dequant.T)
    return output.to(torch.bfloat16)
