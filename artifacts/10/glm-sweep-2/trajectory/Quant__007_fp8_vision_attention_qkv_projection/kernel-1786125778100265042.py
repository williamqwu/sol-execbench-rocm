import torch

E4M3_MAX = 448.0


def _dequant_act_1x128(qx, scale_x):
    """Dequantize FP8 activation (M,K) with 1x128 block scales (M, K//128).
    Returns bf16 (M,K)."""
    M, K = qx.shape
    # scale_x: (M, K//128) -> broadcast to (M, K)
    # dequant: fp8_val * scale (multiply, since scale = amax/448 is the inverse scale)
    a_f32 = qx.to(torch.float32)
    # reshape (M, K//128, 128)
    a_f32 = a_f32.view(M, K // 128, 128)
    a_f32 = a_f32 * scale_x.unsqueeze(2)
    return a_f32.reshape(M, K).to(torch.bfloat16)


def _dequant_weight_128x128(qw, scale_w):
    """Dequantize FP8 weight (N,K) with 128x128 block scales (N//128, K//128).
    Returns bf16 (N,K)."""
    N, K = qw.shape
    b_f32 = qw.to(torch.float32)
    # reshape (N//128, 128, K//128, 128)
    b_f32 = b_f32.view(N // 128, 128, K // 128, 128)
    b_f32 = b_f32 * scale_w.unsqueeze(1).unsqueeze(3)
    return b_f32.reshape(N, K).to(torch.bfloat16)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    num_heads = 16
    head_dim = 96
    hidden_size = 1536
    seq_length = hidden_states.shape[0]

    # --- Activation quantization (1x128) ---
    x_f32 = hidden_states.to(torch.float32)
    # compute 1x128 block scales: (seq_len, hidden_size//128)
    M, K = x_f32.shape
    x_blocked = x_f32.view(M, K // 128, 128)
    block_max = x_blocked.abs().amax(dim=2)
    scale_x = torch.clamp(block_max / E4M3_MAX, min=1e-12)
    # quantize
    x_scaled = x_blocked / scale_x.unsqueeze(2)
    x_scaled = torch.clamp(x_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    qx = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)

    # --- Weight quantization (128x128) ---
    # qkv_weight: (N, hidden_size). Transpose to (hidden_size, N) for blocking.
    w_fp32 = qkv_weight.T.to(torch.float32)  # (K, N) = (1536, 4608)
    K2, N = w_fp32.shape
    w_blocked = w_fp32.view(K2 // 128, 128, N // 128, 128)
    w_block_max = w_blocked.abs().amax(dim=3).amax(dim=1)  # (K2//128, N//128)
    weight_scales = torch.clamp(w_block_max / E4M3_MAX, min=1e-12)
    w_scaled = w_blocked / weight_scales.unsqueeze(1).unsqueeze(3)
    w_scaled = torch.clamp(w_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    # qw as (N, K) row-major: transpose the dequantized weight
    qw = w_scaled.reshape(K2, N).T.contiguous().to(torch.float8_e4m3fn)

    # --- Dequant + GEMM (bf16) ---
    # scale_w for (N,K) layout: weight_scales is (K//128, N//128); need (N//128, K//128)
    scale_w_nk = weight_scales.T.contiguous()

    a_bf16 = _dequant_act_1x128(qx, scale_x)           # (M, K) bf16
    b_bf16 = _dequant_weight_128x128(qw, scale_w_nk)   # (N, K) bf16

    # matmul: a (M,K) @ b.T (K,N) -> (M,N), bf16 with fp32 accum
    qkv = a_bf16 @ b_bf16.t()
    qkv = qkv + qkv_bias  # broadcast (N,)

    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=1)
    return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()
