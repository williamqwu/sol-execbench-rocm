import torch

E4M3_MAX = 448.0


@torch.compile(dynamic=True)
def _act_quant_dequant(hidden_states_f32):
    """Compute 1x128 scales, quantize to FP8, dequant back to bf16."""
    M, K = hidden_states_f32.shape
    x_blocked = hidden_states_f32.view(M, K // 128, 128)
    block_max = x_blocked.abs().amax(dim=2)
    scale_x = torch.clamp(block_max / E4M3_MAX, min=1e-12)
    x_scaled = x_blocked / scale_x.unsqueeze(2)
    x_scaled = torch.clamp(x_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    qx = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    a_f32 = qx.to(torch.float32).view(M, K // 128, 128)
    a_f32 = a_f32 * scale_x.unsqueeze(2)
    return a_f32.reshape(M, K).to(torch.bfloat16)


@torch.compile(dynamic=True)
def _weight_quant_dequant(w_fp32):
    """w_fp32: (K, N). 128x128 scales. Quantize to FP8, dequant to bf16 (N,K)."""
    K, N = w_fp32.shape
    w_blocked = w_fp32.view(K // 128, 128, N // 128, 128)
    w_block_max = w_blocked.abs().amax(dim=3).amax(dim=1)
    weight_scales = torch.clamp(w_block_max / E4M3_MAX, min=1e-12)
    w_scaled = w_blocked / weight_scales.unsqueeze(1).unsqueeze(3)
    w_scaled = torch.clamp(w_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    b_f32 = w_scaled.reshape(K, N)
    return b_f32.T.contiguous().to(torch.bfloat16)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    num_heads = 16
    head_dim = 96
    seq_length = hidden_states.shape[0]

    x_f32 = hidden_states.to(torch.float32)
    w_fp32 = qkv_weight.T.to(torch.float32)

    a_bf16 = _act_quant_dequant(x_f32)
    b_bf16 = _weight_quant_dequant(w_fp32)

    qkv = a_bf16 @ b_bf16.t()
    qkv = qkv + qkv_bias

    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=1)
    return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()
