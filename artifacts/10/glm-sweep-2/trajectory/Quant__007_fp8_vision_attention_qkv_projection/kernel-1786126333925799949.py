import torch

E4M3_MAX = 448.0


def _dequant_act_1x128(qx, scale_x):
    M, K = qx.shape
    a_f32 = qx.to(torch.float32)
    a_f32 = a_f32.view(M, K // 128, 128)
    a_f32 = a_f32 * scale_x.unsqueeze(2)
    return a_f32.reshape(M, K).to(torch.bfloat16)


def _dequant_weight_128x128(qw, scale_w):
    N, K = qw.shape
    b_f32 = qw.to(torch.float32)
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
    seq_length = hidden_states.shape[0]

    x_f32 = hidden_states.to(torch.float32)
    M, K = x_f32.shape
    x_blocked = x_f32.view(M, K // 128, 128)
    block_max = x_blocked.abs().amax(dim=2)
    scale_x = torch.clamp(block_max / E4M3_MAX, min=1e-12)
    x_scaled = x_blocked / scale_x.unsqueeze(2)
    x_scaled = torch.clamp(x_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    qx = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)

    w_fp32 = qkv_weight.T.to(torch.float32)
    K2, N = w_fp32.shape
    w_blocked = w_fp32.view(K2 // 128, 128, N // 128, 128)
    w_block_max = w_blocked.abs().amax(dim=3).amax(dim=1)
    weight_scales = torch.clamp(w_block_max / E4M3_MAX, min=1e-12)
    w_scaled = w_blocked / weight_scales.unsqueeze(1).unsqueeze(3)
    w_scaled = torch.clamp(w_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    qw = w_scaled.reshape(K2, N).T.contiguous().to(torch.float8_e4m3fn)
    scale_w_nk = weight_scales.T.contiguous()

    a_bf16 = _dequant_act_1x128(qx, scale_x)
    b_bf16 = _dequant_weight_128x128(qw, scale_w_nk)

    qkv = a_bf16 @ b_bf16.t()
    qkv = qkv + qkv_bias

    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=1)
    return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()
