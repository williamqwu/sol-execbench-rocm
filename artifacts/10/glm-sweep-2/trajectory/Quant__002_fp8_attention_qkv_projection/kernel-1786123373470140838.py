import torch

E4M3_MAX = 448.0

def _compute_scales_1x128(tensor):
    M, K = tensor.shape
    return (tensor.reshape(M, K // 128, 128).abs().amax(2) / E4M3_MAX).clamp(min=1e-12)

def _compute_scales_128x128_wt(weight_t):
    K, N = weight_t.shape
    return (weight_t.reshape(K // 128, 128, N // 128, 128).abs().amax(3).amax(1) / E4M3_MAX).clamp(min=1e-12)

def _fp8_linear_bf16(x_fp32, weight_fp32, bias, scale_x, scales_w_t):
    M, K = x_fp32.shape
    N = weight_fp32.shape[0]
    Kb = K // 128
    sw_b = scales_w_t.T.contiguous()
    xq = (x_fp32.reshape(M, Kb, 128) / scale_x.unsqueeze(2)).clamp(-E4M3_MAX, E4M3_MAX).reshape(M, K)
    qx = xq.to(torch.float8_e4m3fn)
    bq = (weight_fp32.reshape(N // 128, 128, Kb, 128) / sw_b.unsqueeze(1).unsqueeze(3)).clamp(-E4M3_MAX, E4M3_MAX).reshape(N, K)
    qw = bq.to(torch.float8_e4m3fn)
    da = (qx.to(torch.float32).reshape(M, Kb, 128) * scale_x.unsqueeze(2)).reshape(M, K).to(torch.bfloat16)
    db = (qw.to(torch.float32).reshape(N // 128, 128, Kb, 128) * sw_b.unsqueeze(1).unsqueeze(3)).reshape(N, K).to(torch.bfloat16)
    y = da @ db.T
    if bias is not None and bias.numel():
        y = y + bias
    return y

def _rms_norm(hidden_states, weight, eps):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)

@torch.no_grad()
def _run_impl(hidden_states, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias,
        q_norm_weight, k_norm_weight, rms_norm_eps):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 60
    num_key_value_heads = 8
    head_dim = 128
    hidden_states_flat = hidden_states.reshape(-1, hidden_size)
    x_fp32 = hidden_states_flat.to(torch.float32)
    scale_x = _compute_scales_1x128(x_fp32)
    q_w_f32 = q_weight.to(torch.float32)
    k_w_f32 = k_weight.to(torch.float32)
    v_w_f32 = v_weight.to(torch.float32)
    sw_q = _compute_scales_128x128_wt(q_w_f32.T)
    sw_k = _compute_scales_128x128_wt(k_w_f32.T)
    sw_v = _compute_scales_128x128_wt(v_w_f32.T)
    query_states = _fp8_linear_bf16(x_fp32, q_w_f32, q_bias, scale_x, sw_q)
    key_states = _fp8_linear_bf16(x_fp32, k_w_f32, k_bias, scale_x, sw_k)
    value_states = _fp8_linear_bf16(x_fp32, v_w_f32, v_bias, scale_x, sw_v)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    query_states = _rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = _rms_norm(key_states, k_norm_weight, rms_norm_eps)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    return query_states, key_states, value_states

_run_compiled = torch.compile(_run_impl, mode="max-autotune", fullgraph=True)

@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias,
        q_norm_weight, k_norm_weight, rms_norm_eps):
    return _run_compiled(hidden_states, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias,
                         q_norm_weight, k_norm_weight, rms_norm_eps)
