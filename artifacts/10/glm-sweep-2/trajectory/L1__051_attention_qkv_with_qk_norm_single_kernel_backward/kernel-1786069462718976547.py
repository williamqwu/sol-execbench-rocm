import torch
import torch.nn.functional as F
import torch._inductor.config as inductor_config

inductor_config.coordinate_descent_tuning = True
inductor_config.autotune_multi_dot = True

@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
def _compiled_run(grad_query, grad_key, grad_value, hidden_states,
                  q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight,
                  q_rstd, k_rstd, q_normed, k_normed):
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256
    hidden_size = 640

    batch_size, seq_len, _ = hidden_states.shape

    grad_query_float = grad_query.float()
    q_scale = 1.0 + q_norm_weight.float()
    grad_q_norm_weight = (grad_query_float * q_normed.float()).sum(dim=(0, 1, 2))
    grad_q_normed = grad_query_float * q_scale
    q_normed_f = q_normed.float()
    q_mean_term = (grad_q_normed * q_normed_f).mean(dim=-1, keepdim=True)
    grad_q_transposed = (q_rstd * (grad_q_normed - q_mean_term * q_normed_f)).to(torch.bfloat16)

    grad_key_float = grad_key.float()
    k_scale = 1.0 + k_norm_weight.float()
    grad_k_norm_weight = (grad_key_float * k_normed.float()).sum(dim=(0, 1, 2))
    grad_k_normed = grad_key_float * k_scale
    k_normed_f = k_normed.float()
    k_mean_term = (grad_k_normed * k_normed_f).mean(dim=-1, keepdim=True)
    grad_k_transposed = (k_rstd * (grad_k_normed - k_mean_term * k_normed_f)).to(torch.bfloat16)

    grad_query_proj = grad_q_transposed.transpose(1, 2).contiguous().view(batch_size, seq_len, num_attention_heads * head_dim)
    grad_key_proj = grad_k_transposed.transpose(1, 2).contiguous().view(batch_size, seq_len, num_key_value_heads * head_dim)
    grad_value_proj = grad_value.transpose(1, 2).contiguous().view(batch_size, seq_len, num_key_value_heads * head_dim)

    grad_qkv = torch.cat([grad_query_proj, grad_key_proj, grad_value_proj], dim=-1)
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    grad_hidden_states = torch.matmul(grad_qkv, qkv_weight)

    grad_qkv_2d = grad_qkv.reshape(-1, grad_qkv.shape[-1])
    hidden_states_2d = hidden_states.reshape(-1, hidden_size)
    grad_qkv_weight = torch.matmul(grad_qkv_2d.t(), hidden_states_2d)
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    grad_q_weight = grad_qkv_weight[:q_size]
    grad_k_weight = grad_qkv_weight[q_size:q_size + kv_size]
    grad_v_weight = grad_qkv_weight[q_size + kv_size:]

    return (grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight,
            grad_q_norm_weight, grad_k_norm_weight)

@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_transposed: torch.Tensor,
    key_transposed: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    q_normed: torch.Tensor,
    k_normed: torch.Tensor,
    rms_norm_eps: float,
):
    return _compiled_run(grad_query, grad_key, grad_value, hidden_states,
                         q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight,
                         q_rstd, k_rstd, q_normed, k_normed)
