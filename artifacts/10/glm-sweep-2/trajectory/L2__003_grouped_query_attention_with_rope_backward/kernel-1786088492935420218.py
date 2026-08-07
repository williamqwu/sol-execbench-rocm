import torch
import torch.nn.functional as F

def _run_impl(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    scaling: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    kv_seq_len = key_states.shape[2]

    grad_attn_output = torch.matmul(grad_output, o_weight)
    grad_o_weight = torch.matmul(
        grad_output.reshape(-1, grad_output.shape[-1]).t(),
        attn_output.reshape(-1, attn_output.shape[-1])
    )

    grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, num_attention_heads, head_dim)
    grad_attn_output = grad_attn_output.transpose(1, 2)

    grad_attn_weights = torch.matmul(grad_attn_output, value_states.transpose(2, 3))
    grad_value_states = torch.matmul(attn_weights.transpose(2, 3), grad_attn_output)

    attn_weights_fp32 = attn_weights.to(torch.float32)
    grad_attn_weights_fp32 = grad_attn_weights.to(torch.float32)
    sum_grad = (grad_attn_weights_fp32 * attn_weights_fp32).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights_fp32 * (grad_attn_weights_fp32 - sum_grad)
    grad_attn_scores = grad_attn_scores.to(query_states.dtype)

    grad_attn_scores = grad_attn_scores * scaling

    grad_query_states = torch.matmul(grad_attn_scores, key_states)
    grad_key_states = torch.matmul(grad_attn_scores.transpose(2, 3), query_states)

    if num_key_value_groups != 1:
        grad_key_states = grad_key_states.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim
        ).sum(dim=2)
        grad_value_states = grad_value_states.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim
        ).sum(dim=2)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    grad_q_cos = grad_query_states * cos_expanded
    grad_q_sin = grad_query_states * sin_expanded
    grad_q_sin_1 = grad_q_sin[..., : head_dim // 2]
    grad_q_sin_2 = grad_q_sin[..., head_dim // 2 :]
    grad_q_sin_rotated = torch.cat((grad_q_sin_2, -grad_q_sin_1), dim=-1)
    grad_query_states_pre_rope = grad_q_cos + grad_q_sin_rotated

    grad_k_cos = grad_key_states * cos_expanded
    grad_k_sin = grad_key_states * sin_expanded
    grad_k_sin_1 = grad_k_sin[..., : head_dim // 2]
    grad_k_sin_2 = grad_k_sin[..., head_dim // 2 :]
    grad_k_sin_rotated = torch.cat((grad_k_sin_2, -grad_k_sin_1), dim=-1)
    grad_key_states_pre_rope = grad_k_cos + grad_k_sin_rotated

    grad_value_states_pre_rope = grad_value_states

    grad_query_states_pre_rope = grad_query_states_pre_rope.transpose(1, 2).contiguous()
    grad_query_proj = grad_query_states_pre_rope.reshape(batch_size, seq_len, num_attention_heads * head_dim)

    grad_key_states_pre_rope = grad_key_states_pre_rope.transpose(1, 2).contiguous()
    grad_key_proj = grad_key_states_pre_rope.reshape(batch_size, seq_len, num_key_value_heads * head_dim)

    grad_value_states_pre_rope = grad_value_states_pre_rope.transpose(1, 2).contiguous()
    grad_value_proj = grad_value_states_pre_rope.reshape(batch_size, seq_len, num_key_value_heads * head_dim)

    grad_hidden_states_q = torch.matmul(grad_query_proj, q_weight)
    grad_q_weight = torch.matmul(
        grad_query_proj.reshape(-1, grad_query_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )

    grad_hidden_states_k = torch.matmul(grad_key_proj, k_weight)
    grad_k_weight = torch.matmul(
        grad_key_proj.reshape(-1, grad_key_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )

    grad_hidden_states_v = torch.matmul(grad_value_proj, v_weight)
    grad_v_weight = torch.matmul(
        grad_value_proj.reshape(-1, grad_value_proj.shape[-1]).t(),
        hidden_states.reshape(-1, hidden_states.shape[-1])
    )

    grad_hidden_states = grad_hidden_states_q + grad_hidden_states_k + grad_hidden_states_v

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_q_weight.to(torch.bfloat16),
        grad_k_weight.to(torch.bfloat16),
        grad_v_weight.to(torch.bfloat16),
        grad_o_weight.to(torch.bfloat16),
    )


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    scaling: float,
):
    return _compiled_run(
        grad_output, hidden_states, cos, sin, q_weight, k_weight, v_weight,
        o_weight, query_states, key_states, value_states, attn_weights,
        attn_output, scaling,
    )


_compiled_run = torch.compile(_run_impl, dynamic=True, mode="max-autotune-no-cudagraphs")
