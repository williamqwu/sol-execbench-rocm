import torch

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def _run_compiled(
    grad_query, grad_key, grad_value, hidden_states,
    q_weight, k_weight, v_weight, cos, sin,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 24
    num_kv_heads = 4
    head_dim = 96
    half_head_dim = head_dim // 2

    cos_unsqueezed = cos.unsqueeze(1)
    sin_unsqueezed = sin.unsqueeze(1)

    grad_q_cos_term = grad_query * cos_unsqueezed
    grad_q_sin_term = grad_query * sin_unsqueezed

    grad_q_sin_1 = grad_q_sin_term[..., :half_head_dim]
    grad_q_sin_2 = grad_q_sin_term[..., half_head_dim:]
    grad_q_sin_reversed = torch.cat((grad_q_sin_2, -grad_q_sin_1), dim=-1)

    grad_query_unrotated = grad_q_cos_term + grad_q_sin_reversed

    grad_k_cos_term = grad_key * cos_unsqueezed
    grad_k_sin_term = grad_key * sin_unsqueezed

    grad_k_sin_1 = grad_k_sin_term[..., :half_head_dim]
    grad_k_sin_2 = grad_k_sin_term[..., half_head_dim:]
    grad_k_sin_reversed = torch.cat((grad_k_sin_2, -grad_k_sin_1), dim=-1)

    grad_key_unrotated = grad_k_cos_term + grad_k_sin_reversed

    grad_query_transposed = grad_query_unrotated.transpose(1, 2)
    grad_query_proj = grad_query_transposed.contiguous().view(
        batch_size, seq_len, num_heads * head_dim
    )

    grad_key_transposed = grad_key_unrotated.transpose(1, 2)
    grad_key_proj = grad_key_transposed.contiguous().view(
        batch_size, seq_len, num_kv_heads * head_dim
    )

    grad_value_transposed = grad_value.transpose(1, 2)
    grad_value_proj = grad_value_transposed.contiguous().view(
        batch_size, seq_len, num_kv_heads * head_dim
    )

    grad_hidden_from_q = torch.matmul(grad_query_proj, q_weight)
    grad_hidden_from_k = torch.matmul(grad_key_proj, k_weight)
    grad_hidden_from_v = torch.matmul(grad_value_proj, v_weight)
    grad_hidden_states = grad_hidden_from_q + grad_hidden_from_k + grad_hidden_from_v

    grad_query_proj_2d = grad_query_proj.view(-1, num_heads * head_dim)
    hidden_states_2d = hidden_states.view(-1, hidden_size)
    grad_q_weight = torch.matmul(grad_query_proj_2d.t(), hidden_states_2d)

    grad_key_proj_2d = grad_key_proj.view(-1, num_kv_heads * head_dim)
    grad_k_weight = torch.matmul(grad_key_proj_2d.t(), hidden_states_2d)

    grad_value_proj_2d = grad_value_proj.view(-1, num_kv_heads * head_dim)
    grad_v_weight = torch.matmul(grad_value_proj_2d.t(), hidden_states_2d)

    return grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight


@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    return _run_compiled(
        grad_query, grad_key, grad_value, hidden_states,
        q_weight, k_weight, v_weight, cos, sin,
    )
