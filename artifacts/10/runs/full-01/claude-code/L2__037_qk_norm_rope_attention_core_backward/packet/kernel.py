import torch
import torch.nn.functional as F


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
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    scaling: float,
    attn_logit_softcapping: float,
    rms_norm_eps: float,
):
    # Constants
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads

    batch_size, seq_len, hidden_size = hidden_states.shape

    # === FORWARD RECOMPUTATION ===
    query_states = F.linear(hidden_states, q_weight)
    key_states = F.linear(hidden_states, k_weight)
    value_states = F.linear(hidden_states, v_weight)

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    q_variance = query_states.pow(2).mean(-1, keepdim=True)
    q_normed = query_states * torch.rsqrt(q_variance + rms_norm_eps)
    query_states_normalized = q_normed * (1.0 + q_norm_weight)

    k_variance = key_states.pow(2).mean(-1, keepdim=True)
    k_normed = key_states * torch.rsqrt(k_variance + rms_norm_eps)
    key_states_normalized = k_normed * (1.0 + k_norm_weight)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    q_half_dim = head_dim // 2
    q1 = query_states_normalized[..., :q_half_dim]
    q2 = query_states_normalized[..., q_half_dim:]
    q_rotated_half = torch.cat((-q2, q1), dim=-1)
    query_states_rope = (query_states_normalized * cos_expanded) + (q_rotated_half * sin_expanded)

    k1 = key_states_normalized[..., :q_half_dim]
    k2 = key_states_normalized[..., q_half_dim:]
    k_rotated_half = torch.cat((-k2, k1), dim=-1)
    key_states_rope = (key_states_normalized * cos_expanded) + (k_rotated_half * sin_expanded)

    key_states_repeated = key_states_rope[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states_repeated = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    attn_weights = torch.matmul(query_states_rope, key_states_repeated.transpose(2, 3)) * scaling

    attn_weights_for_tanh = attn_weights / attn_logit_softcapping
    attn_weights_tanh = torch.tanh(attn_weights_for_tanh)
    attn_weights_capped = attn_weights_tanh * attn_logit_softcapping

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
    attn_weights_capped = attn_weights_capped.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights_softmax = F.softmax(attn_weights_capped, dim=-1, dtype=torch.float32)

    attn_output = torch.matmul(attn_weights_softmax, value_states_repeated)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output_flat = attn_output.view(batch_size, seq_len, num_attention_heads * head_dim)

    # === BACKWARD PASS ===
    grad_attn_output_flat = F.linear(grad_output, o_weight.t())
    grad_o_weight = grad_output.view(-1, hidden_size).t() @ attn_output_flat.view(-1, num_attention_heads * head_dim)

    grad_attn_output = grad_attn_output_flat.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)

    grad_attn_weights_dropped = torch.matmul(grad_attn_output, value_states_repeated.transpose(2, 3))
    grad_value_states_repeated = torch.matmul(attn_weights_softmax.transpose(2, 3), grad_attn_output)

    sum_grad = (grad_attn_weights_dropped * attn_weights_softmax).sum(dim=-1, keepdim=True)
    grad_attn_weights = attn_weights_softmax * (grad_attn_weights_dropped - sum_grad)

    grad_attn_weights_uncapped = grad_attn_weights * (1.0 - attn_weights_tanh.pow(2))

    grad_attn_weights_scaled = grad_attn_weights_uncapped * scaling
    grad_query_states_rope = torch.matmul(grad_attn_weights_scaled, key_states_repeated)
    grad_key_states_repeated = torch.matmul(grad_attn_weights_scaled.transpose(2, 3), query_states_rope)

    grad_key_states_rope = grad_key_states_repeated.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)
    grad_value_states = grad_value_states_repeated.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).sum(dim=2)

    grad_q_from_cos = grad_query_states_rope * cos_expanded
    grad_q_rotated = grad_query_states_rope * sin_expanded
    grad_q_rot_1 = grad_q_rotated[..., :q_half_dim]
    grad_q_rot_2 = grad_q_rotated[..., q_half_dim:]
    grad_q_from_sin = torch.cat((grad_q_rot_2, -grad_q_rot_1), dim=-1)
    grad_query_states_normalized = grad_q_from_cos + grad_q_from_sin

    grad_k_from_cos = grad_key_states_rope * cos_expanded
    grad_k_rotated = grad_key_states_rope * sin_expanded
    grad_k_rot_1 = grad_k_rotated[..., :q_half_dim]
    grad_k_rot_2 = grad_k_rotated[..., q_half_dim:]
    grad_k_from_sin = torch.cat((grad_k_rot_2, -grad_k_rot_1), dim=-1)
    grad_key_states_normalized = grad_k_from_cos + grad_k_from_sin

    grad_q_normed = grad_query_states_normalized * (1.0 + q_norm_weight)
    grad_q_norm_weight = (grad_query_states_normalized * q_normed).sum(dim=(0, 1, 2))

    rsqrt_q_var = torch.rsqrt(q_variance + rms_norm_eps)
    grad_q_var = -0.5 * (grad_q_normed * query_states).sum(dim=-1, keepdim=True) * rsqrt_q_var.pow(3)
    grad_query_states = grad_q_normed * rsqrt_q_var + 2.0 * query_states * grad_q_var / head_dim

    grad_k_normed = grad_key_states_normalized * (1.0 + k_norm_weight)
    grad_k_norm_weight = (grad_key_states_normalized * k_normed).sum(dim=(0, 1, 2))

    rsqrt_k_var = torch.rsqrt(k_variance + rms_norm_eps)
    grad_k_var = -0.5 * (grad_k_normed * key_states).sum(dim=-1, keepdim=True) * rsqrt_k_var.pow(3)
    grad_key_states = grad_k_normed * rsqrt_k_var + 2.0 * key_states * grad_k_var / head_dim

    grad_query_states = grad_query_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    grad_key_states = grad_key_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    grad_value_states = grad_value_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    grad_hidden_states_q = F.linear(grad_query_states, q_weight.t())
    grad_q_weight = grad_query_states.view(-1, num_attention_heads * head_dim).t() @ hidden_states.view(-1, hidden_size)

    grad_hidden_states_k = F.linear(grad_key_states, k_weight.t())
    grad_k_weight = grad_key_states.view(-1, num_key_value_heads * head_dim).t() @ hidden_states.view(-1, hidden_size)

    grad_hidden_states_v = F.linear(grad_value_states, v_weight.t())
    grad_v_weight = grad_value_states.view(-1, num_key_value_heads * head_dim).t() @ hidden_states.view(-1, hidden_size)

    grad_hidden_states = grad_hidden_states_q + grad_hidden_states_k + grad_hidden_states_v

    return (
        grad_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_o_weight,
        grad_q_norm_weight,
        grad_k_norm_weight,
    )
