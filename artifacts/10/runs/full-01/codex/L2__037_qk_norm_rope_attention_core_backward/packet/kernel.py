import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True, dynamic=True)
def _rope_rotated_mul_both(query, key, sin):
    query_rotated = torch.cat(
        (-query[..., 64:] * sin[..., :64], query[..., :64] * sin[..., 64:]),
        dim=-1,
    )
    key_rotated = torch.cat(
        (-key[..., 64:] * sin[..., :64], key[..., :64] * sin[..., 64:]),
        dim=-1,
    )
    return query_rotated, key_rotated


@torch.compile(fullgraph=True, dynamic=True)
def _rope_grad_rotated_mul_both(query, key, sin):
    query_rotated = torch.cat(
        (
            query[..., 64:] * sin[..., 64:],
            -(query[..., :64] * sin[..., :64]),
        ),
        dim=-1,
    )
    key_rotated = torch.cat(
        (key[..., 64:] * sin[..., 64:], -(key[..., :64] * sin[..., :64])),
        dim=-1,
    )
    return query_rotated, key_rotated


@torch.compile(fullgraph=True, dynamic=True)
def _cap_and_causal_mask(x, cap):
    n = x.shape[-1]
    indices = torch.arange(n, device=x.device)
    causal = indices[None, None, None, :] > indices[None, None, :, None]
    return (x * cap).masked_fill(causal, float("-inf"))


@torch.compile(fullgraph=True, dynamic=True)
def _attention_pointwise_backward(grad, probs, row_sum, tanh_scores, scaling):
    return (probs * (grad - row_sum)) * (1.0 - tanh_scores.pow(2)) * scaling


@torch.compile(fullgraph=True, dynamic=True)
def _rms_second_term_both(query, query_grad_variance, key, key_grad_variance):
    return (
        2.0 * query * query_grad_variance / 128,
        2.0 * key * key_grad_variance / 128,
    )


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    cos,
    sin,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    q_norm_weight,
    k_norm_weight,
    scaling,
    attn_logit_softcapping,
    rms_norm_eps,
):
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 3
    batch_size, seq_len, hidden_size = hidden_states.shape
    use_fused = batch_size * seq_len > 256

    query_states = F.linear(hidden_states, q_weight)
    key_states = F.linear(hidden_states, k_weight)
    value_states = F.linear(hidden_states, v_weight)
    query_states = query_states.view(batch_size, seq_len, 24, 128).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)

    q_variance = query_states.pow(2).mean(-1, keepdim=True)
    q_normed = query_states * torch.rsqrt(q_variance + rms_norm_eps)
    q_norm_scale = 1.0 + q_norm_weight
    query_states_normalized = q_normed * q_norm_scale
    k_variance = key_states.pow(2).mean(-1, keepdim=True)
    k_normed = key_states * torch.rsqrt(k_variance + rms_norm_eps)
    k_norm_scale = 1.0 + k_norm_weight
    key_states_normalized = k_normed * k_norm_scale

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    if use_fused:
        q_rotated_sin, k_rotated_sin = _rope_rotated_mul_both(
            query_states_normalized, key_states_normalized, sin_expanded
        )
    else:
        q_rotated_sin = torch.cat(
            (-query_states_normalized[..., 64:], query_states_normalized[..., :64]), dim=-1
        ) * sin_expanded
        k_rotated_sin = torch.cat(
            (-key_states_normalized[..., 64:], key_states_normalized[..., :64]), dim=-1
        ) * sin_expanded
    query_states_normalized.mul_(cos_expanded)
    query_states_normalized.add_(q_rotated_sin)
    query_states_rope = query_states_normalized
    key_states_normalized.mul_(cos_expanded)
    key_states_normalized.add_(k_rotated_sin)
    key_states_rope = key_states_normalized

    key_states_repeated = torch.repeat_interleave(key_states_rope, 3, dim=1)
    value_states_repeated = torch.repeat_interleave(value_states, 3, dim=1)
    attn_weights = torch.matmul(query_states_rope, key_states_repeated.transpose(2, 3))
    attn_weights.mul_(scaling)
    attn_weights.div_(attn_logit_softcapping)
    attn_weights.tanh_()
    attn_weights_tanh = attn_weights
    if use_fused:
        attn_weights_capped = _cap_and_causal_mask(
            attn_weights_tanh, attn_logit_softcapping
        )
    else:
        attn_weights_capped = attn_weights_tanh * attn_logit_softcapping
        causal_mask = torch.ones(
            seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
        ).triu_(diagonal=1)
        attn_weights_capped.masked_fill_(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    attn_weights_softmax = F.softmax(attn_weights_capped, dim=-1, dtype=torch.float32)
    if batch_size == 1:
        attn_output_flat = torch.empty(
            batch_size,
            seq_len,
            num_attention_heads,
            head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        torch.matmul(
            attn_weights_softmax,
            value_states_repeated,
            out=attn_output_flat.transpose(1, 2),
        )
        attn_output_flat = attn_output_flat.view(batch_size, seq_len, 3072)
    else:
        attn_output = torch.matmul(attn_weights_softmax, value_states_repeated)
        attn_output_flat = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, 3072
        )

    grad_attn_output_flat = F.linear(grad_output, o_weight.t())
    grad_o_weight = grad_output.view(-1, hidden_size).t() @ attn_output_flat.view(-1, 3072)
    grad_attn_output = grad_attn_output_flat.view(batch_size, seq_len, 24, 128).transpose(1, 2)
    grad_attn_weights_dropped = torch.matmul(grad_attn_output, value_states_repeated.transpose(2, 3))
    grad_value_states_repeated = torch.matmul(attn_weights_softmax.transpose(2, 3), grad_attn_output)
    torch.mul(
        grad_attn_weights_dropped,
        attn_weights_softmax,
        out=attn_weights_capped,
    )
    sum_grad = attn_weights_capped.sum(dim=-1, keepdim=True)
    if use_fused:
        grad_attn_weights_scaled = _attention_pointwise_backward(
            grad_attn_weights_dropped,
            attn_weights_softmax,
            sum_grad,
            attn_weights_tanh,
            scaling,
        )
    else:
        grad_attn_weights_dropped.sub_(sum_grad)
        grad_attn_weights_dropped.mul_(attn_weights_softmax)
        attn_weights_tanh.pow_(2)
        torch.sub(1.0, attn_weights_tanh, out=attn_weights_tanh)
        grad_attn_weights_dropped.mul_(attn_weights_tanh)
        grad_attn_weights_dropped.mul_(scaling)
        grad_attn_weights_scaled = grad_attn_weights_dropped
    grad_query_states_rope = torch.matmul(grad_attn_weights_scaled, key_states_repeated)
    grad_key_states_repeated = torch.matmul(grad_attn_weights_scaled.transpose(2, 3), query_states_rope)
    grad_key_states_rope = grad_key_states_repeated.view(batch_size, 8, 3, seq_len, 128).sum(dim=2)
    grad_value_states = torch.empty(
        batch_size,
        seq_len,
        num_key_value_heads,
        head_dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    torch.sum(
        grad_value_states_repeated.view(batch_size, 8, 3, seq_len, 128),
        dim=2,
        out=grad_value_states.transpose(1, 2),
    )

    if use_fused:
        grad_q_rotated_sin, grad_k_rotated_sin = _rope_grad_rotated_mul_both(
            grad_query_states_rope, grad_key_states_rope, sin_expanded
        )
    else:
        grad_q_rotated = grad_query_states_rope * sin_expanded
        grad_q_rotated_sin = torch.cat(
            (grad_q_rotated[..., 64:], -grad_q_rotated[..., :64]), dim=-1
        )
        grad_k_rotated = grad_key_states_rope * sin_expanded
        grad_k_rotated_sin = torch.cat(
            (grad_k_rotated[..., 64:], -grad_k_rotated[..., :64]), dim=-1
        )
    grad_query_states_rope.mul_(cos_expanded)
    grad_query_states_rope.add_(grad_q_rotated_sin)
    grad_query_states_normalized = grad_query_states_rope
    grad_key_states_rope.mul_(cos_expanded)
    grad_key_states_rope.add_(grad_k_rotated_sin)
    grad_key_states_normalized = grad_key_states_rope

    grad_q_norm_weight = (grad_query_states_normalized * q_normed).sum(dim=(0, 1, 2))
    grad_query_states_normalized.mul_(q_norm_scale)
    grad_q_normed = grad_query_states_normalized
    rsqrt_q_var = torch.rsqrt(q_variance + rms_norm_eps)
    grad_q_dot = (grad_q_normed * query_states).sum(dim=-1, keepdim=True)
    grad_q_normed.mul_(rsqrt_q_var)
    grad_query_first = grad_q_normed
    rsqrt_q_var.pow_(3)
    grad_q_var = -0.5 * grad_q_dot * rsqrt_q_var
    grad_k_norm_weight = (grad_key_states_normalized * k_normed).sum(dim=(0, 1, 2))
    grad_key_states_normalized.mul_(k_norm_scale)
    grad_k_normed = grad_key_states_normalized
    rsqrt_k_var = torch.rsqrt(k_variance + rms_norm_eps)
    grad_k_dot = (grad_k_normed * key_states).sum(dim=-1, keepdim=True)
    grad_k_normed.mul_(rsqrt_k_var)
    grad_key_first = grad_k_normed
    rsqrt_k_var.pow_(3)
    grad_k_var = -0.5 * grad_k_dot * rsqrt_k_var
    if use_fused:
        grad_query_second, grad_key_second = _rms_second_term_both(
            query_states, grad_q_var, key_states, grad_k_var
        )
    else:
        grad_query_second = 2.0 * query_states * grad_q_var / head_dim
        grad_key_second = 2.0 * key_states * grad_k_var / head_dim
    grad_query_states = torch.empty(
        batch_size,
        seq_len,
        num_attention_heads,
        head_dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    torch.add(
        grad_query_first,
        grad_query_second,
        out=grad_query_states.transpose(1, 2),
    )
    grad_key_states = torch.empty(
        batch_size,
        seq_len,
        num_key_value_heads,
        head_dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    torch.add(
        grad_key_first,
        grad_key_second,
        out=grad_key_states.transpose(1, 2),
    )

    grad_query_states = grad_query_states.view(batch_size, seq_len, 3072)
    grad_key_states = grad_key_states.view(batch_size, seq_len, 1024)
    grad_value_states = grad_value_states.view(batch_size, seq_len, 1024)
    grad_hidden_states_q = F.linear(grad_query_states, q_weight.t())
    grad_q_weight = grad_query_states.view(-1, 3072).t() @ hidden_states.view(-1, hidden_size)
    grad_k_weight = grad_key_states.view(-1, 1024).t() @ hidden_states.view(-1, hidden_size)
    grad_v_weight = grad_value_states.view(-1, 1024).t() @ hidden_states.view(-1, hidden_size)
    grad_hidden_states_q.view(-1, hidden_size).addmm_(
        grad_key_states.view(-1, 1024), k_weight
    )
    grad_hidden_states_q.view(-1, hidden_size).addmm_(
        grad_value_states.view(-1, 1024), v_weight
    )
    grad_hidden_states = grad_hidden_states_q
    return (
        grad_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_o_weight,
        grad_q_norm_weight,
        grad_k_norm_weight,
    )
