import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    output = x_normed * (1.0 + weight.float())
    return output.type_as(x)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.no_grad()
def _run_impl(
    hidden_states, cos, sin, attention_mask,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    q_norm_weight, k_norm_weight,
    attn_logit_softcapping, rms_norm_eps,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5

    query_states = F.linear(hidden_states, q_proj_weight)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)

    key_states = F.linear(hidden_states, k_proj_weight)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    value_states = F.linear(hidden_states, v_proj_weight)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    attn_weights = attn_weights / attn_logit_softcapping
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * attn_logit_softcapping

    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)

    attn_output = F.linear(attn_output, o_proj_weight)
    return attn_output


_compiled = torch.compile(_run_impl, mode="default", dynamic=True)


@torch.no_grad()
def run(
    hidden_states, cos, sin, attention_mask,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    q_norm_weight, k_norm_weight,
    attn_logit_softcapping, rms_norm_eps,
):
    return _compiled(
        hidden_states, cos, sin, attention_mask,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        q_norm_weight, k_norm_weight,
        attn_logit_softcapping, rms_norm_eps,
    )
