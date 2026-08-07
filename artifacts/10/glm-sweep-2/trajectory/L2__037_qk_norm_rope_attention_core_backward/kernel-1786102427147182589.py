import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    num_attention_heads = axes_and_scalars["num_attention_heads"]
    num_key_value_heads = axes_and_scalars["num_key_value_heads"]
    head_dim = axes_and_scalars["head_dim"]
    scaling = 0.08838834764831845
    attn_logit_softcapping = 50.0
    rms_norm_eps = 1e-6
    q_proj_out = num_attention_heads * head_dim
    kv_proj_out = num_key_value_heads * head_dim
    with torch.no_grad():
        q_weight = torch.randn(q_proj_out, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
        k_weight = torch.randn(kv_proj_out, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
        v_weight = torch.randn(kv_proj_out, hidden_size, device=device) * (2.0 / hidden_size) ** 0.5
        o_weight = torch.randn(hidden_size, q_proj_out, device=device) * (2.0 / q_proj_out) ** 0.5
        q_norm_weight = torch.randn(head_dim, device=device) * 0.02
        k_norm_weight = torch.randn(head_dim, device=device) * 0.02
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device) * (1.0 / hidden_size ** 0.5)
        rope_theta = 10000.0
        half_dim = head_dim // 2
        freq_seq = torch.arange(0, half_dim, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (freq_seq / half_dim))
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(position_ids, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_vals = emb.cos()
        sin_vals = emb.sin()
        cos = cos_vals.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        sin = sin_vals.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        grad_output = torch.randn(batch_size, seq_len, hidden_size, device=device)
    return {
        "grad_output": grad_output, "hidden_states": hidden_states, "cos": cos, "sin": sin,
        "q_weight": q_weight, "k_weight": k_weight, "v_weight": v_weight, "o_weight": o_weight,
        "q_norm_weight": q_norm_weight, "k_norm_weight": k_norm_weight,
        "scaling": scaling, "attn_logit_softcapping": attn_logit_softcapping, "rms_norm_eps": rms_norm_eps,
    }


@torch.no_grad()
def run(
    grad_output, hidden_states, cos, sin,
    q_weight, k_weight, v_weight, o_weight,
    q_norm_weight, k_norm_weight,
    scaling, attn_logit_softcapping, rms_norm_eps,
):
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    batch_size, seq_len, hidden_size = hidden_states.shape
    q_half_dim = head_dim // 2
    inv_softcap = 1.0 / attn_logit_softcapping

    # === FORWARD ===
    query_states = F.linear(hidden_states, q_weight)
    key_states = F.linear(hidden_states, k_weight)
    value_states = F.linear(hidden_states, v_weight)

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # RMSNorm Q - reuse rsqrt
    q_variance = query_states.pow(2).mean(-1, keepdim=True)
    q_rsqrt = torch.rsqrt(q_variance + rms_norm_eps)
    q_normed = query_states * q_rsqrt
    query_states_normalized = q_normed * (1.0 + q_norm_weight)
    # RMSNorm K
    k_variance = key_states.pow(2).mean(-1, keepdim=True)
    k_rsqrt = torch.rsqrt(k_variance + rms_norm_eps)
    k_normed = key_states * k_rsqrt
    key_states_normalized = k_normed * (1.0 + k_norm_weight)

    # RoPE without cat: rotate via slicing
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    cos_h = cos_expanded[..., :q_half_dim]
    sin_h = sin_expanded[..., :q_half_dim]
    # rotate(x): for x=[a,b], rot=[-b,a]. So rope_out = x*cos + rot(x)*sin
    q1 = query_states_normalized[..., :q_half_dim]
    q2 = query_states_normalized[..., q_half_dim:]
    query_states_rope = torch.empty_like(query_states_normalized)
    query_states_rope[..., :q_half_dim] = q1 * cos_h - q2 * sin_h
    query_states_rope[..., q_half_dim:] = q2 * cos_h + q1 * sin_h
    k1 = key_states_normalized[..., :q_half_dim]
    k2 = key_states_normalized[..., q_half_dim:]
    key_states_rope = torch.empty_like(key_states_normalized)
    key_states_rope[..., :q_half_dim] = k1 * cos_h - k2 * sin_h
    key_states_rope[..., q_half_dim:] = k2 * cos_h + k1 * sin_h

    key_states_repeated = key_states_rope[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states_repeated = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    attn_weights = torch.matmul(query_states_rope, key_states_repeated.transpose(2, 3)) * scaling
    attn_weights_for_tanh = attn_weights * inv_softcap
    attn_weights_tanh = torch.tanh(attn_weights_for_tanh)
    attn_weights_capped = attn_weights_tanh * attn_logit_softcapping
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
    attn_weights_capped = attn_weights_capped.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn_weights_softmax = F.softmax(attn_weights_capped, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights_softmax, value_states_repeated)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output_flat = attn_output.view(batch_size, seq_len, num_attention_heads * head_dim)

    # === BACKWARD ===
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
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim).sum(dim=2)
    grad_value_states = grad_value_states_repeated.view(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim).sum(dim=2)

    # RoPE grad without cat
    gq1 = grad_query_states_rope[..., :q_half_dim]
    gq2 = grad_query_states_rope[..., q_half_dim:]
    grad_query_states_normalized = torch.empty_like(grad_query_states_rope)
    # d/dx of rope: grad_cos part + grad_sin part
    # forward: out1 = x1*cos - x2*sin; out2 = x2*cos + x1*sin
    # grad: x1' = g_out1*cos + g_out2*sin; x2' = g_out2*cos - g_out1*sin
    grad_query_states_normalized[..., :q_half_dim] = gq1 * cos_h + gq2 * sin_h
    grad_query_states_normalized[..., q_half_dim:] = gq2 * cos_h - gq1 * sin_h
    gk1 = grad_key_states_rope[..., :q_half_dim]
    gk2 = grad_key_states_rope[..., q_half_dim:]
    grad_key_states_normalized = torch.empty_like(grad_key_states_rope)
    grad_key_states_normalized[..., :q_half_dim] = gk1 * cos_h + gk2 * sin_h
    grad_key_states_normalized[..., q_half_dim:] = gk2 * cos_h - gk1 * sin_h

    # RMSNorm grad Q
    grad_q_normed = grad_query_states_normalized * (1.0 + q_norm_weight)
    grad_q_norm_weight = (grad_query_states_normalized * q_normed).sum(dim=(0, 1, 2))
    grad_q_var = -0.5 * (grad_q_normed * query_states).sum(dim=-1, keepdim=True) * q_rsqrt.pow(3)
    grad_query_states = grad_q_normed * q_rsqrt + 2.0 * query_states * grad_q_var / head_dim
    # RMSNorm grad K
    grad_k_normed = grad_key_states_normalized * (1.0 + k_norm_weight)
    grad_k_norm_weight = (grad_key_states_normalized * k_normed).sum(dim=(0, 1, 2))
    grad_k_var = -0.5 * (grad_k_normed * key_states).sum(dim=-1, keepdim=True) * k_rsqrt.pow(3)
    grad_key_states = grad_k_normed * k_rsqrt + 2.0 * key_states * grad_k_var / head_dim

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
        grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight, grad_o_weight,
        grad_q_norm_weight, grad_k_norm_weight,
    )
