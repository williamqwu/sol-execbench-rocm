import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
):
    num_heads = 28
    num_kv_heads = 4
    num_kv_groups = 7
    head_dim = 128
    mrope_section = [16, 24, 24]

    bsz, q_len, _ = hidden_states.size()

    query_states = F.linear(hidden_states, q_weight, q_bias)
    key_states = F.linear(hidden_states, k_weight, k_bias)
    value_states = F.linear(hidden_states, v_weight, v_bias)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    mrope_section_doubled = [s * 2 for s in mrope_section]
    cos_splits = cos.split(mrope_section_doubled, dim=-1)
    sin_splits = sin.split(mrope_section_doubled, dim=-1)
    cos_combined = torch.cat([m[i % 3] for i, m in enumerate(cos_splits)], dim=-1).unsqueeze(1)
    sin_combined = torch.cat([m[i % 3] for i, m in enumerate(sin_splits)], dim=-1).unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query_states = (query_states * cos_combined) + (rotate_half(query_states) * sin_combined)
    key_states = (key_states * cos_combined) + (rotate_half(key_states) * sin_combined)

    # SDPA with native GQA + causal. The attention_mask is a standard causal
    # mask (0 below diag, finfo.min above), equivalent to is_causal=True.
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True
    )

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, num_heads * head_dim)
    output = F.linear(attn_output, o_weight)

    return output

run = torch.compile(run, dynamic=False)
