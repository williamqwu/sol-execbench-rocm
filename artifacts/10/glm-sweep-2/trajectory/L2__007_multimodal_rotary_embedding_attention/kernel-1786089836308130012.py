import torch
import torch.nn.functional as F

NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
MROPE_DOUBLED = [32, 48, 48]


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
    bsz, q_len, _ = hidden_states.size()

    query_states = F.linear(hidden_states, q_weight, q_bias)
    key_states = F.linear(hidden_states, k_weight, k_bias)
    value_states = F.linear(hidden_states, v_weight, v_bias)

    query_states = query_states.view(bsz, q_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    cos_splits = cos.split(MROPE_DOUBLED, dim=-1)
    sin_splits = sin.split(MROPE_DOUBLED, dim=-1)
    cos_combined = torch.cat(
        [cos_splits[0][0], cos_splits[1][1], cos_splits[2][2]], dim=-1
    ).unsqueeze(1)
    sin_combined = torch.cat(
        [sin_splits[0][0], sin_splits[1][1], sin_splits[2][2]], dim=-1
    ).unsqueeze(1)

    half = HEAD_DIM // 2
    q0 = query_states[..., :half]
    q1 = query_states[..., half:]
    k0 = key_states[..., :half]
    k1 = key_states[..., half:]
    cc0 = cos_combined[..., :half]
    cc1 = cos_combined[..., half:]
    sc0 = sin_combined[..., :half]
    sc1 = sin_combined[..., half:]
    query_states = torch.cat(
        (q0 * cc0 - q1 * sc0, q1 * cc1 + q0 * sc1), dim=-1
    )
    key_states = torch.cat(
        (k0 * cc0 - k1 * sc0, k1 * cc1 + k0 * sc1), dim=-1
    )

    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True
    )

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, NUM_HEADS * HEAD_DIM)
    output = F.linear(attn_output, o_weight)

    return output
