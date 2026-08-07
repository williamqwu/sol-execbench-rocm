import torch
import torch.nn.functional as F

_NUM_HEADS = 160
_HEAD_DIM = 24
_INNER_DIM = _NUM_HEADS * _HEAD_DIM
_SCALE = _HEAD_DIM ** -0.5


@torch.compile(dynamic=True)
def _run_impl(
    hidden_states, encoder_hidden_states,
    norm1_weight, norm1_bias,
    attn1_to_q_weight, attn1_to_k_weight, attn1_to_v_weight,
    attn1_to_out_weight, attn1_to_out_bias,
    norm2_weight, norm2_bias,
    attn2_to_q_weight, attn2_to_k_weight, attn2_to_v_weight,
    attn2_to_out_weight, attn2_to_out_bias,
    norm3_weight, norm3_bias,
    ff_linear1_weight, ff_linear1_bias,
    ff_linear2_weight, ff_linear2_bias,
    norm_eps,
):
    batch_size = hidden_states.shape[0]
    spatial_seq_len = hidden_states.shape[1]
    encoder_seq_len = encoder_hidden_states.shape[1]

    # Self-Attention
    norm_hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],), norm1_weight, norm1_bias, norm_eps)
    query = F.linear(norm_hidden_states, attn1_to_q_weight)
    key = F.linear(norm_hidden_states, attn1_to_k_weight)
    value = F.linear(norm_hidden_states, attn1_to_v_weight)
    query = query.view(batch_size, spatial_seq_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
    key = key.view(batch_size, spatial_seq_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
    value = value.view(batch_size, spatial_seq_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * _SCALE
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_output = torch.matmul(attention_probs, value)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, _INNER_DIM)
    attn_output = F.linear(attn_output, attn1_to_out_weight, attn1_to_out_bias)
    hidden_states = hidden_states + attn_output

    # Cross-Attention
    norm_hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],), norm2_weight, norm2_bias, norm_eps)
    query = F.linear(norm_hidden_states, attn2_to_q_weight)
    key = F.linear(encoder_hidden_states, attn2_to_k_weight)
    value = F.linear(encoder_hidden_states, attn2_to_v_weight)
    query = query.view(batch_size, spatial_seq_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
    key = key.view(batch_size, encoder_seq_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
    value = value.view(batch_size, encoder_seq_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * _SCALE
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_output = torch.matmul(attention_probs, value)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, _INNER_DIM)
    attn_output = F.linear(attn_output, attn2_to_out_weight, attn2_to_out_bias)
    hidden_states = hidden_states + attn_output

    # Feed-Forward
    norm_hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],), norm3_weight, norm3_bias, norm_eps)
    ff_output = F.linear(norm_hidden_states, ff_linear1_weight, ff_linear1_bias)
    x, gate = ff_output.chunk(2, dim=-1)
    ff_output = x * F.gelu(gate, approximate='tanh')
    ff_output = F.linear(ff_output, ff_linear2_weight, ff_linear2_bias)
    output = hidden_states + ff_output
    return output


@torch.no_grad()
def run(
    hidden_states, encoder_hidden_states,
    norm1_weight, norm1_bias,
    attn1_to_q_weight, attn1_to_k_weight, attn1_to_v_weight,
    attn1_to_out_weight, attn1_to_out_bias,
    norm2_weight, norm2_bias,
    attn2_to_q_weight, attn2_to_k_weight, attn2_to_v_weight,
    attn2_to_out_weight, attn2_to_out_bias,
    norm3_weight, norm3_bias,
    ff_linear1_weight, ff_linear1_bias,
    ff_linear2_weight, ff_linear2_bias,
    norm_eps,
):
    return _run_impl(
        hidden_states, encoder_hidden_states,
        norm1_weight, norm1_bias,
        attn1_to_q_weight, attn1_to_k_weight, attn1_to_v_weight,
        attn1_to_out_weight, attn1_to_out_bias,
        norm2_weight, norm2_bias,
        attn2_to_q_weight, attn2_to_k_weight, attn2_to_v_weight,
        attn2_to_out_weight, attn2_to_out_bias,
        norm3_weight, norm3_bias,
        ff_linear1_weight, ff_linear1_bias,
        ff_linear2_weight, ff_linear2_bias,
        norm_eps,
    )
