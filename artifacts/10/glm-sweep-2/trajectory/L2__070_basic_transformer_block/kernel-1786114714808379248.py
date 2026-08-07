import torch
import torch.nn.functional as F

_NUM_HEADS = 160
_HEAD_DIM = 24
_INNER_DIM = _NUM_HEADS * _HEAD_DIM  # 3840
_SCALE = _HEAD_DIM ** -0.5


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    attn1_to_q_weight: torch.Tensor,
    attn1_to_k_weight: torch.Tensor,
    attn1_to_v_weight: torch.Tensor,
    attn1_to_out_weight: torch.Tensor,
    attn1_to_out_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    attn2_to_q_weight: torch.Tensor,
    attn2_to_k_weight: torch.Tensor,
    attn2_to_v_weight: torch.Tensor,
    attn2_to_out_weight: torch.Tensor,
    attn2_to_out_bias: torch.Tensor,
    norm3_weight: torch.Tensor,
    norm3_bias: torch.Tensor,
    ff_linear1_weight: torch.Tensor,
    ff_linear1_bias: torch.Tensor,
    ff_linear2_weight: torch.Tensor,
    ff_linear2_bias: torch.Tensor,
    norm_eps: float,
):
    batch_size = hidden_states.shape[0]
    spatial_seq_len = hidden_states.shape[1]
    encoder_seq_len = encoder_hidden_states.shape[1]

    def attn(norm_hs, kv_hs, kv_len, wq, wk, wv, w_out, b_out, kv_normed=False):
        q = F.linear(norm_hs, wq)
        k = F.linear(kv_hs, wk)
        v = F.linear(kv_hs, wv)
        q = q.view(batch_size, -1, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
        k = k.view(batch_size, kv_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
        v = v.view(batch_size, kv_len, _NUM_HEADS, _HEAD_DIM).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=_SCALE)
        out = out.transpose(1, 2).reshape(batch_size, -1, _INNER_DIM)
        return F.linear(out, w_out, b_out)

    # Self-attention
    norm_hs = F.layer_norm(hidden_states, (hidden_states.shape[-1],), norm1_weight, norm1_bias, norm_eps)
    hidden_states = hidden_states + attn(
        norm_hs, norm_hs, spatial_seq_len,
        attn1_to_q_weight, attn1_to_k_weight, attn1_to_v_weight,
        attn1_to_out_weight, attn1_to_out_bias)

    # Cross-attention
    norm_hs = F.layer_norm(hidden_states, (hidden_states.shape[-1],), norm2_weight, norm2_bias, norm_eps)
    hidden_states = hidden_states + attn(
        norm_hs, encoder_hidden_states, encoder_seq_len,
        attn2_to_q_weight, attn2_to_k_weight, attn2_to_v_weight,
        attn2_to_out_weight, attn2_to_out_bias)

    # Feed-forward
    norm_hs = F.layer_norm(hidden_states, (hidden_states.shape[-1],), norm3_weight, norm3_bias, norm_eps)
    ff_output = F.linear(norm_hs, ff_linear1_weight, ff_linear1_bias)
    x, gate = ff_output.chunk(2, dim=-1)
    ff_output = x * F.gelu(gate, approximate='tanh')
    ff_output = F.linear(ff_output, ff_linear2_weight, ff_linear2_bias)
    output = hidden_states + ff_output

    return output
