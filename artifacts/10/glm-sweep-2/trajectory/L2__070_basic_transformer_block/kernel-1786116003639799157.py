import torch
import torch.nn.functional as F

_NUM_HEADS = 160
_HEAD_DIM = 24
_INNER_DIM = _NUM_HEADS * _HEAD_DIM
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
    num_attention_heads = 160
    attention_head_dim = 24
    inner_dim = num_attention_heads * attention_head_dim
    scale = attention_head_dim ** -0.5

    batch_size = hidden_states.shape[0]
    spatial_seq_len = hidden_states.shape[1]
    encoder_seq_len = encoder_hidden_states.shape[1]

    # ============ Self-Attention Block ============
    mean1 = hidden_states.mean(dim=-1, keepdim=True)
    var1 = ((hidden_states - mean1) ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = (hidden_states - mean1) / torch.sqrt(var1 + norm_eps)
    norm_hidden_states = norm_hidden_states * norm1_weight + norm1_bias

    # Fused QKV projection
    qkv_weight = torch.cat([attn1_to_q_weight, attn1_to_k_weight, attn1_to_v_weight], dim=0)
    qkv = F.linear(norm_hidden_states, qkv_weight)
    query, key, value = qkv.split(inner_dim, dim=-1)

    query = query.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    key = key.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    value = value.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_output = torch.matmul(attention_probs, value)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, inner_dim)
    attn_output = F.linear(attn_output, attn1_to_out_weight, attn1_to_out_bias)
    hidden_states = hidden_states + attn_output

    # ============ Cross-Attention Block ============
    mean2 = hidden_states.mean(dim=-1, keepdim=True)
    var2 = ((hidden_states - mean2) ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = (hidden_states - mean2) / torch.sqrt(var2 + norm_eps)
    norm_hidden_states = norm_hidden_states * norm2_weight + norm2_bias

    # Q from norm_hidden_states, KV from encoder_hidden_states
    q_proj = F.linear(norm_hidden_states, attn2_to_q_weight)
    kv_weight = torch.cat([attn2_to_k_weight, attn2_to_v_weight], dim=0)
    kv = F.linear(encoder_hidden_states, kv_weight)
    key, value = kv.split(inner_dim, dim=-1)

    query = q_proj.view(batch_size, spatial_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    key = key.view(batch_size, encoder_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)
    value = value.view(batch_size, encoder_seq_len, num_attention_heads, attention_head_dim).transpose(1, 2)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_output = torch.matmul(attention_probs, value)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, spatial_seq_len, inner_dim)
    attn_output = F.linear(attn_output, attn2_to_out_weight, attn2_to_out_bias)
    hidden_states = hidden_states + attn_output

    # ============ Feed-Forward Block ============
    mean3 = hidden_states.mean(dim=-1, keepdim=True)
    var3 = ((hidden_states - mean3) ** 2).mean(dim=-1, keepdim=True)
    norm_hidden_states = (hidden_states - mean3) / torch.sqrt(var3 + norm_eps)
    norm_hidden_states = norm_hidden_states * norm3_weight + norm3_bias

    ff_output = F.linear(norm_hidden_states, ff_linear1_weight, ff_linear1_bias)
    x, gate = ff_output.chunk(2, dim=-1)
    ff_output = x * F.gelu(gate, approximate='tanh')
    ff_output = F.linear(ff_output, ff_linear2_weight, ff_linear2_bias)
    output = hidden_states + ff_output

    return output
