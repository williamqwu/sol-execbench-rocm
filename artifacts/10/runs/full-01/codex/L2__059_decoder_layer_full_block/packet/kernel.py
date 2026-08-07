import torch
import torch.nn.functional as F
import aiter
from flash_attn import attention


@torch.compile(fullgraph=True, dynamic=True)
def _rope(q, k, position_ids, inv_freq):
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.float()
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16).unsqueeze(1)
    sin = emb.sin().to(torch.bfloat16).unsqueeze(1)
    q1, q2 = q[..., :64], q[..., 64:]
    k1, k2 = k[..., :64], k[..., 64:]
    q = q * cos + torch.cat((-q2, q1), dim=-1) * sin
    k = k * cos + torch.cat((-k2, k1), dim=-1) * sin
    return q, k


@torch.no_grad()
def run(
    hidden_states,
    position_ids,
    attention_mask,
    input_layernorm_weight,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    q_norm_weight,
    k_norm_weight,
    o_proj_weight,
    post_attention_layernorm_weight,
    gate_proj_weight,
    up_proj_weight,
    down_proj_weight,
    inv_freq,
    rms_norm_eps,
    attention_scale,
):
    batch_size, seq_len, hidden_size = hidden_states.shape

    def rms_norm(x, weight):
        return aiter.rms_norm(x, weight, rms_norm_eps)

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)

    q = F.linear(hidden_states, q_proj_weight).view(batch_size, seq_len, 40, 128)
    k = F.linear(hidden_states, k_proj_weight).view(batch_size, seq_len, 8, 128)
    v = F.linear(hidden_states, v_proj_weight).view(batch_size, seq_len, 8, 128)

    q = rms_norm(q, q_norm_weight).transpose(1, 2)
    k = rms_norm(k, k_norm_weight).transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = _rope(q, k, position_ids, inv_freq)

    attn_output = attention(q, k, v, attention_mask, attention_scale).reshape(batch_size, seq_len, 5120)
    attn_output = F.linear(attn_output, o_proj_weight)

    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)
    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    mlp_output = F.linear(gate * up, down_proj_weight)
    return residual + mlp_output
