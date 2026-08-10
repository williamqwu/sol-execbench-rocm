import torch
import math

def _rope(query_states, key_states, inv_freq, position_ids):
    batch_size = query_states.shape[0]
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    q_rotated = torch.cat((-query_states[..., 32:], query_states[..., :32]), dim=-1)
    k_rotated = torch.cat((-key_states[..., 32:], key_states[..., :32]), dim=-1)
    return query_states * cos + q_rotated * sin, key_states * cos + k_rotated * sin

@torch.no_grad()
def run(hidden_states, position_ids, q_proj_weight, k_proj_weight, v_proj_weight,
        o_proj_weight, inv_freq, is_causal):
    batch_size, seq_len, hidden_size = hidden_states.shape
    query_states = torch.matmul(hidden_states, q_proj_weight.t()).view(batch_size, seq_len, 16, 64).transpose(1, 2)
    kv_states = torch.matmul(hidden_states, torch.cat((k_proj_weight, v_proj_weight)).t())
    key_states, value_states = kv_states.split(256, dim=-1)
    key_states = key_states.view(batch_size, seq_len, 4, 64).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, 4, 64).transpose(1, 2)
    query_states, key_states = _rope(query_states, key_states, inv_freq, position_ids)
    key_states = key_states[:, :, None].expand(batch_size, 4, 4, seq_len, 64).reshape(batch_size, 16, seq_len, 64)
    value_states = value_states[:, :, None].expand(batch_size, 4, 4, seq_len, 64).reshape(batch_size, 16, seq_len, 64)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask[None, None], float('-inf'))
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_output = torch.matmul(attn_output, o_proj_weight.t())
    return attn_output, attn_weights
