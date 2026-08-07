import torch
import torch.nn.functional as F
from typing import Tuple


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def _run_batched(
    hidden_states, cu_seqlens, cos, sin,
    qkv_weight, qkv_bias, proj_weight, proj_bias,
    embed_dim, num_heads, head_dim, scaling,
):
    """Batched attention over padded sequences (wins when many sequences)."""
    seq_length = hidden_states.shape[0]
    device = hidden_states.device

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)

    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    zero = torch.zeros(1, dtype=torch.int64, device=device)
    cu_seqlens_with_zero = torch.cat([zero, cu_seqlens])
    lengths_all = cu_seqlens_with_zero[1:] - cu_seqlens_with_zero[:-1]
    valid = lengths_all > 0
    lengths = lengths_all[valid]
    num_seq = lengths.shape[0]
    if num_seq == 0:
        return torch.zeros(seq_length, embed_dim, device=device, dtype=hidden_states.dtype)

    max_len = int(lengths.max().item())
    offsets = torch.cat([zero, lengths.cumsum(0)])

    arange_max = torch.arange(max_len, device=device)
    idx = offsets[:-1, None] + arange_max[None, :]
    pad_key = arange_max[None, :] < lengths[:, None]
    idx_safe = torch.where(pad_key, idx, torch.zeros_like(idx))

    q_pad = query_states[idx_safe].permute(0, 2, 1, 3)
    k_pad = key_states[idx_safe].permute(0, 2, 1, 3)
    v_pad = value_states[idx_safe].permute(0, 2, 1, 3)

    scores = torch.matmul(q_pad, k_pad.transpose(-1, -2)) * scaling
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~pad_key[:, None, None, :], neg_inf)
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_pad.dtype)
    attn_out = torch.matmul(attn, v_pad)

    seq_id = torch.repeat_interleave(torch.arange(num_seq, device=device), lengths)
    pos_id = torch.arange(seq_length, device=device) - offsets[:-1][seq_id]
    attn_out = attn_out.permute(0, 2, 1, 3)[seq_id, pos_id]

    attn_out = attn_out.reshape(seq_length, embed_dim)
    attn_out = F.linear(attn_out, proj_weight, proj_bias)
    return attn_out


def _run_per_sequence(
    hidden_states, cu_seqlens, cos, sin,
    qkv_weight, qkv_bias, proj_weight, proj_bias,
    embed_dim, num_heads, head_dim, scaling,
):
    """Per-sequence loop (wins when few sequences: low launch overhead)."""
    seq_length = hidden_states.shape[0]
    device = hidden_states.device

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)

    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    zero = torch.zeros(1, dtype=torch.int64, device=device)
    cu_seqlens_with_zero = torch.cat([zero, cu_seqlens])
    lengths = (cu_seqlens_with_zero[1:] - cu_seqlens_with_zero[:-1]).tolist()
    lengths = [l for l in lengths if l > 0]
    if len(lengths) == 0:
        return torch.zeros(seq_length, embed_dim, device=device, dtype=hidden_states.dtype)

    query_splits = torch.split(query_states, lengths, dim=0)
    key_splits = torch.split(key_states, lengths, dim=0)
    value_splits = torch.split(value_states, lengths, dim=0)

    attn_outputs = []
    for q, k, v in zip(query_splits, key_splits, value_splits):
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_outputs.append(attn_output)

    attn_output = torch.cat(attn_outputs, dim=0)
    attn_output = attn_output.reshape(seq_length, embed_dim)
    attn_output = F.linear(attn_output, proj_weight, proj_bias)
    return attn_output


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
) -> torch.Tensor:
    """Variable-length vision attention with cu_seqlens.

    Picks batched attention when there are many sequences (loop overhead
    dominates) and the per-sequence loop when there are few (lower fixed cost).
    """
    embed_dim = 1152
    num_heads = 16
    head_dim = 72
    scaling = head_dim ** -0.5

    num_sequences = cu_seqlens.shape[0]

    if num_sequences >= 6:
        return _run_batched(
            hidden_states, cu_seqlens, cos, sin,
            qkv_weight, qkv_bias, proj_weight, proj_bias,
            embed_dim, num_heads, head_dim, scaling,
        )
    return _run_per_sequence(
        hidden_states, cu_seqlens, cos, sin,
        qkv_weight, qkv_bias, proj_weight, proj_bias,
        embed_dim, num_heads, head_dim, scaling,
    )
