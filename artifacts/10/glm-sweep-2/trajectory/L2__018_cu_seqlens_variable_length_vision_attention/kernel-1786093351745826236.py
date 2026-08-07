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
    """Variable-length vision attention with cu_seqlens (batched SDPA)."""
    embed_dim = 1152
    num_heads = 16
    head_dim = 72
    scaling = head_dim ** -0.5

    seq_length = hidden_states.shape[0]
    device = hidden_states.device

    # QKV projection: [seq_len, embed_dim] -> [seq_len, 3, num_heads, head_dim]
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)
    # Each is now [seq_len, num_heads, head_dim]

    # Apply rotary position embeddings (on the full concatenated tensor)
    query_states, key_states = _apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    # Build cu_seqlens with prepended 0
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    cu_seqlens_with_zero = torch.cat([zero, cu_seqlens])
    lengths_all = cu_seqlens_with_zero[1:] - cu_seqlens_with_zero[:-1]

    # Filter out zero-length sequences
    valid = lengths_all > 0
    lengths = lengths_all[valid]
    num_seq = lengths.shape[0]
    if num_seq == 0:
        return torch.zeros(seq_length, embed_dim, device=device, dtype=hidden_states.dtype)

    max_len = int(lengths.max().item())
    offsets = torch.cat([zero, lengths.cumsum(0)])  # [num_seq+1]

    # Build padded batch via gather: each sequence -> its own batch slot
    arange_max = torch.arange(max_len, device=device)
    idx = offsets[:-1, None] + arange_max[None, :]  # [num_seq, max_len]
    pad_key = arange_max[None, :] < lengths[:, None]  # [num_seq, max_len] bool
    idx_safe = torch.where(pad_key, idx, torch.zeros_like(idx))

    # gather: q[idx_safe] -> [num_seq, max_len, num_heads, head_dim]
    q_pad = query_states[idx_safe].permute(0, 2, 1, 3)  # [num_seq, H, max_len, D]
    k_pad = key_states[idx_safe].permute(0, 2, 1, 3)
    v_pad = value_states[idx_safe].permute(0, 2, 1, 3)

    # key-only padding mask (True = attend); broadcast over query/heads
    attn_mask = pad_key[:, None, None, :]  # [num_seq, 1, 1, max_len]

    # Single batched attention call
    attn_output = F.scaled_dot_product_attention(
        q_pad, k_pad, v_pad, attn_mask=attn_mask
    )  # [num_seq, H, max_len, D]

    # Scatter back to concatenated [seq_length, num_heads, head_dim]
    seq_id = torch.repeat_interleave(
        torch.arange(num_seq, device=device), lengths
    )  # [seq_length]
    pos_id = torch.arange(seq_length, device=device) - offsets[:-1][seq_id]
    attn_out = attn_output.permute(0, 2, 1, 3)[seq_id, pos_id]  # [seq_length, H, D]

    # Reshape and project
    attn_out = attn_out.reshape(seq_length, embed_dim)
    attn_out = F.linear(attn_out, proj_weight, proj_bias)
    return attn_out
