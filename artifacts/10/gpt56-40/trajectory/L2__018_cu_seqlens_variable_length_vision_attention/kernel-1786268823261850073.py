import torch
import torch.nn.functional as F
from typing import Tuple


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with valid cu_seqlens that sum to total_seq_len."""
    total_seq_len = axes_and_scalars["total_seq_len"]
    num_sequences = axes_and_scalars["num_sequences"]
    embed_dim = axes_and_scalars["embed_dim"]
    num_heads = axes_and_scalars["num_heads"]
    head_dim = axes_and_scalars["head_dim"]
    qkv_dim = embed_dim * 3
    
    # Generate random sequence lengths that sum to total_seq_len
    # Ensure each sequence has at least 1 token
    if num_sequences >= total_seq_len:
        # Edge case: more sequences than tokens, give 1 token each to first total_seq_len sequences
        lengths = [1] * total_seq_len + [0] * (num_sequences - total_seq_len)
    else:
        # Distribute tokens among sequences
        base_len = total_seq_len // num_sequences
        remainder = total_seq_len % num_sequences
        lengths = [base_len] * num_sequences
        for i in range(remainder):
            lengths[i] += 1
    
    # Create cumulative sequence lengths (excluding initial 0)
    cu_seqlens = torch.cumsum(torch.tensor(lengths, dtype=torch.int64, device=device), dim=0)
    
    # Generate other inputs
    hidden_states = torch.randn(total_seq_len, embed_dim, device=device, dtype=torch.bfloat16)
    cos = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    sin = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    qkv_weight = torch.randn(qkv_dim, embed_dim, device=device, dtype=torch.bfloat16) * 0.02
    qkv_bias = torch.zeros(qkv_dim, device=device, dtype=torch.bfloat16)
    proj_weight = torch.randn(embed_dim, embed_dim, device=device, dtype=torch.bfloat16) * 0.02
    proj_bias = torch.zeros(embed_dim, device=device, dtype=torch.bfloat16)
    
    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens,
        "cos": cos,
        "sin": sin,
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "proj_weight": proj_weight,
        "proj_bias": proj_bias,
    }


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
    """Variable-length vision attention with cu_seqlens."""
    embed_dim = 1152
    num_heads = 16
    head_dim = 72
    scaling = head_dim ** -0.5
    
    seq_length = hidden_states.shape[0]
    
    # QKV projection: [seq_len, embed_dim] -> [seq_len, 3 * embed_dim]
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)
    # Each is now [seq_len, num_heads, head_dim]
    
    # Apply rotary position embeddings
    query_states, key_states = _apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    
    # A host copy is already required to drive the Python grouping loop.  Copy
    # cumulative endpoints directly and difference them on the host, avoiding
    # GPU allocation/concatenation/subtraction kernels for this tiny tensor.
    endpoints = cu_seqlens.tolist()
    previous = 0
    lengths = []
    for endpoint in endpoints:
        length = endpoint - previous
        if length > 0:
            lengths.append(length)
        previous = endpoint
    
    if len(lengths) == 0:
        # No valid sequences, return zeros
        return torch.zeros(seq_length, embed_dim, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Adjacent equal-length sequences can share one batched attention launch.
    attn_outputs = []
    token_start = 0
    seq_start = 0
    while seq_start < len(lengths):
        length = lengths[seq_start]
        seq_end = seq_start + 1
        while seq_end < len(lengths) and lengths[seq_end] == length:
            seq_end += 1
        batch = seq_end - seq_start
        token_end = token_start + batch * length
        q = query_states[token_start:token_end].reshape(batch, length, num_heads, head_dim).transpose(1, 2)
        k = key_states[token_start:token_end].reshape(batch, length, num_heads, head_dim).transpose(1, 2)
        v = value_states[token_start:token_end].reshape(batch, length, num_heads, head_dim).transpose(1, 2)

        q_bmm = q.flatten(0, 1)
        k_bmm = k.transpose(2, 3).flatten(0, 1)
        attn_weights = torch.baddbmm(
            q.new_empty((1, 1, 1)), q_bmm, k_bmm, beta=0, alpha=scaling
        ).reshape(batch, num_heads, length, length)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_outputs.append(attn_output.transpose(1, 2).reshape(batch * length, num_heads, head_dim))
        token_start = token_end
        seq_start = seq_end
    
    # Concatenate all sequences: [total_seq_len, num_heads, head_dim]
    attn_output = attn_outputs[0] if len(attn_outputs) == 1 else torch.cat(attn_outputs, dim=0)
    
    # Reshape and project: [total_seq_len, embed_dim]
    attn_output = attn_output.reshape(seq_length, embed_dim)
    attn_output = F.linear(attn_output, proj_weight, proj_bias)
    
    return attn_output
