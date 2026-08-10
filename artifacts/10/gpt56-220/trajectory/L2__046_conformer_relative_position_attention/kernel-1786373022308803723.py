import torch
import torch.nn.functional as F
import math

_rel_seq = torch.arange(512, device="cuda")
_ATTENTION_DISTS = _rel_seq.view(-1, 1) - _rel_seq.view(1, -1) + 512

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    pre_norm_weight: torch.Tensor,
    pre_norm_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_kv_weight: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    rel_pos_emb_weight: torch.Tensor,
    scale: float,
):
    # Constants
    hidden_dim = 1024
    num_heads = 8
    dim_head = 128
    max_pos_emb = 512
    context_size = 512
    inner_dim = 1024
    
    bsz, num_features, _ = hidden_states.shape
    
    # Pre-normalization (LayerNorm)
    hidden_states = F.layer_norm(hidden_states, (hidden_dim,), pre_norm_weight, pre_norm_bias)
    
    # Calculate blocking parameters
    num_blocks = math.ceil(num_features / context_size)
    remainder = num_features % context_size
    
    query_states = F.linear(hidden_states, to_q_weight)
    key_value_states = F.linear(hidden_states, to_kv_weight)
    if remainder > 0:
        pad_amount = context_size - remainder
        query_states = F.pad(query_states, (0, 0, 0, pad_amount))
        key_value_states = F.pad(key_value_states, (0, 0, 0, pad_amount))

    key_states, value_states = key_value_states.chunk(2, dim=-1)
    
    # Reshape into blocks: (B, num_blocks, context_size, num_heads, dim_head)
    query_states = query_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)  # (B, num_blocks, num_heads, context_size, dim_head)
    query_states = query_states * scale
    
    key_states = key_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)
    
    value_states = value_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)
    
    # Compute relative position distances for context window
    device = hidden_states.device
    attention_dists = _ATTENTION_DISTS
    
    # A dense distance GEMM is fastest for small workloads, where it exposes
    # enough parallelism to saturate the device. At larger volumes the direct
    # contraction avoids the dense method's nearly 2x excess arithmetic.
    if bsz * num_blocks < 16:
        all_pos = F.linear(query_states, rel_pos_emb_weight)
        pos_attn = torch.gather(
            all_pos, -1,
            attention_dists.view(1, 1, 1, context_size, context_size).expand(
                bsz, num_blocks, num_heads, -1, -1
            ),
        )
    else:
        rel_pos_emb = F.embedding(attention_dists, rel_pos_emb_weight)
        pos_attn = torch.einsum(
            'bmhcd,crd->bmhcr', query_states, rel_pos_emb
        )
    
    pos_attn = pos_attn.to(query_states.dtype)
    
    # Scaled dot-product attention with positional bias
    if remainder == 0:
        out = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=pos_attn, scale=1.0,
        )
    else:
        last = F.scaled_dot_product_attention(
            query_states[:, -1:, :, :remainder],
            key_states[:, -1:, :, :remainder],
            value_states[:, -1:, :, :remainder],
            attn_mask=pos_attn[:, -1:, :, :remainder, :remainder],
            scale=1.0,
        )
        last = F.pad(last, (0, 0, 0, context_size - remainder))
        if num_blocks == 1:
            out = last
        else:
            full = F.scaled_dot_product_attention(
                query_states[:, :-1], key_states[:, :-1], value_states[:, :-1],
                attn_mask=pos_attn[:, :-1], scale=1.0,
            )
            out = torch.cat((full, last), dim=1)
    
    # Reshape back: (B, M, Nh, C, D) -> (B, S_padded, inner_dim)
    out = out.transpose(2, 3).reshape(bsz, num_blocks * context_size, inner_dim)
    
    # Remove padding and project to output
    out = out[:, :num_features, :]
    out = F.linear(out, to_out_weight, to_out_bias)
    
    return out
