import torch
import torch.nn.functional as F
import math

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
    
    # Project to Q, K, V
    query_states = F.linear(hidden_states, to_q_weight)
    key_value_states = F.linear(hidden_states, to_kv_weight)

    # The projections have no bias, so projecting zero padding is exactly
    # equivalent to padding their outputs, while avoiding wasted GEMM work.
    if remainder > 0:
        pad_amount = context_size - remainder
        query_states = F.pad(query_states, (0, 0, 0, pad_amount))
        key_value_states = F.pad(key_value_states, (0, 0, 0, pad_amount))

    key_states, value_states = key_value_states.chunk(2, dim=-1)
    
    # Reshape into blocks: (B, num_blocks, context_size, num_heads, dim_head)
    query_states = query_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)  # (B, num_blocks, num_heads, context_size, dim_head)
    
    key_states = key_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)
    
    value_states = value_states.reshape(
        bsz, num_blocks, context_size, num_heads, dim_head
    ).transpose(2, 3)
    
    # Compute relative position distances for context window
    device = hidden_states.device
    seq = torch.arange(context_size, device=device)
    relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
    attention_dists = torch.clamp(relpos_dist, -context_size, context_size) + max_pos_emb
    
    # Compute all relative-distance scores as one MFMA-friendly GEMM, then
    # select the Toeplitz diagonals required by each query position.
    all_pos = F.linear(query_states, rel_pos_emb_weight)
    pos_attn = torch.gather(
        all_pos, -1,
        attention_dists.view(1, 1, 1, context_size, context_size).expand(
            bsz, num_blocks, num_heads, -1, -1
        ),
    ) * scale
    
    # Apply masking for incomplete final block
    if remainder > 0:
        mask = torch.ones(
            context_size, context_size, 
            dtype=torch.bool, 
            device=device
        )
        mask[:remainder, :remainder] = False
        mask_value = -torch.finfo(pos_attn.dtype).max
        pos_attn[:, -1, :].masked_fill_(mask, mask_value)
    
    pos_attn = pos_attn.to(query_states.dtype)
    
    # Scaled dot-product attention with positional bias
    # Force MATH backend to ensure pos_attn bias is applied correctly
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=pos_attn,  # Relative position bias
            scale=scale
        )
    
    # Reshape back: (B, M, Nh, C, D) -> (B, S_padded, inner_dim)
    out = out.transpose(2, 3).reshape(bsz, num_blocks * context_size, inner_dim)
    
    # Remove padding and project to output
    out = out[:, :num_features, :]
    out = F.linear(out, to_out_weight, to_out_bias)
    
    return out
