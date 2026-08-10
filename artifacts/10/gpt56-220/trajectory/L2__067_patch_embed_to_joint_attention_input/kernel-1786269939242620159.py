import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    pos_embed: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    pooled_linear1_weight: torch.Tensor,
    pooled_linear1_bias: torch.Tensor,
    pooled_linear2_weight: torch.Tensor,
    pooled_linear2_bias: torch.Tensor,
    context_embedder_weight: torch.Tensor,
    context_embedder_bias: torch.Tensor,
    freqs: torch.Tensor,
):
    batch_size = hidden_states.shape[0]
    
    # 1. Patch Embedding: Conv2d for patch extraction
    # (B, C, H, W) -> (B, inner_dim, H//patch_size, W//patch_size)
    # Materialize non-overlapping 2x2 patches in sequence order, then use GEMM.
    # Weight flattening preserves conv's (channel, kh, kw) reduction order.
    patch_embedded = hidden_states.reshape(batch_size, 16, 64, 2, 64, 2)
    patch_embedded = patch_embedded.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, 4096, 64)
    patch_embedded = F.linear(patch_embedded, proj_weight.reshape(2432, 64), proj_bias)
    
    # 2. Add Positional Embeddings
    num_patches = patch_embedded.shape[1]
    pos_embed_slice = pos_embed[:, :num_patches, :]
    output_hidden_states = patch_embedded + pos_embed_slice
    
    # 3. Timestep Embedding: sinusoidal encoding
    # timesteps: (B,) -> (B, 1)
    timestep_expanded = timestep.unsqueeze(-1)
    
    # Compute sinusoidal features: (B, 128)
    args = timestep_expanded * freqs.unsqueeze(0)
    timestep_sinusoidal = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 256)
    
    # Timestep MLP: Linear -> SiLU -> Linear
    timestep_embed = F.linear(timestep_sinusoidal, timestep_linear1_weight, timestep_linear1_bias)
    timestep_embed = F.silu(timestep_embed)
    timestep_embed = F.linear(timestep_embed, timestep_linear2_weight, timestep_linear2_bias)
    
    # 4. Pooled Projection Embedding: Linear -> SiLU -> Linear
    pooled_embed = F.linear(pooled_projections, pooled_linear1_weight, pooled_linear1_bias)
    pooled_embed = F.silu(pooled_embed)
    pooled_embed = F.linear(pooled_embed, pooled_linear2_weight, pooled_linear2_bias)
    
    # 5. Combine temporal embeddings
    temb = timestep_embed + pooled_embed
    
    # 6. Context Embedder: Linear projection
    output_encoder_hidden_states = F.linear(encoder_hidden_states, context_embedder_weight, context_embedder_bias)
    
    return output_hidden_states, temb, output_encoder_hidden_states
