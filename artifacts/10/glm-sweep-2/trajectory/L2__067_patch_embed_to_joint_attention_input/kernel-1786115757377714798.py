import torch
import torch.nn.functional as F
import math

def _impl(
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

    # Patch embedding: conv -> flatten to [B, inner_dim, num_patches] (contiguous)
    patch_embedded = F.conv2d(hidden_states, proj_weight, proj_bias, stride=2).flatten(2)

    # Add positional embeddings in the contiguous [B, inner_dim, num_patches] layout,
    # then return a transposed view [B, num_patches, inner_dim]. Avoids a transpose-then-add.
    num_patches = patch_embedded.shape[2]
    pos_embed_t = pos_embed[:, :num_patches, :].transpose(1, 2)
    output_hidden_states = (patch_embedded + pos_embed_t).transpose(1, 2)

    # Timestep Embedding
    timestep_expanded = timestep.unsqueeze(-1)
    args = timestep_expanded * freqs.unsqueeze(0)
    timestep_sinusoidal = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    timestep_embed = F.linear(timestep_sinusoidal, timestep_linear1_weight, timestep_linear1_bias)
    timestep_embed = F.silu(timestep_embed)
    timestep_embed = F.linear(timestep_embed, timestep_linear2_weight, timestep_linear2_bias)

    # Pooled Projection Embedding
    pooled_embed = F.linear(pooled_projections, pooled_linear1_weight, pooled_linear1_bias)
    pooled_embed = F.silu(pooled_embed)
    pooled_embed = F.linear(pooled_embed, pooled_linear2_weight, pooled_linear2_bias)

    temb = timestep_embed + pooled_embed

    # Context Embedder
    output_encoder_hidden_states = F.linear(encoder_hidden_states, context_embedder_weight, context_embedder_bias)

    return output_hidden_states, temb, output_encoder_hidden_states

_compiled = torch.compile(_impl, dynamic=True, fullgraph=True)

@torch.no_grad()
def run(*args):
    return _compiled(*args)
