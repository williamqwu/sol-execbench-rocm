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

    patch_embedded = F.conv2d(hidden_states, proj_weight, proj_bias, stride=2).flatten(2).transpose(1, 2)

    num_patches = patch_embedded.shape[1]
    pos_embed_slice = pos_embed[:, :num_patches, :]
    output_hidden_states = patch_embedded + pos_embed_slice

    timestep_expanded = timestep.unsqueeze(-1)
    args = timestep_expanded * freqs.unsqueeze(0)
    timestep_sinusoidal = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    timestep_embed = F.linear(timestep_sinusoidal, timestep_linear1_weight, timestep_linear1_bias)
    timestep_embed = F.silu(timestep_embed)
    timestep_embed = F.linear(timestep_embed, timestep_linear2_weight, timestep_linear2_bias)

    pooled_embed = F.linear(pooled_projections, pooled_linear1_weight, pooled_linear1_bias)
    pooled_embed = F.silu(pooled_embed)
    pooled_embed = F.linear(pooled_embed, pooled_linear2_weight, pooled_linear2_bias)

    temb = timestep_embed + pooled_embed

    output_encoder_hidden_states = F.linear(encoder_hidden_states, context_embedder_weight, context_embedder_bias)

    return output_hidden_states, temb, output_encoder_hidden_states

# Cache of static-shape compiled graphs, keyed by (batch_size, seq_len_context).
_static_cache: dict = {}

@torch.no_grad()
def run(*args):
    hidden_states = args[0]
    encoder_hidden_states = args[1]
    key = (hidden_states.shape[0], encoder_hidden_states.shape[1])
    compiled = _static_cache.get(key)
    if compiled is None:
        compiled = torch.compile(_impl, dynamic=False, fullgraph=True)
        _static_cache[key] = compiled
    return compiled(*args)
