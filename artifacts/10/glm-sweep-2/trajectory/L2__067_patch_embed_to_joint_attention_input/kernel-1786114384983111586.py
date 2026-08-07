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

    # stride=2 kernel=2 conv == non-overlapping patches -> reshape + matmul
    # hidden_states: [B, 16, 128, 128] -> [B, 64, 2, 64, 2, 16, 2, 2] -> [B, 4096, 64]
    B, C, H, W = hidden_states.shape
    ps = 2
    h = H // ps
    w = W // ps
    # [B, C, h, ps, w, ps] -> [B, h, w, C, ps, ps] -> [B, h*w, C*ps*ps]
    x = hidden_states.view(B, C, h, ps, w, ps).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * ps * ps)
    wr = proj_weight.reshape(proj_weight.shape[0], -1).t()  # [64, 2432]
    patch_embedded = torch.matmul(x, wr) + proj_bias  # [B, 4096, 2432]

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

_compiled = torch.compile(_impl, dynamic=True, fullgraph=True)

@torch.no_grad()
def run(*args):
    return _compiled(*args)
