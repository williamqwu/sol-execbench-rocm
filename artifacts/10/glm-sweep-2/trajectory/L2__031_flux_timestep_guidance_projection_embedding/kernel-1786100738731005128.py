import torch
import math

_compiled_run = None

@torch.no_grad()
def run(
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor,
    freqs: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    text_embedder_weight: torch.Tensor,
    text_embedder_bias: torch.Tensor,
):
    global _compiled_run
    if _compiled_run is None:
        _compiled_run = torch.compile(_run_impl, mode="max-autotune-no-cudagraphs", dynamic=False)
    return _compiled_run(
        timestep, pooled_projections, freqs,
        timestep_linear1_weight, timestep_linear1_bias,
        timestep_linear2_weight, timestep_linear2_bias,
        text_embedder_weight, text_embedder_bias,
    )

def _run_impl(
    timestep, pooled_projections, freqs,
    timestep_linear1_weight, timestep_linear1_bias,
    timestep_linear2_weight, timestep_linear2_bias,
    text_embedder_weight, text_embedder_bias,
):
    timestep_scaled = timestep * 1000.0
    args = timestep_scaled[:, None] * freqs[None, :]
    sin_embed = torch.sin(args)
    cos_embed = torch.cos(args)
    timestep_embed = torch.cat([cos_embed, sin_embed], dim=-1)
    x = torch.nn.functional.linear(timestep_embed, timestep_linear1_weight, timestep_linear1_bias)
    x = x * torch.sigmoid(x)
    timestep_embed = torch.nn.functional.linear(x, timestep_linear2_weight, timestep_linear2_bias)
    text_embed = torch.nn.functional.linear(pooled_projections, text_embedder_weight, text_embedder_bias)
    conditioning = timestep_embed + text_embed
    return conditioning
