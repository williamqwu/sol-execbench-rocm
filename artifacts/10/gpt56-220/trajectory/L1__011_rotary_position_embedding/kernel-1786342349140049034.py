import torch


@torch.compile(fullgraph=True, dynamic=True)
def _compiled(position_ids: torch.Tensor, inv_freq: torch.Tensor,
              attention_scaling: float) -> torch.Tensor:
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.stack((emb.cos() * attention_scaling,
                        emb.sin() * attention_scaling), dim=-1).bfloat16()


@torch.no_grad()
def run(position_ids: torch.Tensor, inv_freq: torch.Tensor,
        attention_scaling: float) -> torch.Tensor:
    return _compiled(position_ids, inv_freq, attention_scaling)
