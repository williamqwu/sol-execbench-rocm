import torch


def _half_then_duplicate(position_ids: torch.Tensor, inv_freq: torch.Tensor,
                         attention_scaling: float) -> torch.Tensor:
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    pairs = torch.stack((freqs.cos() * attention_scaling,
                         freqs.sin() * attention_scaling), dim=-1).bfloat16()
    return torch.cat((pairs, pairs), dim=-2)


@torch.no_grad()
def run(position_ids: torch.Tensor, inv_freq: torch.Tensor,
        attention_scaling: float) -> torch.Tensor:
    return _half_then_duplicate(position_ids, inv_freq, attention_scaling)
