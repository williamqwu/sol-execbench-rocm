import torch


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    cos_half = (freqs.cos() * attention_scaling).to(torch.bfloat16)
    sin_half = (freqs.sin() * attention_scaling).to(torch.bfloat16)
    return (
        torch.cat((cos_half, cos_half), dim=-1),
        torch.cat((sin_half, sin_half), dim=-1),
    )
