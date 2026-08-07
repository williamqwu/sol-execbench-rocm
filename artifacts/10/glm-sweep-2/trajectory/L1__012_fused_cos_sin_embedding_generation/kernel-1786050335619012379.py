import torch

@torch.compile(dynamic=True)
@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    cos_half = torch.cos(freqs) * attention_scaling
    sin_half = torch.sin(freqs) * attention_scaling
    cos = torch.cat((cos_half, cos_half), dim=-1).to(torch.bfloat16)
    sin = torch.cat((sin_half, sin_half), dim=-1).to(torch.bfloat16)
    return cos, sin
