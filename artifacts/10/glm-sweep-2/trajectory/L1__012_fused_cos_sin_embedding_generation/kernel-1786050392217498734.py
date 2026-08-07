import torch

@torch.compile(dynamic=True, mode="reduce-overhead")
@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).to(torch.bfloat16)
    sin = (emb.sin() * attention_scaling).to(torch.bfloat16)
    return cos, sin
