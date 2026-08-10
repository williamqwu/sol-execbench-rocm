import torch


@torch.compile(fullgraph=True, dynamic=False)
def _compiled(freqs: torch.Tensor, attention_scaling: float):
    emb = torch.cat((freqs, freqs), dim=-1)
    return (
        (emb.cos() * attention_scaling).to(torch.bfloat16),
        (emb.sin() * attention_scaling).to(torch.bfloat16),
    )


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    return _compiled(freqs, attention_scaling)
