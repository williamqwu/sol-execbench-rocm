import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
@torch.no_grad()
def run(emb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    emb_out = F.linear(emb, weight, bias)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=1)
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
