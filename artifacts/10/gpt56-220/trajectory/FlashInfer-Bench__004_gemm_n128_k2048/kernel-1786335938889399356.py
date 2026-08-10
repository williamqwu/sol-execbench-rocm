import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True, dynamic=True)
def run(A, B):
    return F.linear(A, B)
