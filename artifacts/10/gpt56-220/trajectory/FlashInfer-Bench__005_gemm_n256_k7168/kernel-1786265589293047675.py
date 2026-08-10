import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True, dynamic=True, mode="max-autotune")
def run(A, B):
    return F.linear(A, B)
