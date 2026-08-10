import torch

@torch.compile(fullgraph=True, mode="max-autotune")
def run(A, B):
    return torch.mm(A, B.T)
