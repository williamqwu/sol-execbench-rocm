import torch

@torch.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
def run(A, B):
    return torch.mm(A, B.t())
