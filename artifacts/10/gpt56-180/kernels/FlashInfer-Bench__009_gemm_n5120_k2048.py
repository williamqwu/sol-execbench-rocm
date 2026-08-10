import torch

@torch.compile(fullgraph=True, dynamic=True)
def run(A, B):
    return torch.mm(A, B.mT)
