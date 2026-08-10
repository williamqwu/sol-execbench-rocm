import torch

@torch.compile(fullgraph=True, dynamic=False)
def run(A, B):
    return torch.mm(A, B.mT)
