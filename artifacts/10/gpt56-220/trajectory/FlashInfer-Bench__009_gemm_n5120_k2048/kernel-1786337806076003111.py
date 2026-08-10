import torch

@torch.compile(fullgraph=True, dynamic=True)
def run(A, B):
    return torch.bmm(A.unsqueeze(0), B.t().unsqueeze(0)).squeeze(0)
