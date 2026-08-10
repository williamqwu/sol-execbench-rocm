import torch

@torch.compile(fullgraph=True, dynamic=True)
def run(A, B):
    if A.shape[0] <= 8:
        return torch.bmm(A.unsqueeze(0), B.t().unsqueeze(0)).squeeze(0)
    return torch.mm(A, B.t())
