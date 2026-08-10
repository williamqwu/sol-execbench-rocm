import torch

def run(A, B):
    return torch.bmm(A.unsqueeze(0), B.t().unsqueeze(0)).squeeze(0)
