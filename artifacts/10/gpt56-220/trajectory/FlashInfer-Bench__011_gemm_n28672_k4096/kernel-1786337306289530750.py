import torch

@torch.compile
def run(A, B):
    return torch.mm(A, B.t())
