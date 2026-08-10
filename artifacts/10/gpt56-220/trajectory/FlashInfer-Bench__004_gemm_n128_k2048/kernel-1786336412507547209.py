import torch

def run(A, B):
    return torch.mm(A, B.mT)
