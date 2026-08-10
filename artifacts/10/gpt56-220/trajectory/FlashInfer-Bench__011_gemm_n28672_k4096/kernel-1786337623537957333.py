import torch

def run(A, B):
    return torch.mm(B, A.t()).t()
