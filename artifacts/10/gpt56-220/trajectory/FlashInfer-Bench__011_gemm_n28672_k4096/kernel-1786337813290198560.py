import torch

def run(A, B):
    return torch.einsum("mk,nk->mn", A, B)
