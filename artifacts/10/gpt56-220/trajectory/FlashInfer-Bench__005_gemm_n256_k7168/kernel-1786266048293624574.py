import torch


def run(A, B):
    if A.shape[0] == 1:
        return torch.mm(A, B.t())
    return torch.mm(B, A.t()).t()
