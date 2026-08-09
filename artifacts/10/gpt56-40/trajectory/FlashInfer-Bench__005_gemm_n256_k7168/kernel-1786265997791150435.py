import torch


def run(A, B):
    if 256 <= A.shape[0] < 2048:
        return torch.mm(A, B.t())
    return torch.mm(B, A.t()).t()
