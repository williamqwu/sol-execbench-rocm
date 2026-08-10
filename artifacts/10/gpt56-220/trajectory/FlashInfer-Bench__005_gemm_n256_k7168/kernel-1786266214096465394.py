import torch


def run(A, B):
    M = A.shape[0]
    if M == 1 or M == 32 or 256 <= M < 2048:
        return torch.mm(A, B.t())
    return torch.mm(B, A.t()).t()
