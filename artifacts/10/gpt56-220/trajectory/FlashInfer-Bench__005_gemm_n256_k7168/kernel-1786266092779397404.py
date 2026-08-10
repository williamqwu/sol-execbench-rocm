import torch


def run(A, B):
    M = A.shape[0]
    if M <= 32 and M & (M - 1) == 0:
        return torch.mm(A, B.t())
    return torch.mm(B, A.t()).t()
