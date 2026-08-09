import torch
import torch.nn.functional as F


def run(A, B):
    M = A.shape[0]
    if M == 1 or M == 32 or 256 <= M < 2048:
        return torch.mm(A, B.t())
    return F.linear(B, A).t()
