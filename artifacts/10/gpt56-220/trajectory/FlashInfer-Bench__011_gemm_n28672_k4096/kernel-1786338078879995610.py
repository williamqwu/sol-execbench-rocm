import torch


@torch.compile(mode="max-autotune")
def run(A, B):
    return torch.mm(A, B.t())
