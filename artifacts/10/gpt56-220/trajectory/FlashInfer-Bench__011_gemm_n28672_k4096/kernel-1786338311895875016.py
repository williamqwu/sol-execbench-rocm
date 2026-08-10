import torch


def run(A, B):
    return torch.tensordot(A, B, dims=([1], [1]))
