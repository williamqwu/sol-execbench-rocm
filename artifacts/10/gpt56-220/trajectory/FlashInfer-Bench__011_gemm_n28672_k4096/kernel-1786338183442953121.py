import torch


def run(A, B):
    return torch.linalg.multi_dot((A, B.t()))
