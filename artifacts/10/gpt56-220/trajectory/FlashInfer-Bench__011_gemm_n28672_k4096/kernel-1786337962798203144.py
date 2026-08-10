import torch


def run(A, B):
    return torch.ops.aten.mm.default(A, B.t())
