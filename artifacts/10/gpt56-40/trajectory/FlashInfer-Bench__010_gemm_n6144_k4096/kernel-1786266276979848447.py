import torch

def run(A, B):
    return torch.ops.aten.mm.default(B, A.T).T
