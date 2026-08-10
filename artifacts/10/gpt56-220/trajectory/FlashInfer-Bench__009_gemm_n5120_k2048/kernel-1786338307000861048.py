import torch

@torch.jit.script
def run(A, B):
    return torch.mm(A, B.t())
