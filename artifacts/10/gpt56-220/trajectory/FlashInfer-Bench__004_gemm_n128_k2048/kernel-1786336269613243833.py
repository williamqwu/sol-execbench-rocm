import torch

def run(A, B):
    return torch.ops.aten.linear.default(A, B, None)
