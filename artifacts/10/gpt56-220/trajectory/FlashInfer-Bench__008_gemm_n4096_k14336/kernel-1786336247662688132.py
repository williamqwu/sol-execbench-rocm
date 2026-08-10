import torch
def run(A, B):
    return torch._C._nn.linear(A, B)
