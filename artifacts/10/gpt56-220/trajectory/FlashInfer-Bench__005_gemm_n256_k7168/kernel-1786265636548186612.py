import torch

_linear = torch._C._nn.linear

def run(A, B):
    return _linear(A, B)
