import torch

_mm = torch.ops.aten.mm.default

def run(A, B):
    return _mm(A, B.t())
