import torch

def run(A, B):
    return torch.ops.aten.mm.default(A, torch.ops.aten.t.default(B))
