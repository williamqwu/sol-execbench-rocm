import torch

_compiled = torch.compile(torch.matmul, mode="max-autotune", fullgraph=True)

def run(A, B):
    C = _compiled(A, B.T)
    return C
