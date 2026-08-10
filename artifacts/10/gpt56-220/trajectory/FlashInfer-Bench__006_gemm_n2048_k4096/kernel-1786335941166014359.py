import torch

@torch.compile(fullgraph=True, dynamic=True)
def run(A, B):
    return torch.matmul(A, B.T)
