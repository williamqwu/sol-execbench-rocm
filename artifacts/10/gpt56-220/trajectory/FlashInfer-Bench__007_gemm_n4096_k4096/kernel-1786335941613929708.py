import torch

@torch.compile(fullgraph=True)
def run(A, B):
    return torch.matmul(A, B.T)
