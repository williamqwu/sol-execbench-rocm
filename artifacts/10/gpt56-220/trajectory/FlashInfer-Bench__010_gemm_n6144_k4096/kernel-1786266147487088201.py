import torch

def run(A, B):
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    return torch.mm(A, B.T, out=C)
