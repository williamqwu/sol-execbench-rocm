import torch

def run(A, B):
    C = torch.empty((A.shape[0], 128), dtype=A.dtype, device=A.device)
    return torch.mm(A, B.mT, out=C)
