import torch


def run(A, B):
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    torch.mm(B, A.t(), out=C.t())
    return C
