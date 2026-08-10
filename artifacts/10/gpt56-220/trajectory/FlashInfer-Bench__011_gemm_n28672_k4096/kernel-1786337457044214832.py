import torch

torch.backends.cuda.preferred_blas_library("ck")

def run(A, B):
    return torch.mm(A, B.t())
