import torch

torch.backends.cuda.preferred_blas_library("hipblaslt")

def run(A, B):
    return torch.mm(A, B.t())
