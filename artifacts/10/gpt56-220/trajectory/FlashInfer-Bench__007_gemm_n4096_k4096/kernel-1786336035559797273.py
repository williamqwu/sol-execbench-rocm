import torch

torch.backends.cuda.preferred_blas_library("hipblaslt")

def run(A, B):
    C = torch.matmul(A, B.T)
    return C
