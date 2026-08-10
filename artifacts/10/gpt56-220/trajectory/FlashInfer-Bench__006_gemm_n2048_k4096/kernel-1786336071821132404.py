import torch

def run(A, B):
    torch.backends.cuda.preferred_blas_library("hipblas")
    C = torch.matmul(A, B.T)
    torch.backends.cuda.preferred_blas_library("hipblaslt")
    return C
