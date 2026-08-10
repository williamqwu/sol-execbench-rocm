import torch

torch.backends.cuda.preferred_blas_library("rocblas")

def run(A, B):
    C = torch.matmul(A, B.T)
    return C
