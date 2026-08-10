import torch

torch.backends.cuda.preferred_blas_library("hipblas")

def run(A, B):
    C = torch.matmul(A, B.T)
    return C
