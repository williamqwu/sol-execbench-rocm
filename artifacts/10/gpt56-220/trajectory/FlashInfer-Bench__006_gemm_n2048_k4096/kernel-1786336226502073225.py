import torch

def run(A, B):
    if A.shape[0] <= 16:
        return torch.mm(B, A.T).T
    return torch.matmul(A, B.T)
