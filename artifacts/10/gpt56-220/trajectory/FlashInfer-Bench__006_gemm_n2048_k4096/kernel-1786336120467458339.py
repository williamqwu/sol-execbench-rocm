import torch

def run(A, B):
    if A.shape[0] == 1:
        return torch.mv(B, A[0]).unsqueeze(0)
    return torch.matmul(A, B.T)
