import torch

def run(A, B):
    if A.shape[0] == 1:
        return torch.mv(B, A[0]).unsqueeze(0)
    if A.shape[0] <= 16:
        return torch.mm(B, A.T).T
    return torch.mm(A, B.T)
