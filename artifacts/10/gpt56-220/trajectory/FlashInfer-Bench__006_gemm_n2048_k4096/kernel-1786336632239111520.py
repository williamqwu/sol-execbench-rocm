import torch
import torch.nn.functional as F

def run(A, B):
    if A.shape[0] == 1:
        return torch.mv(B, A[0]).unsqueeze(0)
    if A.shape[0] <= 64:
        return torch.mm(B, A.T).T
    return F.linear(A, B)
