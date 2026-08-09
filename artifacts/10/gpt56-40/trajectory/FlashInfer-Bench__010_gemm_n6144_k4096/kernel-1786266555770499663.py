import torch

def run(A, B):
    m = A.shape[0]
    if m <= 4 or 8 <= m <= 32 or 48 <= m <= 64 or 88 <= m <= 128 or 200 <= m <= 224:
        return torch.matmul(B, A.T).T
    return torch.matmul(A, B.T)
