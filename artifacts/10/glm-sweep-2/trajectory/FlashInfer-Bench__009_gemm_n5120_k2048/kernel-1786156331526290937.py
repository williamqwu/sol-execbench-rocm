import torch

def run(A, B):
    # C = A @ B.T. Use bmm with batch dim for lower dispatch overhead on small M.
    C = torch.bmm(A.unsqueeze(0), B.unsqueeze(0).transpose(1, 2)).squeeze(0)
    return C
