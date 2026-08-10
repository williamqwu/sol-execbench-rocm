import torch

def run(A, B):
    return torch.mm(A, B.t(), out_dtype=torch.float16)
