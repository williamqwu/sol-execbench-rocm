import torch

torch.backends.cuda.matmul.allow_fp16_accumulation = True

def run(A, B):
    return torch.mm(A, B.T)
