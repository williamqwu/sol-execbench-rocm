import torch

torch.backends.cuda.matmul.allow_fp16_accumulation = True

def run(A, B):
    return torch.bmm(A.unsqueeze(0), B.T.unsqueeze(0)).squeeze(0)
