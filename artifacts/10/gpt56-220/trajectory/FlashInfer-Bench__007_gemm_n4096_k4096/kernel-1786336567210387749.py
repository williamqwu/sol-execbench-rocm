import torch

torch.backends.cuda.matmul.allow_fp16_accumulation = True
_zero = torch.zeros((), device="cuda", dtype=torch.float16)

def run(A, B):
    return torch.addmm(_zero, A, B.T, beta=0)
