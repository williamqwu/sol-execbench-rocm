import torch

torch.backends.cuda.matmul.allow_fp16_accumulation = True
_zero = torch.zeros((), device="cuda", dtype=torch.float16)

def run(A, B):
    return torch.baddbmm(_zero, A.unsqueeze(0), B.T.unsqueeze(0), beta=0).squeeze(0)
