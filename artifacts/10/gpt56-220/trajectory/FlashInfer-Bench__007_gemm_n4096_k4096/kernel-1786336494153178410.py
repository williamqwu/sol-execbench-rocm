import torch

torch.backends.cuda.matmul.allow_fp16_accumulation = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

def run(A, B):
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    torch.mm(A, B.T, out=C)
    return C
