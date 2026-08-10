import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_fp16_accumulation = True

def run(A, B):
    return F.linear(A, B)
