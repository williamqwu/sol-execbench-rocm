import torch

torch.backends.cuda.matmul.allow_fp16_accumulation = True

def run(A, B):
    return torch.ops.aten.mm.default(A, B.T)
