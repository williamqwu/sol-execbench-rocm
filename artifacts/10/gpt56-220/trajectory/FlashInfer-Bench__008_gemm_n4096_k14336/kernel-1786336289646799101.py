import torch
def run(A, B):
    return torch.ops.aten.mm.dtype(A, B.T, out_dtype=torch.float16)
