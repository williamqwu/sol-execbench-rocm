import torch

def run(A, B):
    m = A.shape[0]
    if m <= 4 or 8 <= m <= 32 or 48 <= m <= 64 or 88 <= m <= 128 or 200 <= m <= 224:
        C = torch.empty((m, B.shape[0]), device=A.device, dtype=A.dtype)
        torch.ops.aten.mm.out(B, A.T, out=C.T)
        return C
    return torch.ops.aten.mm.default(A, B.T)
