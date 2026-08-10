import torch

_one = torch.tensor(1.0, device="cuda", dtype=torch.float32)

def run(A, B):
    return torch._scaled_mm(A, B.T, _one, _one,
                            out_dtype=torch.float16, use_fast_accum=True)
