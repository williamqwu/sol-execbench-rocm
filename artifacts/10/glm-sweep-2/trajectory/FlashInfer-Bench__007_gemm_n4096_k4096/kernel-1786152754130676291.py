import torch

def run(A, B):
    import sys
    print(f"DBG A {A.shape} {A.dtype} contig={A.is_contiguous()} dev={A.device}", file=sys.stderr)
    print(f"DBG B {B.shape} {B.dtype} contig={B.is_contiguous()} dev={B.device}", file=sys.stderr)
    print(f"DBG A.stride={A.stride()} B.stride={B.stride()}", file=sys.stderr)
    C = torch.matmul(A, B.T)
    return C
