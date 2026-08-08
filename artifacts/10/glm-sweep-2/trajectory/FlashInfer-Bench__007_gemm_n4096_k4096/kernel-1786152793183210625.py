import torch

def run(A, B):
    with open("/tmp/dbg_kernel.txt", "a") as f:
        f.write(f"A {A.shape} {A.dtype} contig={A.is_contiguous()} stride={A.stride()} dev={A.device}\n")
        f.write(f"B {B.shape} {B.dtype} contig={B.is_contiguous()} stride={B.stride()} dev={B.device}\n")
    C = torch.matmul(A, B.T)
    return C
