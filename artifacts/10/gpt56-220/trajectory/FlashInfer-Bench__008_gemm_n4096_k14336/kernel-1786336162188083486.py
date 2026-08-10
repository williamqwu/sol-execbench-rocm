import torch
import torch.nn.functional as F


@torch.compile(mode="max-autotune")
def run(A, B):
    return F.linear(A, B)
