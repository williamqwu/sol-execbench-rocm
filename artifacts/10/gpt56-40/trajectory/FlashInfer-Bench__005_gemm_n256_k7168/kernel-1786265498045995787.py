import torch
import torch.nn.functional as F

def run(A, B):
    return F.linear(A, B)
