import torch
import torch.nn.functional as F

torch.cuda.tunable.enable(True)
torch.cuda.tunable.tuning_enable(False)

def run(A, B):
    return F.linear(A, B)
