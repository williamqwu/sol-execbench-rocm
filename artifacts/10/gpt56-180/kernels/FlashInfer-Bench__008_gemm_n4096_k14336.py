import torch
import torch.nn.functional as F

torch.cuda.tunable.enable(True)
torch.cuda.tunable.tuning_enable(True)
torch.cuda.tunable.set_max_tuning_duration(100)
torch.cuda.tunable.set_max_tuning_iterations(20)

def run(A, B):
    return F.linear(A, B)
