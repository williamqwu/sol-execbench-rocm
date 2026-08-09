import torch

torch.cuda.tunable.enable(True)
torch.cuda.tunable.tuning_enable(True)
torch.cuda.tunable.set_max_tuning_iterations(30)

def run(A, B):
    return torch.bmm(A.unsqueeze(0), B.T.unsqueeze(0))[0]
