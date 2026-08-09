import torch

torch.cuda.tunable.enable(True)
torch.cuda.tunable.tuning_enable(True)
torch.cuda.tunable.set_max_tuning_iterations(20)


def run(A, B):
    return torch.mm(A, B.t())
