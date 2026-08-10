import torch

torch.cuda.tunable.enable(True)
torch.cuda.tunable.tuning_enable(True)
torch.cuda.tunable.set_max_tuning_iterations(30)
torch.cuda.tunable.set_max_tuning_duration(5)

@torch.compile(fullgraph=True, dynamic=True)
def run(A, B):
    return torch.mm(A, B.t())
