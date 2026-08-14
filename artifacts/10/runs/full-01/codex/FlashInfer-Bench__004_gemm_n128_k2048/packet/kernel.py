import torch
def run(A, B):
    # beta=0 makes the broadcast input semantically irrelevant while taking
    # the addmm/hipBLASLt dispatch path.
    seed = torch.empty((1, 1), dtype=A.dtype, device=A.device)
    return torch.addmm(seed, A, B.T, beta=0.0)
