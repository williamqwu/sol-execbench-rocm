import torch


_outputs = {}


def run(A, B):
    key = (A.shape[0], A.device)
    out = _outputs.get(key)
    if out is None:
        out = torch.empty((A.shape[0], B.shape[0]), dtype=A.dtype, device=A.device)
        _outputs[key] = out
    return torch.mm(A, B.T, out=out)
