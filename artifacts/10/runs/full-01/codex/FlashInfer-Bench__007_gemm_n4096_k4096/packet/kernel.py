import torch
import aiter


# AITER exposes hipBLASLt's individual, already-compiled solution kernels.
# Create its process-local handle once, outside the timed entry point.
aiter.hipb_create_extension()
aiter.rocb_create_extension()
from aiter.jit import module_hipbsolgemm as _hipb
from aiter.jit import module_rocsolgemm as _rocb


_SOLUTIONS = {
    256: 476893,
    248: 476893,
    240: 476893,
    232: 476893,
    224: 476893,
    216: 476893,
    208: 476893,
    200: 476893,
    192: 476894,
    184: 476894,
    176: 476894,
    168: 476894,
    160: 476894,
    152: 476894,
    144: 476894,
    136: 476894,
    112: 476981,
    104: 476893,
    96: 476894,
    88: 476894,
    80: 476894,
    48: 476984,
    32: 476997,
    24: 476999,
    15: 476762,
    8: 476762,
    7: 476762,
    4: 476762,
    2: 476762,
    1: 476762,
    35: 476895,
    972: 476643,
    2053: 476704,
    2379: 476751,
}

_ROC_SOLUTIONS = {
    128: 476984,
    120: 476982,
    40: 476985,
    16: 476763,
}


def run(A, B):
    M = A.shape[0]
    solution = _ROC_SOLUTIONS.get(M)
    if solution is not None:
        return _rocb.rocb_mm(A, B.T, solution)
    solution = _SOLUTIONS.get(M)
    if solution is None:
        return torch.matmul(A, B.T)
    return _hipb.hipb_mm(A, B.T, solution)
