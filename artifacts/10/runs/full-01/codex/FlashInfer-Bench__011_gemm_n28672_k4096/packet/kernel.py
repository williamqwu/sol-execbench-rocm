import importlib

import torch
import aiter


# AITER exposes tuned hipBLASLt solutions.  Initialize its handle at module
# load (outside timed calls), then use the underlying extension directly to
# avoid the decorated Python wrapper on the three shapes where it wins.
aiter.hipb_create_extension()
_hipb = importlib.import_module("aiter.jit.module_hipbsolgemm")
aiter.rocb_create_extension()
_rocb = importlib.import_module("aiter.jit.module_rocsolgemm")


_SOLUTIONS = {
    972: 477480,
    2053: 477480,
    2379: 476720,
}


_ROC_SOLUTIONS = {
    192: 476775,
    184: 476775,
    176: 476775,
    168: 476775,
    160: 476893,
    152: 476893,
    144: 476893,
    136: 476893,
    48: 476951,
    40: 476951,
    32: 476982,
    24: 476982,
    16: 476994,
    15: 476994,
    8: 476994,
    7: 476997,
    4: 476997,
    2: 476994,
    1: 476994,
}


def run(A, B):
    M = A.shape[0]
    solution = _ROC_SOLUTIONS.get(M)
    if solution is not None:
        return _rocb.rocb_mm(A, B.T, solution)
    solution = _SOLUTIONS.get(M)
    if solution is not None:
        return _hipb.hipb_mm(A, B.T, solution)
    return torch.matmul(A, B.T)
