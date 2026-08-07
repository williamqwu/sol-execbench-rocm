import torch
import aiter


# Load AITER's selectable rocBLAS solution dispatcher once, outside the timed
# entry point.  It lets us select the best native Tensile kernel for this fixed
# N/K pair instead of relying on the generic library heuristic.
aiter.rocb_create_extension()
aiter.hipb_create_extension()


def _rocblas(A, BT, solution):
    return torch.ops.aiter.rocb_mm(A, BT, solution)


def _hipblaslt(A, BT, solution):
    return torch.ops.aiter.hipb_mm(A, BT, solution)


# Native solution selected by cold-cache timing for each benchmark shape.  A
# missing entry deliberately falls back to PyTorch's normal heuristic.
_SOLUTIONS = {
    256: 477430,
    240: 477104,
    224: 476878,
    216: 477361,
    208: 477104,
    200: 476878,
    176: 476652,
    168: 476648,
    160: 477097,
    152: 476878,
    144: 476714,
    136: 477097,
    128: 476704,
    120: 477119,
    112: 476692,
    104: 476692,
    96: 476589,
    88: 476746,
    80: 476695,
    72: 476695,
    70: 476746,
    64: 476695,
    56: 476695,
    48: 476695,
    40: 476982,
    35: 476984,
    32: 476982,
    24: 476982,
    16: 477218,
    15: 476998,
    8: 477454,
    7: 476998,
    4: 476993,
    2: 476998,
    1: 476998,
    972: 476631,
    2053: 476753,
    8192: 477481,
}


_HIP_SOLUTIONS = {
    48: 476963,
    56: 476979,
    64: 476963,
    70: 476962,
    72: 476962,
    80: 476758,
    88: 476957,
    96: 476956,
    104: 476956,
    112: 476963,
    120: 476951,
}


def run(A, B):
    m = A.shape[0]
    BT = B.T
    hip_solution = _HIP_SOLUTIONS.get(m)
    if hip_solution is not None:
        return _hipblaslt(A, BT, hip_solution)
    solution = _SOLUTIONS.get(m)
    if solution is None:
        return torch.mm(A, BT)
    return _rocblas(A, BT, solution)
