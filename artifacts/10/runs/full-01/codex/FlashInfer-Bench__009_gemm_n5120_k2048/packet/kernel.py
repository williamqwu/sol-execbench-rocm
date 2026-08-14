import torch
import aiter


# hipBLASLt exposes many valid implementations for this fixed N/K pair.  The
# fastest implementation changes with M, so select the measured winner for
# every declared workload.  These are genuine GEMM solution IDs; the fallback
# preserves the operation for any additional shape.
_SOLUTIONS = {
    1: 477218,
    2: 477226,
    4: 476997,
    5: 477440,
    6: 477226,
    8: 476992,
    16: 477470,
    17: 477218,
    25: 476732,
    32: 476598,
    34: 477222,
    63: 476613,
    64: 476613,
    93: 477215,
    128: 476616,
    172: 476963,
    289: 476950,
    492: 476633,
    952: 477181,
    8828: 477480,
    11006: 476720,
    12251: 477480,
    12853: 477480,
    14915: 477480,
    16294: 477479,
}


# Initialize the library handle while the solution module is loaded, outside
# the timed entry point.
aiter.hipb_create_extension()
_hipb_mm = torch.ops.aiter.hipb_mm.default


def run(A, B):
    M = A.shape[0]
    # For the latency-bound skinny cases PyTorch's in-process hipBLASLt path
    # has less dispatch overhead than AITER's selectable-solution interface.
    if M < 172:
        return torch.matmul(A, B.T)
    solution = _SOLUTIONS.get(M)
    if solution is None:
        return torch.mm(A, B.T)
    return _hipb_mm(A, B.T, solution)
