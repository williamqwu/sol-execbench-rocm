import torch
import aiter


# Build the library handles once, outside the measured call.  The selected
# solution IDs below are all valid for fp16 A[M,14336] @ B[4096,14336].T on
# gfx950; no search or temporary weight transform is needed at run time.
aiter.hipb_create_extension()
aiter.rocb_create_extension()


_HIPBLASLT_SOLUTIONS = {
    256: 476634,
    248: 477183,
    240: 476949,
    232: 476696,
    224: 477128,
    216: 476894,
    208: 477187,
    200: 476901,
    192: 477413,
    184: 476756,
    176: 476756,
    168: 476756,
    160: 476756,
    152: 476756,
    48: 476895,
    35: 476595,
    32: 476594,
    24: 477452,
    16: 477464,
    15: 477464,
    8: 477464,
    7: 477464,
    4: 477461,
    2: 476997,
    1: 476997,
    972: 476751,
    2053: 476860,
    2379: 476749,
}

_ROCBLAS_SOLUTIONS = {
    8192: 476721,
}


def run(A, B):
    M = A.shape[0]
    solution = _HIPBLASLT_SOLUTIONS.get(M)
    if solution is not None:
        return aiter.hipb_mm(A, B.T, solution)
    solution = _ROCBLAS_SOLUTIONS.get(M)
    if solution is not None:
        return aiter.rocb_mm(A, B.T, solution)
    return torch.matmul(A, B.T)
