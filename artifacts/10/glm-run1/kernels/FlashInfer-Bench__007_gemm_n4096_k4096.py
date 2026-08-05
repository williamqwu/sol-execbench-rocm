import ctypes
import torch
from ctypes import (
    c_void_p, c_int, c_int32, c_int64, c_uint64, c_size_t,
    c_uint8, c_float, POINTER, byref,
)

# Direct hipBLASLt GEMM via ctypes. Computes C = A @ B.T for fp16 inputs with
# fp32 accumulation, matching the reference.
#
# For each unique (M,N,K) shape the first call times several heuristic
# hipBLASLt algorithms AND torch.matmul, all under cold-cache conditions that
# match the harness (the 256 MiB Infinity Cache is flushed before each timed
# iteration). The fastest method is cached and used for all later calls. This
# never regresses below torch.matmul because that is one of the candidates.

_lib = ctypes.CDLL("/opt/rocm/lib/libhipblaslt.so.1")

_OP_N = 111
_OP_T = 112
_R_32F = 0
_R_16F = 2
_COMPUTE_32F = 2
_DESC_TRANSA = 0
_DESC_TRANSB = 1
_PREF_MAX_WORKSPACE_BYTES = 1


class _Algo(ctypes.Structure):
    _fields_ = [("data", c_uint8 * 16), ("max_workspace_bytes", c_size_t)]


class _HeurResult(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("workspaceSize", c_size_t),
        ("state", c_int),
        ("wavesCount", c_float),
        ("reserved", c_int * 4),
    ]


_t = c_void_p

_lib.hipblasLtCreate.argtypes = [POINTER(_t)]
_lib.hipblasLtCreate.restype = c_int
_lib.hipblasLtMatmulDescCreate.argtypes = [POINTER(_t), c_int, c_int]
_lib.hipblasLtMatmulDescCreate.restype = c_int
_lib.hipblasLtMatmulDescSetAttribute.argtypes = [_t, c_int, c_void_p, c_size_t]
_lib.hipblasLtMatmulDescSetAttribute.restype = c_int
_lib.hipblasLtMatrixLayoutCreate.argtypes = [POINTER(_t), c_int, c_uint64, c_uint64, c_int64]
_lib.hipblasLtMatrixLayoutCreate.restype = c_int
_lib.hipblasLtMatmulPreferenceCreate.argtypes = [POINTER(_t)]
_lib.hipblasLtMatmulPreferenceCreate.restype = c_int
_lib.hipblasLtMatmulPreferenceSetAttribute.argtypes = [_t, c_int, c_void_p, c_size_t]
_lib.hipblasLtMatmulPreferenceSetAttribute.restype = c_int
_lib.hipblasLtMatmulPreferenceDestroy.argtypes = [_t]
_lib.hipblasLtMatmulPreferenceDestroy.restype = c_int
_lib.hipblasLtMatmulAlgoGetHeuristic.argtypes = [
    _t, _t, _t, _t, _t, _t, _t, c_int, POINTER(_HeurResult), POINTER(c_int)
]
_lib.hipblasLtMatmulAlgoGetHeuristic.restype = c_int
_lib.hipblasLtMatmul.argtypes = [
    _t, _t, c_void_p, c_void_p, _t, c_void_p, _t, c_void_p,
    c_void_p, _t, c_void_p, _t, POINTER(_Algo), c_void_p, c_size_t, c_void_p,
]
_lib.hipblasLtMatmul.restype = c_int

_handle = _t()
_lib.hipblasLtCreate(byref(_handle))

_WS_BYTES = 128 * 1024 * 1024
_workspace = torch.empty(_WS_BYTES // 4, dtype=torch.int32, device="cuda")
_ws_ptr = _workspace.data_ptr()
_alpha = c_float(1.0)
_beta = c_float(0.0)

# Cache-flush buffer: 512 MiB (2x the 256 MiB Infinity Cache) to evict MALL.
_FLUSH = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

_plans = {}


def _build_layouts(M, N, K):
    # C[M,N] = A[M,K] @ B[N,K]^T (row-major). Col-major identity:
    #   D[N,M] = op(B_cm) @ op(A_cm), D aliases C memory.
    #   B_row[N,K] == col-major [K,N] ld=K, op=T -> [N,K]
    #   A_row[M,K] == col-major [K,M] ld=K, op=N -> [K,M]
    desc = _t()
    _lib.hipblasLtMatmulDescCreate(byref(desc), _COMPUTE_32F, _R_32F)
    opA = c_int32(_OP_T)
    opB = c_int32(_OP_N)
    _lib.hipblasLtMatmulDescSetAttribute(desc, _DESC_TRANSA, byref(opA), ctypes.sizeof(opA))
    _lib.hipblasLtMatmulDescSetAttribute(desc, _DESC_TRANSB, byref(opB), ctypes.sizeof(opB))
    adesc = _t()
    _lib.hipblasLtMatrixLayoutCreate(byref(adesc), _R_16F, c_uint64(K), c_uint64(N), c_int64(K))
    bdesc = _t()
    _lib.hipblasLtMatrixLayoutCreate(byref(bdesc), _R_16F, c_uint64(K), c_uint64(M), c_int64(K))
    cdesc = _t()
    _lib.hipblasLtMatrixLayoutCreate(byref(cdesc), _R_16F, c_uint64(N), c_uint64(M), c_int64(N))
    return desc, adesc, bdesc, cdesc


def _get_algos(desc, adesc, bdesc, cdesc, n=200):
    pref = _t()
    _lib.hipblasLtMatmulPreferenceCreate(byref(pref))
    ws = c_uint64(_WS_BYTES)
    _lib.hipblasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WORKSPACE_BYTES, byref(ws), ctypes.sizeof(ws))
    results = (_HeurResult * n)()
    retC = c_int(0)
    _lib.hipblasLtMatmulAlgoGetHeuristic(_handle, desc, adesc, bdesc, cdesc, cdesc, pref, n, results, byref(retC))
    _lib.hipblasLtMatmulPreferenceDestroy(pref)
    return [results[i].algo for i in range(retC.value)]


def _hbl_call(desc, adesc, bdesc, cdesc, algo, A, B):
    M, N = A.shape[0], B.shape[0]
    C = torch.empty(M, N, dtype=torch.float16, device=A.device)
    algo_ptr = byref(algo) if algo is not None else None
    stream = c_void_p(torch.cuda.current_stream().cuda_stream)
    _lib.hipblasLtMatmul(
        _handle, desc, byref(_alpha), c_void_p(B.data_ptr()), adesc,
        c_void_p(A.data_ptr()), bdesc, byref(_beta),
        c_void_p(C.data_ptr()), cdesc, c_void_p(C.data_ptr()), cdesc,
        algo_ptr, c_void_p(_ws_ptr), c_size_t(_WS_BYTES), stream,
    )
    return C


def _time_cold(fn, A, B, iters=8):
    # Time fn under cold-cache conditions matching the harness: flush the
    # Infinity Cache before each timed iteration.
    for _ in range(3):
        _FLUSH.zero_()
        fn(A, B)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        _FLUSH.zero_()
        starts[i].record()
        fn(A, B)
        ends[i].record()
    torch.cuda.synchronize()
    return min(s.elapsed_time(e) for s, e in zip(starts, ends))


def _torch_mm(A, B):
    return torch.matmul(A, B.T)


def run(A, B):
    M, N, K = A.shape[0], B.shape[0], B.shape[1]
    key = (M, N, K)
    plan = _plans.get(key)
    if plan is None:
        # Baseline: torch.matmul (the reference). Always a candidate so we
        # never regress below it.
        best_fn = _torch_mm
        best_t = _time_cold(_torch_mm, A, B)

        desc, adesc, bdesc, cdesc = _build_layouts(M, N, K)
        algos = _get_algos(desc, adesc, bdesc, cdesc)
        n_try = min(12, len(algos))
        for i in range(n_try):
            algo = algos[i]
            fn = lambda A, B, a=algo, d=desc, ad=adesc, bd=bdesc, cd=cdesc: _hbl_call(d, ad, bd, cd, a, A, B)
            t = _time_cold(fn, A, B)
            if t < best_t:
                best_t = t
                best_fn = fn
        plan = best_fn
        _plans[key] = plan
    return plan(A, B)
