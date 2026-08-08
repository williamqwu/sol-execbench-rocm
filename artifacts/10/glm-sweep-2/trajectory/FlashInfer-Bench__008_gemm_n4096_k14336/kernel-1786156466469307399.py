import ctypes
import torch

# hipblaslt GEMM: C = A @ B.T for fp16 inputs on AMD MI350X.
# Beats rocBLAS (torch.matmul) for this problem's shapes via direct hipblaslt
# algo selection. The reference is torch.matmul(A, B.T) which goes through
# rocBLAS; hipblaslt exposes additional / better-tuned kernels for these N,K.

_lib = ctypes.CDLL("/opt/rocm-7.2.0/lib/libhipblaslt.so", mode=ctypes.RTLD_GLOBAL)

# --- types ---
hipblasLtHandle_t = ctypes.c_void_p
hipblasLtMatmulDesc_t = ctypes.c_void_p
hipblasLtMatrixLayout_t = ctypes.c_void_p
hipblasLtMatmulPreference_t = ctypes.c_void_p


class hipblasLtMatmulAlgo_t(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16),
                ("max_workspace_bytes", ctypes.c_size_t)]


class hipblasLtMatmulHeuristicResult_t(ctypes.Structure):
    _fields_ = [("algo", hipblasLtMatmulAlgo_t),
                ("workspaceSize", ctypes.c_size_t),
                ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float),
                ("reserved", ctypes.c_int * 4)]


# --- enum constants ---
HIP_R_32F = 0
HIP_R_16F = 2
HIPBLAS_OP_N = 111
HIPBLAS_OP_T = 112
HIPBLAS_COMPUTE_32F = 2
HIPBLASLT_MATMUL_DESC_TRANSA = 0
HIPBLASLT_MATMUL_DESC_TRANSB = 1
HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1

# --- signatures ---
_lib.hipblasLtCreate.argtypes = [ctypes.POINTER(hipblasLtHandle_t)]
_lib.hipblasLtCreate.restype = ctypes.c_int
_lib.hipblasLtMatmulDescCreate.argtypes = [ctypes.POINTER(hipblasLtMatmulDesc_t), ctypes.c_int, ctypes.c_int]
_lib.hipblasLtMatmulDescCreate.restype = ctypes.c_int
_lib.hipblasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(hipblasLtMatrixLayout_t), ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64]
_lib.hipblasLtMatrixLayoutCreate.restype = ctypes.c_int
_lib.hipblasLtMatmulDescSetAttribute.argtypes = [hipblasLtMatmulDesc_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.hipblasLtMatmulDescSetAttribute.restype = ctypes.c_int
_lib.hipblasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(hipblasLtMatmulPreference_t)]
_lib.hipblasLtMatmulPreferenceCreate.restype = ctypes.c_int
_lib.hipblasLtMatmulPreferenceSetAttribute.argtypes = [hipblasLtMatmulPreference_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.hipblasLtMatmulPreferenceSetAttribute.restype = ctypes.c_int
_lib.hipblasLtMatmulAlgoGetHeuristic.argtypes = [hipblasLtHandle_t, hipblasLtMatmulDesc_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatrixLayout_t, hipblasLtMatmulPreference_t, ctypes.c_int, ctypes.POINTER(hipblasLtMatmulHeuristicResult_t), ctypes.POINTER(ctypes.c_int)]
_lib.hipblasLtMatmulAlgoGetHeuristic.restype = ctypes.c_int
_lib.hipblasLtMatmul.argtypes = [hipblasLtHandle_t, hipblasLtMatmulDesc_t, ctypes.c_void_p, ctypes.c_void_p, hipblasLtMatrixLayout_t, ctypes.c_void_p, hipblasLtMatrixLayout_t, ctypes.c_void_p, ctypes.c_void_p, hipblasLtMatrixLayout_t, ctypes.c_void_p, hipblasLtMatrixLayout_t, ctypes.POINTER(hipblasLtMatmulAlgo_t), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_lib.hipblasLtMatmul.restype = ctypes.c_int


def _check(r, ctx=""):
    if r != 0:
        raise RuntimeError(f"hipblaslt error {r} at {ctx}")


# --- one-time init ---
_handle = hipblasLtHandle_t()
_check(_lib.hipblasLtCreate(ctypes.byref(_handle)), "Create")

_WORKSPACE_BYTES = 256 * 1024 * 1024
_workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.int8, device="cuda")

_alpha = ctypes.c_float(1.0)
_beta = ctypes.c_float(0.0)

# matmul desc: compute 32F, scale 32F, transA=T, transB=N
# C_cm[N,M] = B_cm^T @ A_cm  <=>  C[M,N] = A @ B.T  (row-major)
_desc = hipblasLtMatmulDesc_t()
_check(_lib.hipblasLtMatmulDescCreate(ctypes.byref(_desc), HIPBLAS_COMPUTE_32F, HIP_R_32F), "DescCreate")
_opA = ctypes.c_int32(HIPBLAS_OP_T)
_opB = ctypes.c_int32(HIPBLAS_OP_N)
_lib.hipblasLtMatmulDescSetAttribute(_desc, HIPBLASLT_MATMUL_DESC_TRANSA, ctypes.byref(_opA), ctypes.sizeof(_opA))
_lib.hipblasLtMatmulDescSetAttribute(_desc, HIPBLASLT_MATMUL_DESC_TRANSB, ctypes.byref(_opB), ctypes.sizeof(_opB))

_pref = hipblasLtMatmulPreference_t()
_check(_lib.hipblasLtMatmulPreferenceCreate(ctypes.byref(_pref)), "PrefCreate")
_ws_attr = ctypes.c_uint64(_WORKSPACE_BYTES)
_lib.hipblasLtMatmulPreferenceSetAttribute(_pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ctypes.byref(_ws_attr), ctypes.sizeof(_ws_attr))

_N = 4096
_K = 14336

# Cache: M -> (algo, workspaceSize, Adesc, Bdesc, Ddesc)
_cache = {}


def _build_layouts(M):
    # col-major: A-desc for B matrix [K,N], B-desc for A matrix [K,M], D-desc for C [N,M]
    Adesc = hipblasLtMatrixLayout_t()
    _lib.hipblasLtMatrixLayoutCreate(ctypes.byref(Adesc), HIP_R_16F, _K, _N, _K)
    Bdesc = hipblasLtMatrixLayout_t()
    _lib.hipblasLtMatrixLayoutCreate(ctypes.byref(Bdesc), HIP_R_16F, _K, M, _K)
    Ddesc = hipblasLtMatrixLayout_t()
    _lib.hipblasLtMatrixLayoutCreate(ctypes.byref(Ddesc), HIP_R_16F, _N, M, _N)
    return Adesc, Bdesc, Ddesc


def _bench_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _select_algo(M, Adesc, Bdesc, Ddesc, A, B):
    """Search heuristic algos + torch.matmul, return the fastest valid one.

    Returns (algo, wss) for hipblaslt, or ('torch', 0) to use torch.matmul.
    """
    stream = torch.cuda.current_stream().cuda_stream
    D = torch.empty(M, _N, dtype=torch.float16, device="cuda")
    ref = torch.matmul(A, B.T)

    def make_hb_fn(algo, wss):
        def f():
            _lib.hipblasLtMatmul(_handle, _desc, ctypes.byref(_alpha),
                                 B.data_ptr(), Adesc, A.data_ptr(), Bdesc,
                                 ctypes.byref(_beta), D.data_ptr(), Ddesc,
                                 D.data_ptr(), Ddesc, ctypes.byref(algo),
                                 _workspace.data_ptr(), wss, stream)
        return f

    def torch_fn():
        torch.matmul(A, B.T, out=D)

    # Benchmark torch.matmul as the baseline to beat.
    best_t = _bench_fn(torch_fn)
    best = ("torch", 0)

    MAX = 50
    results = (hipblasLtMatmulHeuristicResult_t * MAX)()
    rc = ctypes.c_int(0)
    _lib.hipblasLtMatmulAlgoGetHeuristic(_handle, _desc, Adesc, Bdesc, Ddesc, Ddesc, _pref, MAX, results, ctypes.byref(rc))
    n = rc.value
    n_try = min(n, 20)
    for i in range(n_try):
        if results[i].state != 0:
            continue
        algo = results[i].algo
        wss = results[i].workspaceSize
        if wss > _WORKSPACE_BYTES:
            continue
        try:
            fn = make_hb_fn(algo, wss)
            fn()
            torch.cuda.synchronize()
            err = (D.float() - ref.float()).abs().max().item()
            if err > 2.0:
                continue
            t = _bench_fn(fn)
            if t < best_t:
                best_t = t
                best = (algo, wss)
        except Exception:
            continue
    return best


def run(A, B):
    M = A.shape[0]
    entry = _cache.get(M)
    if entry is None:
        Adesc, Bdesc, Ddesc = _build_layouts(M)
        algo, wss = _select_algo(M, Adesc, Bdesc, Ddesc, A, B)
        if algo == "torch":
            _cache[M] = ("torch", 0, None, None, None)
        else:
            _cache[M] = (algo, wss, Adesc, Bdesc, Ddesc)
        entry = _cache[M]
    if entry[0] == "torch":
        return torch.matmul(A, B.T)
    algo, wss, Adesc, Bdesc, Ddesc = entry
    C = torch.empty(M, _N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream().cuda_stream
    r = _lib.hipblasLtMatmul(_handle, _desc, ctypes.byref(_alpha),
                             B.data_ptr(), Adesc, A.data_ptr(), Bdesc,
                             ctypes.byref(_beta), C.data_ptr(), Ddesc,
                             C.data_ptr(), Ddesc, ctypes.byref(algo),
                             _workspace.data_ptr(), wss, stream)
    if r != 0:
        return torch.matmul(A, B.T)
    return C
