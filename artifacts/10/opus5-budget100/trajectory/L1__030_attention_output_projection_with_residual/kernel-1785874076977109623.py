import ctypes, torch

_lt = ctypes.CDLL("libhipblaslt.so", mode=ctypes.RTLD_GLOBAL)

HIPBLAS_OP_N, HIPBLAS_OP_T = 111, 112
HIP_R_32F, HIP_R_16BF, HIP_R_16F = 0, 14, 2
HIPBLAS_COMPUTE_32F = 2
DESC_TRANSA, DESC_TRANSB = 0, 1
PREF_MAX_WS = 1

class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t)]

class _Heur(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t),
                ("state", ctypes.c_int), ("wavesCount", ctypes.c_float),
                ("reserved", ctypes.c_int * 4)]

def _chk(s, what=""):
    if s != 0:
        raise RuntimeError(f"hipblaslt {what} status {s}")

_lt.hipblasLtCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lt.hipblasLtMatmulDescCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
_lt.hipblasLtMatmulDescSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lt.hipblasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64]
_lt.hipblasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lt.hipblasLtMatmulPreferenceSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lt.hipblasLtMatmulAlgoGetHeuristic.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(_Heur), ctypes.POINTER(ctypes.c_int)]
_lt.hipblasLtMatmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(_Algo), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

_handle = ctypes.c_void_p()
_chk(_lt.hipblasLtCreate(ctypes.byref(_handle)), "create")

_WS_BYTES = 64 * 1024 * 1024
_ws = torch.empty(_WS_BYTES, dtype=torch.uint8, device="cuda")
_ws_ptr = ctypes.c_void_p(_ws.data_ptr())

_alpha = ctypes.c_float(1.0)
_beta = ctypes.c_float(1.0)
_cache = {}

_DT = {torch.bfloat16: HIP_R_16BF, torch.float16: HIP_R_16F}

def _make_plan(m, n, k, dt):
    desc = ctypes.c_void_p()
    _chk(_lt.hipblasLtMatmulDescCreate(ctypes.byref(desc), HIPBLAS_COMPUTE_32F, HIP_R_32F), "desc")
    opa = ctypes.c_int32(HIPBLAS_OP_T)
    opb = ctypes.c_int32(HIPBLAS_OP_N)
    _chk(_lt.hipblasLtMatmulDescSetAttribute(desc, DESC_TRANSA, ctypes.byref(opa), 4))
    _chk(_lt.hipblasLtMatmulDescSetAttribute(desc, DESC_TRANSB, ctypes.byref(opb), 4))
    la, lb, lc, ld = (ctypes.c_void_p() for _ in range(4))
    _chk(_lt.hipblasLtMatrixLayoutCreate(ctypes.byref(la), dt, k, n, k))
    _chk(_lt.hipblasLtMatrixLayoutCreate(ctypes.byref(lb), dt, k, m, k))
    _chk(_lt.hipblasLtMatrixLayoutCreate(ctypes.byref(lc), dt, n, m, n))
    _chk(_lt.hipblasLtMatrixLayoutCreate(ctypes.byref(ld), dt, n, m, n))
    pref = ctypes.c_void_p()
    _chk(_lt.hipblasLtMatmulPreferenceCreate(ctypes.byref(pref)))
    ws = ctypes.c_uint64(_WS_BYTES)
    _chk(_lt.hipblasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WS, ctypes.byref(ws), 8))
    N = 1
    res = (_Heur * N)()
    nret = ctypes.c_int(0)
    _chk(_lt.hipblasLtMatmulAlgoGetHeuristic(_handle, desc, la, lb, lc, ld, pref, N, res, ctypes.byref(nret)), "heur")
    if nret.value == 0:
        raise RuntimeError("no algo")
    algo = res[0].algo
    return (desc, la, lb, lc, ld, algo)


@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    b, s, h = attn_output.shape
    x = attn_output.view(-1, h)
    c = residual.view(-1, h)
    m, k = x.shape
    n = o_proj_weight.shape[0]
    key = (m, n, k, x.dtype)
    plan = _cache.get(key)
    if plan is None:
        plan = _make_plan(m, n, k, _DT[x.dtype])
        _cache[key] = plan
    desc, la, lb, lc, ld, algo = plan
    d = torch.empty((m, n), dtype=x.dtype, device=x.device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _chk(_lt.hipblasLtMatmul(_handle, desc,
        ctypes.byref(_alpha), ctypes.c_void_p(o_proj_weight.data_ptr()), la,
        ctypes.c_void_p(x.data_ptr()), lb,
        ctypes.byref(_beta), ctypes.c_void_p(c.data_ptr()), lc,
        ctypes.c_void_p(d.data_ptr()), ld,
        ctypes.byref(algo), _ws_ptr, _WS_BYTES, stream), "matmul")
    return d.view(b, s, h)
