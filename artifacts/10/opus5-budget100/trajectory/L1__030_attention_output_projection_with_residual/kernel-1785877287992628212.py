import ctypes, torch

_lt = ctypes.CDLL("libhipblaslt.so", mode=ctypes.RTLD_GLOBAL)
HIP_R_32F, HIP_R_16BF, HIP_R_16F = 0, 14, 2

class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t)]

class _Heur(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]

def _chk(s, what=""):
    if s != 0: raise RuntimeError(f"hipblaslt {what} {s}")

_lt.hipblasLtCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lt.hipblasLtMatmulDescCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
_lt.hipblasLtMatmulDescSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lt.hipblasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64]
_lt.hipblasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lt.hipblasLtMatmulPreferenceSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lt.hipblasLtMatmulAlgoGetHeuristic.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int, ctypes.POINTER(_Heur), ctypes.POINTER(ctypes.c_int)]
_lt.hipblasLtMatmul.argtypes = [ctypes.c_void_p]*12 + [ctypes.POINTER(_Algo), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

_handle = ctypes.c_void_p()
_chk(_lt.hipblasLtCreate(ctypes.byref(_handle)))
_WS = 128 * 1024 * 1024
_wsbuf = torch.empty(_WS, dtype=torch.uint8, device="cuda")
_wsptr = ctypes.c_void_p(_wsbuf.data_ptr())
_alpha, _beta = ctypes.c_float(1.0), ctypes.c_float(1.0)
_cache = {}
_DT = {torch.bfloat16: HIP_R_16BF, torch.float16: HIP_R_16F}
_NCAND = 96
_raw_stream = torch._C._cuda_getCurrentRawStream
_FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")


def _build(m, n, k, dt):
    desc = ctypes.c_void_p()
    _chk(_lt.hipblasLtMatmulDescCreate(ctypes.byref(desc), 2, HIP_R_32F))
    opa, opb = ctypes.c_int32(112), ctypes.c_int32(111)
    _chk(_lt.hipblasLtMatmulDescSetAttribute(desc, 0, ctypes.byref(opa), 4))
    _chk(_lt.hipblasLtMatmulDescSetAttribute(desc, 1, ctypes.byref(opb), 4))
    L = [ctypes.c_void_p() for _ in range(4)]
    for h, (r, c, ld) in zip(L, [(k, n, k), (k, m, k), (n, m, n), (n, m, n)]):
        _chk(_lt.hipblasLtMatrixLayoutCreate(ctypes.byref(h), dt, r, c, ld))
    pref = ctypes.c_void_p()
    _chk(_lt.hipblasLtMatmulPreferenceCreate(ctypes.byref(pref)))
    ws = ctypes.c_uint64(_WS)
    _chk(_lt.hipblasLtMatmulPreferenceSetAttribute(pref, 1, ctypes.byref(ws), 8))
    res = (_Heur * _NCAND)()
    n_ = ctypes.c_int(0)
    _chk(_lt.hipblasLtMatmulAlgoGetHeuristic(_handle, desc, L[0], L[1], L[2], L[3], pref, _NCAND, res, ctypes.byref(n_)))
    if n_.value == 0: raise RuntimeError("no algo")
    return desc, L, res, n_.value


def _go(desc, L, algo, wp, xp, cp, dp):
    # _cuda_getCurrentRawStream is ~50x cheaper than current_stream().cuda_stream,
    # which matters when the GEMM itself is only ~20us.
    return _lt.hipblasLtMatmul(_handle, desc, ctypes.byref(_alpha), ctypes.c_void_p(wp), L[0],
        ctypes.c_void_p(xp), L[1], ctypes.byref(_beta), ctypes.c_void_p(cp), L[2],
        ctypes.c_void_p(dp), L[3], ctypes.byref(algo), _wsptr, _WS,
        ctypes.c_void_p(_raw_stream(0)))


def _tune(desc, L, res, nret, wp, xp, cp, dp):
    best, ba = None, None
    for i in range(nret):
        a = res[i].algo
        if _go(desc, L, a, wp, xp, cp, dp) != 0: continue
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(5)]
        for s, e in ev:
            _FLUSH.zero_(); s.record(); _go(desc, L, a, wp, xp, cp, dp); e.record()
        torch.cuda.synchronize()
        ts = sorted(s.elapsed_time(e) for s, e in ev)
        t = ts[len(ts) // 2]
        if best is None or t < best: best, ba = t, a
    if ba is None: raise RuntimeError("no working algo")
    return ba


@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    shape = attn_output.shape
    h = shape[-1]
    m = attn_output.numel() // h
    key = (m, h, attn_output.dtype)
    ent = _cache.get(key)
    d = torch.empty(shape, dtype=attn_output.dtype, device=attn_output.device)
    if ent is None:
        desc, L, res, nret = _build(m, h, h, _DT[attn_output.dtype])
        algo = _tune(desc, L, res, nret, o_proj_weight.data_ptr(),
                     attn_output.data_ptr(), residual.data_ptr(), d.data_ptr())
        ent = (desc, L, algo)
        _cache[key] = ent
    desc, L, algo = ent
    _chk(_go(desc, L, algo, o_proj_weight.data_ptr(), attn_output.data_ptr(),
             residual.data_ptr(), d.data_ptr()), "matmul")
    return d
