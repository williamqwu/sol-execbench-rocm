import ctypes, torch

_lt = ctypes.CDLL("libhipblaslt.so", mode=ctypes.RTLD_GLOBAL)
HIP_R_32F, HIP_R_16BF, HIP_R_16F = 0, 14, 2

class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t)]

class _Heur(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]

def _chk(s, what=""):
    if s != 0:
        raise RuntimeError(f"hipblaslt {what} {s}")

_lt.hipblasLtCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lt.hipblasLtMatmulDescCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
_lt.hipblasLtMatmulDescSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lt.hipblasLtMatrixLayoutCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64]
_lt.hipblasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lt.hipblasLtMatmulPreferenceSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lt.hipblasLtMatmulAlgoGetHeuristic.argtypes = [ctypes.c_void_p] * 7 + [
    ctypes.c_int, ctypes.POINTER(_Heur), ctypes.POINTER(ctypes.c_int)]
_lt.hipblasLtMatmul.argtypes = [ctypes.c_void_p] * 12 + [
    ctypes.POINTER(_Algo), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

_handle = ctypes.c_void_p()
_chk(_lt.hipblasLtCreate(ctypes.byref(_handle)))

_WS = 128 * 1024 * 1024
_wsbuf = torch.empty(_WS, dtype=torch.uint8, device="cuda")
_wsptr = ctypes.c_void_p(_wsbuf.data_ptr())
_alpha, _beta = ctypes.c_float(1.0), ctypes.c_float(1.0)
_cache = {}
_DT = {torch.bfloat16: HIP_R_16BF, torch.float16: HIP_R_16F}
_raw_stream = torch._C._cuda_getCurrentRawStream

# The heuristic's own ordering is not a ranking: for these shapes the fastest
# solution sits ~900 entries deep, so ask for the whole library and time it.
_NCAND = 4096
# The harness evicts the last-level cache before every timed call, so tune
# against a cold cache too -- warm-cache ranking picks a different winner.
_FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
_SCREEN, _SEMI, _FINAL = 64, 20, 41


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
    _chk(_lt.hipblasLtMatmulAlgoGetHeuristic(_handle, desc, L[0], L[1], L[2], L[3],
                                             pref, _NCAND, res, ctypes.byref(n_)))
    if n_.value == 0:
        raise RuntimeError("no hipblaslt algo")
    return desc, L, res, n_.value


def _go(desc, L, algo, wp, xp, cp, dp):
    return _lt.hipblasLtMatmul(_handle, desc,
        ctypes.byref(_alpha), ctypes.c_void_p(wp), L[0],
        ctypes.c_void_p(xp), L[1],
        ctypes.byref(_beta), ctypes.c_void_p(cp), L[2],
        ctypes.c_void_p(dp), L[3], ctypes.byref(algo), _wsptr, _WS,
        ctypes.c_void_p(_raw_stream(0)))


def _rank(desc, L, res, idxs, ptrs, reps, flush):
    """Median flushed launch time per candidate, interleaved across rounds."""
    wp, xp, cp, dp = ptrs
    times = {i: [] for i in idxs}
    for _ in range(reps):
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in idxs]
        for (s, e), i in zip(ev, idxs):
            if flush:
                _FLUSH.zero_()
            s.record()
            _go(desc, L, res[i].algo, wp, xp, cp, dp)
            e.record()
        torch.cuda.synchronize()
        for (s, e), i in zip(ev, idxs):
            times[i].append(s.elapsed_time(e))
    return sorted(idxs, key=lambda i: sorted(times[i])[reps // 2])


def _tune(desc, L, res, nret, ptrs):
    wp, xp, cp, dp = ptrs
    live = [i for i in range(nret)
            if _go(desc, L, res[i].algo, wp, xp, cp, dp) == 0]
    torch.cuda.synchronize()
    if not live:
        raise RuntimeError("no working hipblaslt algo")
    # Warm-cache screen first: one pass over ~2000 candidates with a 256 MiB
    # flush each would cost seconds, and it only has to be good enough to keep
    # the real contenders.
    live = _rank(desc, L, res, live, ptrs, 2, False)[:_SCREEN]
    live = _rank(desc, L, res, live, ptrs, 3, True)[:_SEMI]
    return res[_rank(desc, L, res, live, ptrs, _FINAL, True)[0]].algo


@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    shape = attn_output.shape
    h = shape[-1]
    m = attn_output.numel() // h
    d = torch.empty(shape, dtype=attn_output.dtype, device=attn_output.device)
    ptrs = (o_proj_weight.data_ptr(), attn_output.data_ptr(),
            residual.data_ptr(), d.data_ptr())
    key = (m, h, attn_output.dtype)
    ent = _cache.get(key)
    if ent is None:
        desc, L, res, nret = _build(m, h, h, _DT[attn_output.dtype])
        ent = (desc, L, _tune(desc, L, res, nret, ptrs))
        _cache[key] = ent
    desc, L, algo = ent
    _chk(_go(desc, L, algo, *ptrs), "matmul")
    return d
