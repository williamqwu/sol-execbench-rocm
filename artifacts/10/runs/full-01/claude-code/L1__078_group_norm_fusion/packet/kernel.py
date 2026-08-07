import torch
import triton
import triton.language as tl

NG: tl.constexpr = tl.constexpr(32)


# ---------------------------------------------------------------------------
# Path 1: one program per (batch, group).  Two reduction passes then a write
# pass, the later passes hitting cache.  Mean/var accumulated in float32
# exactly as the reference does: mean first, then sum of squared deviations.
# ---------------------------------------------------------------------------
@triton.jit
def _gn_one(X, Wp, Bp, Y, eps,
            GS: tl.constexpr, HW: tl.constexpr, CPG: tl.constexpr,
            BLOCK: tl.constexpr, NITER: tl.constexpr):
    g = tl.program_id(0)
    base = g * GS
    off = tl.arange(0, BLOCK)

    acc = tl.zeros([BLOCK], tl.float32)
    for i in tl.static_range(NITER):
        o = i * BLOCK + off
        acc += tl.load(X + base + o, mask=o < GS, other=0.0)
    mean = tl.sum(acc, 0) / GS

    acc2 = tl.zeros([BLOCK], tl.float32)
    for i in tl.static_range(NITER):
        o = i * BLOCK + off
        m = o < GS
        x = tl.load(X + base + o, mask=m, other=0.0)
        d = tl.where(m, x - mean, 0.0)
        acc2 += d * d
    var = tl.sum(acc2, 0) / GS
    rs = tl.sqrt(var + eps)

    cb = (g % NG) * CPG
    for i in tl.static_range(NITER):
        o = i * BLOCK + off
        m = o < GS
        x = tl.load(X + base + o, mask=m, other=0.0)
        ch = cb + o // HW
        w = tl.load(Wp + ch, mask=m, other=0.0)
        b = tl.load(Bp + ch, mask=m, other=0.0)
        tl.store(Y + base + o, (x - mean) / rs * w + b, mask=m)


# ---------------------------------------------------------------------------
# Path 2: split reduction, for groups that are large but too few to fill the
# machine.  Chunk partials combine with the parallel (Chan) variance formula,
# which is at least as accurate as the reference's reduction.
# ---------------------------------------------------------------------------
@triton.jit
def _gn_stats(X, Sp, Qp, Np,
              GS: tl.constexpr, CHUNK: tl.constexpr, NSPLIT: tl.constexpr,
              BLOCK: tl.constexpr, NITER: tl.constexpr):
    g = tl.program_id(0)
    s = tl.program_id(1)
    start = s * CHUNK
    n = tl.minimum(CHUNK, GS - start)
    base = g * GS + start
    off = tl.arange(0, BLOCK)

    acc = tl.zeros([BLOCK], tl.float32)
    for i in tl.static_range(NITER):
        o = i * BLOCK + off
        acc += tl.load(X + base + o, mask=o < n, other=0.0)
    nf = tl.maximum(n, 1).to(tl.float32)
    mean = tl.sum(acc, 0) / nf

    acc2 = tl.zeros([BLOCK], tl.float32)
    for i in tl.static_range(NITER):
        o = i * BLOCK + off
        m = o < n
        x = tl.load(X + base + o, mask=m, other=0.0)
        d = tl.where(m, x - mean, 0.0)
        acc2 += d * d

    tl.store(Sp + g * NSPLIT + s, mean)
    tl.store(Qp + g * NSPLIT + s, tl.sum(acc2, 0))
    tl.store(Np + g * NSPLIT + s, nf)


@triton.jit
def _gn_norm(X, Wp, Bp, Y, Sp, Qp, Np, eps,
             GS: tl.constexpr, HW: tl.constexpr, CPG: tl.constexpr,
             NSPLIT: tl.constexpr, BNS: tl.constexpr,
             BLOCK: tl.constexpr, NITER: tl.constexpr):
    g = tl.program_id(0)
    j = tl.program_id(1)

    so = tl.arange(0, BNS)
    sm = so < NSPLIT
    mi = tl.load(Sp + g * NSPLIT + so, mask=sm, other=0.0)
    m2 = tl.load(Qp + g * NSPLIT + so, mask=sm, other=0.0)
    ni = tl.load(Np + g * NSPLIT + so, mask=sm, other=0.0)
    mean = tl.sum(tl.where(sm, ni * mi, 0.0), 0) / GS
    dm = tl.where(sm, mi - mean, 0.0)
    var = (tl.sum(m2, 0) + tl.sum(ni * dm * dm, 0)) / GS
    rs = tl.sqrt(var + eps)

    cb = (g % NG) * CPG
    off = tl.arange(0, BLOCK)
    for i in tl.static_range(NITER):
        o = (j * NITER + i) * BLOCK + off
        m = o < GS
        x = tl.load(X + g * GS + o, mask=m, other=0.0)
        ch = cb + o // HW
        w = tl.load(Wp + ch, mask=m, other=0.0)
        b = tl.load(Bp + ch, mask=m, other=0.0)
        tl.store(Y + g * GS + o, (x - mean) / rs * w + b, mask=m)


def _p2(n):
    return 1 << (n - 1).bit_length()


_CFG = {}
_CUS = 256


def _plan(B, C, H, W):
    key = (B, C, H, W)
    p = _CFG.get(key)
    if p is not None:
        return p
    G = B * 32
    CPG = C // 32
    HW = H * W
    GS = CPG * HW

    if G >= 2 * _CUS or GS <= 49152:
        blk = min(4096, _p2(GS))
        nit = -(-GS // blk)
        nw = 4 if nit >= 4 or blk <= 1024 else 8
        p = ("one", dict(G=G, GS=GS, HW=HW, CPG=CPG, BLOCK=blk, NITER=nit,
                         nw=nw))
    else:
        ns = max(1, min(64, -(-(4 * _CUS) // G)))
        ns = min(ns, max(1, GS // 4096))
        chunk = -(-GS // ns)
        b1 = min(2048, _p2(chunk))
        it1 = -(-chunk // b1)
        bn = min(2048, _p2(GS))
        nblk = -(-GS // bn)
        it2 = max(1, nblk // max(1, (2 * _CUS) // G))
        nj = -(-nblk // it2)
        p = ("split", dict(G=G, GS=GS, HW=HW, CPG=CPG, NSPLIT=ns, CHUNK=chunk,
                           B1=b1, IT1=it1, BNS=_p2(ns), BN=bn, NJ=nj, IT2=it2))
    _CFG[key] = p
    return p


_BUF = {}


def _scratch(n, dev):
    b = _BUF.get(dev)
    if b is None or b.numel() < n:
        b = torch.empty(max(n, 8192), dtype=torch.float32, device=dev)
        _BUF[dev] = b
    return b


from triton.runtime import driver as _driver

_DEVC = []


def _DEV():
    if not _DEVC:
        _DEVC.append(_driver.active.get_current_device())
    return _DEVC[0]


def _stream(d):
    return _driver.active.get_current_stream(d)


# ---------------------------------------------------------------------------
# Pre-compiled direct launch.  The generic Triton JIT dispatch path costs
# ~16 us of Python per launch, which dominates every workload here (the GPU
# work itself is 3-30 us).  We compile once per shape and then call the
# compiled kernel's launcher directly, which costs ~4 us.
# ---------------------------------------------------------------------------
_LAUNCH = {}


def _compile(jitfn, example_args, consts, nw, ns):
    ck = jitfn.warmup(*example_args, grid=(1,), num_warps=nw, num_stages=ns,
                      **consts)
    ck._init_handles()
    return ck


def _build(x, weight, bias, y, eps, key):
    B, C, H, W = x.shape
    mode, p = _plan(B, C, H, W)
    if mode == "one":
        consts = dict(GS=p["GS"], HW=p["HW"], CPG=p["CPG"],
                      BLOCK=p["BLOCK"], NITER=p["NITER"])
        ck = _compile(_gn_one, (x, weight, bias, y, eps), consts, p["nw"], 1)
        tail = (p["GS"], p["HW"], p["CPG"], p["BLOCK"], p["NITER"])
        ent = ("one", ck.run, ck.function, ck.packed_metadata, p["G"], tail)
    else:
        G, ns = p["G"], p["NSPLIT"]
        n = G * ns
        buf = _scratch(3 * n, x.device)
        sp, qp, np_ = buf[0:n], buf[n:2 * n], buf[2 * n:3 * n]
        c1 = dict(GS=p["GS"], CHUNK=p["CHUNK"], NSPLIT=ns,
                  BLOCK=p["B1"], NITER=p["IT1"])
        k1 = _compile(_gn_stats, (x, sp, qp, np_), c1, 4, 2)
        t1 = (p["GS"], p["CHUNK"], ns, p["B1"], p["IT1"])
        c2 = dict(GS=p["GS"], HW=p["HW"], CPG=p["CPG"], NSPLIT=ns,
                  BNS=p["BNS"], BLOCK=p["BN"], NITER=p["IT2"])
        k2 = _compile(_gn_norm, (x, weight, bias, y, sp, qp, np_, eps), c2, 8, 1)
        t2 = (p["GS"], p["HW"], p["CPG"], ns, p["BNS"], p["BN"], p["IT2"])
        ent = ("split",
               k1.run, k1.function, k1.packed_metadata, G, ns, t1,
               k2.run, k2.function, k2.packed_metadata, p["NJ"], t2,
               sp, qp, np_)
    _LAUNCH[key] = ent
    return ent


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
        eps: float) -> torch.Tensor:
    shp = x.shape
    key = (shp[0], shp[1], shp[2], shp[3], eps)
    ent = _LAUNCH.get(key)
    y = torch.empty_like(x)
    if ent is None:
        if not x.is_contiguous():
            x = x.contiguous()
            y = torch.empty_like(x)
        try:
            ent = _build(x, weight, bias, y, eps, key)
        except Exception:
            _LAUNCH[key] = False
            ent = False
    if ent is False:
        return _run_slow(x, weight, bias, eps)
    if not x.is_contiguous():
        x = x.contiguous()

    st = _stream(_DEV())
    if ent[0] == "one":
        _, r, fn, pm, G, tail = ent
        r(G, 1, 1, st, fn, pm, None, None, None, x, weight, bias, y, eps, *tail)
    else:
        (_, r1, f1, p1, G, ns, t1,
         r2, f2, p2, nj, t2, sp, qp, np_) = ent
        r1(G, ns, 1, st, f1, p1, None, None, None, x, sp, qp, np_, *t1)
        r2(G, nj, 1, st, f2, p2, None, None, None, x, weight, bias, y,
           sp, qp, np_, eps, *t2)
    return y


@torch.no_grad()
def _run_slow(x, weight, bias, eps):
    """Generic JIT path -- correctness fallback if direct launch is unavailable."""
    B, C, H, W = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty_like(x)
    mode, p = _plan(B, C, H, W)
    if mode == "one":
        _gn_one[(p["G"],)](x, weight, bias, y, eps,
                           GS=p["GS"], HW=p["HW"], CPG=p["CPG"],
                           BLOCK=p["BLOCK"], NITER=p["NITER"],
                           num_warps=p["nw"], num_stages=1)
    else:
        G, ns = p["G"], p["NSPLIT"]
        n = G * ns
        buf = _scratch(3 * n, x.device)
        sp, qp, np_ = buf[0:n], buf[n:2 * n], buf[2 * n:3 * n]
        _gn_stats[(G, ns)](x, sp, qp, np_,
                           GS=p["GS"], CHUNK=p["CHUNK"], NSPLIT=ns,
                           BLOCK=p["B1"], NITER=p["IT1"],
                           num_warps=4, num_stages=2)
        _gn_norm[(G, p["NJ"])](x, weight, bias, y, sp, qp, np_, eps,
                               GS=p["GS"], HW=p["HW"], CPG=p["CPG"],
                               NSPLIT=ns, BNS=p["BNS"], BLOCK=p["BN"],
                               NITER=p["IT2"], num_warps=8, num_stages=1)
    return y
