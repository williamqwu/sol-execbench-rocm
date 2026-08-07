import os
import torch
import triton
import triton.language as tl

_P = int(os.environ.get("KP", "2048"))
_W = int(os.environ.get("KW", "8"))
_ST = int(os.environ.get("KST", "1"))
_RED = os.environ.get("KRED", "torch")
_FAST = os.environ.get("KFAST", "1") == "1"
_RBLK = int(os.environ.get("KRBLK", "256"))


@triton.jit
def _bwd(
    GO, NRM, RSTD, W, O0, O1, PART,
    n_rows, rows_per_prog,
    HF: tl.constexpr, BA: tl.constexpr, BB: tl.constexpr,
    HAS_B: tl.constexpr, MASK_B: tl.constexpr, H: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    oa = tl.arange(0, BA)
    wa = tl.load(W + oa)
    acc_a = tl.zeros([BA], dtype=tl.float32)
    if HAS_B:
        ob = tl.arange(0, BB)
        mb = ob < (H - BA)
        if MASK_B:
            wb = tl.load(W + BA + ob, mask=mb, other=0.0)
        else:
            wb = tl.load(W + BA + ob)
        acc_b = tl.zeros([BB], dtype=tl.float32)

    row0 = pid * rows_per_prog
    if EVEN:
        nit = rows_per_prog
    else:
        nit = tl.minimum(rows_per_prog, n_rows - row0)

    for i in range(nit):
        base = (row0 + i).to(tl.int64) * H

        go_a = tl.load(GO + base + oa).to(tl.float32)
        nr_a = tl.load(NRM + base + oa)
        pa = go_a * nr_a
        s = tl.sum(pa * wa, axis=0)
        if HAS_B:
            if MASK_B:
                go_b = tl.load(GO + base + BA + ob, mask=mb, other=0.0).to(tl.float32)
                nr_b = tl.load(NRM + base + BA + ob, mask=mb, other=0.0)
            else:
                go_b = tl.load(GO + base + BA + ob).to(tl.float32)
                nr_b = tl.load(NRM + base + BA + ob)
            pb = go_b * nr_b
            s += tl.sum(pb * wb, axis=0)

        m = s / HF
        rs = tl.load(RSTD + row0 + i)

        acc_a += pa
        gx_a = (rs * (go_a * wa - m * nr_a)).to(tl.bfloat16)
        tl.store(O0 + base + oa, gx_a)
        tl.store(O1 + base + oa, gx_a)
        if HAS_B:
            acc_b += pb
            gx_b = (rs * (go_b * wb - m * nr_b)).to(tl.bfloat16)
            if MASK_B:
                tl.store(O0 + base + BA + ob, gx_b, mask=mb)
                tl.store(O1 + base + BA + ob, gx_b, mask=mb)
            else:
                tl.store(O0 + base + BA + ob, gx_b)
                tl.store(O1 + base + BA + ob, gx_b)

    tl.store(PART + pid * H + oa, acc_a)
    if HAS_B:
        if MASK_B:
            tl.store(PART + pid * H + BA + ob, acc_b, mask=mb)
        else:
            tl.store(PART + pid * H + BA + ob, acc_b)


@triton.jit
def _reduce(PART, GW, P, H: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLK + tl.arange(0, BLK)
    m = o < H
    a = tl.zeros([BLK], dtype=tl.float32)
    for i in range(P):
        a += tl.load(PART + i * H + o, mask=m, other=0.0)
    tl.store(GW + o, a, mask=m)


def _split(H):
    ba = 1 << (H.bit_length() - 1)
    if ba > H:
        ba >>= 1
    rem = H - ba
    if rem == 0:
        return ba, 0, False, False
    bb = triton.next_power_of_2(rem)
    return ba, bb, True, bb != rem


_cache = {}
_stream = None


def _ispec(v):
    """Mirror Triton's int specialization: ==1 and %16==0 both change codegen."""
    return (v == 1, (v % 16) == 0)


def _get(jit, key, args, constexprs, warps, stages):
    """Compile once, then launch through the low-level path."""
    ent = _cache.get(key)
    if ent is None:
        ent = jit.warmup(*args, *constexprs, grid=(1,),
                         num_warps=warps, num_stages=stages)
        _cache[key] = ent
    return ent


@torch.no_grad()
def run(grad_output, x, normalized, rstd, weight):
    global _stream
    H = grad_output.shape[-1]
    go = grad_output.contiguous()
    nrm = normalized.contiguous()
    rs = rstd.contiguous()
    w = weight.contiguous()
    n_rows = go.numel() // H

    o0 = torch.empty_like(go)
    o1 = torch.empty_like(go)

    P = min(_P, n_rows)
    rpp = triton.cdiv(n_rows, P)
    P = triton.cdiv(n_rows, rpp)
    even = (P * rpp) == n_rows

    part = torch.empty(P, H, device=go.device, dtype=torch.float32)
    BA, BB, HAS_B, MASK_B = _split(H)

    if not _FAST:
        _bwd[(P,)](
            go, nrm, rs, w, o0, o1, part, n_rows, rpp,
            HF=float(H), BA=BA, BB=BB, HAS_B=HAS_B, MASK_B=MASK_B, H=H,
            EVEN=even, num_warps=_W, num_stages=_ST,
        )
        return o0, o1, part.sum(0)

    if _stream is None:
        _stream = torch.cuda.current_stream().cuda_stream
    st = _stream

    align = tuple((p.data_ptr() % 16) == 0
                  for p in (go, nrm, rs, w, o0, o1, part))
    key = (H, BA, BB, HAS_B, MASK_B, even, _ispec(n_rows), _ispec(rpp),
           align, _W, _ST)
    cx = (float(H), BA, BB, HAS_B, MASK_B, H, even)
    kern = _get(_bwd, key, (go, nrm, rs, w, o0, o1, part, n_rows, rpp), cx, _W, _ST)
    kern.run(P, 1, 1, st, kern.function, kern.packed_metadata, None, None, None,
             go, nrm, rs, w, o0, o1, part, n_rows, rpp, *cx)

    if _RED == "torch":
        return o0, o1, part.sum(0)

    gw = torch.empty(H, device=go.device, dtype=torch.float32)
    ng = triton.cdiv(H, _RBLK)
    rkey = ("red", H, _RBLK, _ispec(P),
            (part.data_ptr() % 16 == 0, gw.data_ptr() % 16 == 0))
    rk = _get(_reduce, rkey, (part, gw, P), (H, _RBLK), 4, 1)
    rk.run(ng, 1, 1, st, rk.function, rk.packed_metadata, None, None, None,
           part, gw, P, H, _RBLK)
    return o0, o1, gw
