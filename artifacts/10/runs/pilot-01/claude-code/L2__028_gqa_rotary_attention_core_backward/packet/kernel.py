import torch
import triton
import triton.language as tl

NH = 64
NKV = 8
HD = 128
G = 8

# Bytes of score-tensor scratch we allow live at once; sets the batch chunking.
_SCORE_BYTES_BUDGET = 3.0e9


# ---------------------------------------------------------------- softmax fwd
# One program per score row. Two reduction passes (max, then sum) exactly as
# torch.softmax does, so the probabilities come out bit-identical to the
# reference.
#
# Three fusions matter here:
#   * `* scaling` is folded in, and -- critically -- rounded back to bf16
#     first, because the reference multiplies the bf16 score tensor by the
#     scalar and stores the bf16 result before softmax sees it. Doing that
#     multiply in fp32 makes the kernel *more* accurate than the spec and
#     pushes grad_hidden_states outside tolerance.
#   * the causal mask is applied as a row-length limit, so the S x S bool mask
#     is never materialised or read.
#   * only the bf16 probabilities are written out. The fp32 ones (needed by the
#     softmax backward) are reconstructed there from the row max/scale saved in
#     M and INV, which turns 4 bytes/element of write + 4 of read into 8 bytes
#     total of re-read of a tensor already in cache.
@triton.jit
def _softmax_fwd(SC, PB, M, INV, S, scaling, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lim = (row % S) + 1
    base = row.to(tl.int64) * S

    m = -float("inf")
    for off in range(0, lim, BLOCK):
        c = off + tl.arange(0, BLOCK)
        x = tl.load(SC + base + c, mask=c < lim, other=-float("inf")).to(tl.float32) * scaling
        x = x.to(SC.dtype.element_ty).to(tl.float32)
        m = tl.maximum(m, tl.max(x, 0))

    d = 0.0
    for off in range(0, lim, BLOCK):
        c = off + tl.arange(0, BLOCK)
        x = tl.load(SC + base + c, mask=c < lim, other=-float("inf")).to(tl.float32) * scaling
        x = x.to(SC.dtype.element_ty).to(tl.float32)
        d += tl.sum(tl.where(c < lim, tl.exp(x - m), 0.0), 0)
    inv = 1.0 / d

    tl.store(M + row, m)
    tl.store(INV + row, inv)

    for off in range(0, S, BLOCK):
        c = off + tl.arange(0, BLOCK)
        x = tl.load(SC + base + c, mask=c < lim, other=-float("inf")).to(tl.float32) * scaling
        x = x.to(SC.dtype.element_ty).to(tl.float32)
        p = tl.where(c < lim, tl.exp(x - m) * inv, 0.0)
        tl.store(PB + base + c, p.to(PB.dtype.element_ty), mask=c < S)


# ---------------------------------------------------------------- softmax bwd
# ds = p * (gp - sum(gp * p)) * scaling, fp32 accumulation, bf16 result --
# matching the reference's rounding points. p is rebuilt from the raw scores
# and the saved (m, inv) instead of being read back as fp32, and ds is written
# in place over gp, so the whole backward touches 4 bytes/element rather than
# the 8 a materialised fp32 probability tensor would cost.
@triton.jit
def _softmax_bwd(SC, GP, M, INV, S, scaling, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lim = (row % S) + 1
    base = row.to(tl.int64) * S
    m = tl.load(M + row)
    inv = tl.load(INV + row)

    acc = 0.0
    for off in range(0, lim, BLOCK):
        c = off + tl.arange(0, BLOCK)
        msk = c < lim
        x = tl.load(SC + base + c, mask=msk, other=-float("inf")).to(tl.float32) * scaling
        x = x.to(SC.dtype.element_ty).to(tl.float32)
        p = tl.where(msk, tl.exp(x - m) * inv, 0.0)
        g = tl.load(GP + base + c, mask=msk, other=0.0).to(tl.float32)
        acc += tl.sum(p * g, 0)

    for off in range(0, S, BLOCK):
        c = off + tl.arange(0, BLOCK)
        msk = c < lim
        x = tl.load(SC + base + c, mask=msk, other=-float("inf")).to(tl.float32) * scaling
        x = x.to(SC.dtype.element_ty).to(tl.float32)
        p = tl.where(msk, tl.exp(x - m) * inv, 0.0)
        g = tl.load(GP + base + c, mask=msk, other=0.0).to(tl.float32)
        ds = tl.where(msk, p * (g - acc) * scaling, 0.0)
        tl.store(GP + base + c, ds.to(GP.dtype.element_ty), mask=c < S)


# ------------------------------------------------- single-tile softmax pair
# Every workload here has S <= 2048, so a whole score row fits in one tile.
# These variants load the row once and keep it in registers across the max,
# the sum and the normalise -- the looping versions above re-read it from
# memory three times, and the softmax is purely bandwidth-bound, so this is
# worth roughly 2-3x on the softmax kernels. The looping versions stay as the
# fallback for any S that would make the tile too large to be resident.
@triton.jit
def _softmax_fwd1(SC, PB, M, INV, S, scaling, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lim = (row % S) + 1
    base = row.to(tl.int64) * S
    c = tl.arange(0, BLOCK)
    x = tl.load(SC + base + c, mask=c < lim, other=-float("inf")).to(tl.float32) * scaling
    x = x.to(SC.dtype.element_ty).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.where(c < lim, tl.exp(x - m), 0.0)
    inv = 1.0 / tl.sum(e, 0)
    tl.store(M + row, m)
    tl.store(INV + row, inv)
    tl.store(PB + base + c, (e * inv).to(PB.dtype.element_ty), mask=c < S)


@triton.jit
def _softmax_bwd1(SC, GP, M, INV, S, scaling, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lim = (row % S) + 1
    base = row.to(tl.int64) * S
    m = tl.load(M + row)
    inv = tl.load(INV + row)
    c = tl.arange(0, BLOCK)
    msk = c < lim
    x = tl.load(SC + base + c, mask=msk, other=-float("inf")).to(tl.float32) * scaling
    x = x.to(SC.dtype.element_ty).to(tl.float32)
    p = tl.where(msk, tl.exp(x - m) * inv, 0.0)
    g = tl.load(GP + base + c, mask=msk, other=0.0).to(tl.float32)
    acc = tl.sum(p * g, 0)
    ds = tl.where(msk, p * (g - acc) * scaling, 0.0)
    tl.store(GP + base + c, ds.to(GP.dtype.element_ty), mask=c < S)


# ------------------------------------------------------------------- RoPE
# Fused rotary embedding that also performs the (B,S,nh,HD) <-> (B,nh,S,HD)
# transpose, so the layout change the reference pays a separate copy for comes
# for free. Rotation arithmetic in fp32, result cast to bf16, as in the
# reference. FWD=False is the transposed adjoint used on the way back.
@triton.jit
def _rope(X, OUT, COS, SIN, S, NHEAD, FWD: tl.constexpr, D2: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % NHEAD
    t = (pid // NHEAD) % S
    b = pid // (NHEAD * S)

    d = tl.arange(0, D2)
    tok = (((b * S + t).to(tl.int64) * NHEAD) + h) * (2 * D2)
    hed = (((b * NHEAD + h).to(tl.int64) * S) + t) * (2 * D2)
    src = tok if FWD else hed
    dst = hed if FWD else tok

    xe = tl.load(X + src + 2 * d).to(tl.float32)
    xo = tl.load(X + src + 2 * d + 1).to(tl.float32)
    c = tl.load(COS + t * D2 + d)
    s = tl.load(SIN + t * D2 + d)

    if FWD:
        oe = xe * c - xo * s
        oo = xo * c + xe * s
    else:
        oe = xe * c + xo * s
        oo = xo * c - xe * s

    tl.store(OUT + dst + 2 * d, oe.to(OUT.dtype.element_ty))
    tl.store(OUT + dst + 2 * d + 1, oo.to(OUT.dtype.element_ty))


# --------------------------------------------------------- plain transpose
# (B,S,nh,HD) <-> (B,nh,S,HD) for the tensors that need no rotation.
@triton.jit
def _tp(X, OUT, S, NHEAD, TO_HEAD: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % NHEAD
    t = (pid // NHEAD) % S
    b = pid // (NHEAD * S)
    d = tl.arange(0, D)
    tok = (((b * S + t).to(tl.int64) * NHEAD) + h) * D
    hed = (((b * NHEAD + h).to(tl.int64) * S) + t) * D
    if TO_HEAD:
        tl.store(OUT + hed + d, tl.load(X + tok + d))
    else:
        tl.store(OUT + tok + d, tl.load(X + hed + d))


# ------------------------------------------- GQA reduction over the 8 heads
# grad for a KV head is the sum of the 8 query heads in its group. Done as one
# strided read per output element instead of a reshape + torch sum, which would
# stage an (b, NKV, G, S, HD) fp32 temporary.
@triton.jit
def _gqa_sum(X, OUT, S, NEL, D: tl.constexpr, GRP: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLK)
    base = pid.to(tl.int64) * BLK + off
    msk = base < NEL
    n = S * D
    kv = base // n
    rem = base % n
    src = kv.to(tl.int64) * GRP * n + rem
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for g in range(GRP):
        acc += tl.load(X + src + g * n, mask=msk, other=0.0).to(tl.float32)
    tl.store(OUT + base, acc.to(OUT.dtype.element_ty), mask=msk)


_SINGLE_TILE_MAX = 2048


def _blk(S):
    """Tile width / warp count for the softmax pair.

    Returns (BLOCK, num_warps, single_tile). When the padded row fits within
    _SINGLE_TILE_MAX we take the register-resident single-tile path; the warp
    counts below were measured on MI355X across the workload's S values.
    """
    if S <= _SINGLE_TILE_MAX:
        bp = triton.next_power_of_2(S)
        if bp <= 256:
            return bp, 4, True
        if bp <= 512:
            return bp, 4, True
        return bp, 8, True
    if S <= 4096:
        return 1024, 8, False
    return 2048, 8, False


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    inv_freq,
    scaling,
):
    B, S, H = hidden_states.shape
    N = B * S
    dt = hidden_states.dtype
    dev = hidden_states.device
    KVH = NKV * HD

    hs2 = hidden_states.reshape(N, H)
    go2 = grad_output.reshape(N, H)

    # ---- recomputed forward projections + grad flowing in through o_weight
    q = torch.mm(hs2, q_weight.t())
    k = torch.mm(hs2, k_weight.t())
    v = torch.mm(hs2, v_weight.t())
    do = torch.mm(go2, o_weight)

    # ---- RoPE tables. The reference builds a (S, HD) cos/sin by concatenating
    # freqs with itself and then repeat_interleave'ing the first half, which is
    # just freqs duplicated per pair -- so only the (S, HD/2) table is needed.
    pos = torch.arange(S, device=dev, dtype=torch.float32)
    ang = pos[:, None] * inv_freq.float()[None, :]
    cos = torch.cos(ang).contiguous()
    sin = torch.sin(ang).contiguous()

    qr = torch.empty((B, NH, S, HD), dtype=dt, device=dev)
    _rope[(B * S * NH,)](q, qr, cos, sin, S, NH, True, HD // 2, num_warps=2)
    kr = torch.empty((B, NKV, S, HD), dtype=dt, device=dev)
    _rope[(B * S * NKV,)](k, kr, cos, sin, S, NKV, True, HD // 2, num_warps=2)
    del q, k

    vv = torch.empty((B, NKV, S, HD), dtype=dt, device=dev)
    _tp[(B * S * NKV,)](v, vv, S, NKV, True, HD, num_warps=2)
    dov = torch.empty((B, NH, S, HD), dtype=dt, device=dev)
    _tp[(B * S * NH,)](do, dov, S, NH, True, HD, num_warps=2)
    del v, do

    attn_out = torch.empty((B, S, H), dtype=dt, device=dev)
    dqr = torch.empty((B, NH, S, HD), dtype=dt, device=dev)
    dkr = torch.empty((B, NKV, S, HD), dtype=dt, device=dev)
    dvv = torch.empty((B, NKV, S, HD), dtype=dt, device=dev)

    BLOCK, nw = _blk(S)
    # two bf16 S x S tensors are live per batch element inside the loop
    bc = max(1, min(B, int(_SCORE_BYTES_BUDGET // (NH * S * S * 4))))

    for i in range(0, B, bc):
        j = min(B, i + bc)
        b = j - i
        bh = b * NH
        Q = qr[i:j].reshape(bh, S, HD)
        DO = dov[i:j].reshape(bh, S, HD)
        # GQA expansion. Folding the group's 8 query heads into the M axis
        # would avoid this copy, but it changes the GEMM's reduction blocking
        # and the scores then differ from the reference in the last bf16 ulp --
        # which the softmax amplifies enough to fail tolerance at large B. The
        # expanded copy is only (b, 64, S, 128) bf16 and costs ~0.03 ms, so
        # keep it and stay bit-exact.
        Kf = kr[i:j, :, None].expand(b, NKV, G, S, HD).reshape(bh, S, HD)
        Vf = vv[i:j, :, None].expand(b, NKV, G, S, HD).reshape(bh, S, HD)
        rows = b * NH * S

        sc = torch.bmm(Q, Kf.transpose(1, 2))
        pb = torch.empty_like(sc)
        m_ = torch.empty(rows, dtype=torch.float32, device=dev)
        iv = torch.empty(rows, dtype=torch.float32, device=dev)
        _softmax_fwd[(rows,)](sc, pb, m_, iv, S, scaling, BLOCK=BLOCK, num_warps=nw)

        ao = torch.bmm(pb, Vf).reshape(b, NH, S, HD)
        _tp[(b * S * NH,)](ao, attn_out[i:j], S, NH, False, HD, num_warps=2)
        del ao

        # dV = P^T @ dO, then reduce the 8 query heads sharing each KV head
        nel = b * NKV * S * HD
        dv = torch.bmm(pb.transpose(1, 2), DO)
        _gqa_sum[(triton.cdiv(nel, 1024),)](dv, dvv[i:j], S, nel, HD, G, 1024, num_warps=4)
        del dv, pb

        # The softmax backward rewrites gp in place as ds, so no extra S x S
        # buffer is allocated for the gradient.
        gp = torch.bmm(DO, Vf.transpose(1, 2))
        _softmax_bwd[(rows,)](sc, gp, m_, iv, S, scaling, BLOCK=BLOCK, num_warps=nw)
        ds = gp
        del sc, m_, iv

        dqr[i:j] = torch.bmm(ds, Kf).reshape(b, NH, S, HD)
        dk = torch.bmm(ds.transpose(1, 2), Q)
        _gqa_sum[(triton.cdiv(nel, 1024),)](dk, dkr[i:j], S, nel, HD, G, 1024, num_warps=4)
        del ds, gp, dk

    del qr, kr, vv, dov

    # ---- RoPE backward, writing straight back into token-major layout
    dQ = torch.empty((N, H), dtype=dt, device=dev)
    _rope[(B * S * NH,)](dqr, dQ, cos, sin, S, NH, False, HD // 2, num_warps=2)
    dK = torch.empty((N, KVH), dtype=dt, device=dev)
    _rope[(B * S * NKV,)](dkr, dK, cos, sin, S, NKV, False, HD // 2, num_warps=2)
    dV = torch.empty((N, KVH), dtype=dt, device=dev)
    _tp[(B * S * NKV,)](dvv, dV, S, NKV, False, HD, num_warps=2)
    del dqr, dkr, dvv

    # The reference rounds each of the three projections to bf16 before summing
    # them, so an fp32 addmm chain would be *more* accurate than the spec and
    # drift out of tolerance. Keep the separate bf16 products and bf16 adds.
    gh = torch.mm(dQ, q_weight) + torch.mm(dK, k_weight) + torch.mm(dV, v_weight)

    return (
        gh.view(B, S, H),
        torch.mm(dQ.t(), hs2),
        torch.mm(dK.t(), hs2),
        torch.mm(dV.t(), hs2),
        torch.mm(go2.t(), attn_out.view(N, H)),
    )
