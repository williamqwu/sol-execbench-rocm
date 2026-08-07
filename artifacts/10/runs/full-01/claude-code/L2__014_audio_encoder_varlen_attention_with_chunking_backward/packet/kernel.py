import torch
import triton
import triton.language as tl


# ===========================================================================
# The reference performs *every* step in float32.  All the tensor inputs are
# bfloat16, so a bf16 x bf16 -> fp32 tl.dot reproduces its matmuls exactly;
# the places that genuinely need more than bf16 are the ones where an fp32
# intermediate is consumed by a later matmul (dao, and dq/dk/dv).  For those we
# use a split representation  x = hi + lo  with hi, lo bf16, which recovers
# ~16 mantissa bits -- far more than the bf16 outputs can express -- at the
# cost of one extra tl.dot instead of a 20x-slower fp32 GEMM.
# ===========================================================================


# ---------------------------------------------------------------------------
# C = A @ B  with fp32 accumulation.
#   A: fp32 (split into hi+lo bf16 on the fly) when SPLIT, else bf16
#   B: bf16
# Arbitrary strides on A and B so the same kernel serves A and A^T.
# ---------------------------------------------------------------------------
@triton.jit
def _gemm(
    A, B, C,
    M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    SPLIT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    # group-ordered tile walk for L2 reuse
    wg = GROUP * nn
    g = pid // wg
    first = g * GROUP
    gsz = min(nm - first, GROUP)
    pm = first + ((pid % wg) % gsz)
    pn = (pid % wg) // gsz

    offm = (pm * BM + tl.arange(0, BM)) % M
    offn = (pn * BN + tl.arange(0, BN)) % N
    offk = tl.arange(0, BK)

    a_ptrs = A + offm[:, None] * sam + offk[None, :] * sak
    b_ptrs = B + offk[:, None] * sbk + offn[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            km = offk[None, :] < K - k * BK
            a = tl.load(a_ptrs, mask=km, other=0.0)
            b = tl.load(b_ptrs, mask=tl.trans(km), other=0.0)
        if SPLIT:
            ah = a.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(ah, b, acc)
            acc = tl.dot(al, b, acc)
        else:
            acc = tl.dot(a.to(tl.bfloat16), b, acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    c_ptrs = C + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def _pick(M, N, K):
    """Tile shape: prefer enough tiles to fill 256 CUs without going tiny."""
    if M * N <= 256 * 64 * 64:
        BM, BN = 64, 64
    elif M * N <= 256 * 128 * 64:
        BM, BN = 64, 128
    else:
        BM, BN = 128, 128
    BM = min(BM, max(16, triton.next_power_of_2(M)))
    BN = min(BN, max(16, triton.next_power_of_2(N)))
    BK = 64 if K >= 64 else max(16, triton.next_power_of_2(K))
    nw = 8 if BM * BN >= 128 * 64 else 4
    return BM, BN, BK, nw


def _mm(a, b, out_dtype, split):
    """a @ b -> tensor(out_dtype).  `a` may be an fp32 (possibly transposed)
    view; `b` must be bf16."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    BM, BN, BK, nw = _pick(M, N, K)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        SPLIT=split, BM=BM, BN=BN, BK=BK, GROUP=8,
        EVEN_K=(K % BK == 0),
        num_warps=nw, num_stages=2,
    )
    return c


# ---------------------------------------------------------------------------
# Fused per-chunk attention: forward recompute + full backward.
# One program per (chunk, head).  Chunk lengths here are <= 128, so the entire
# L x L score matrix stays in registers and never touches HBM.
# ---------------------------------------------------------------------------
@triton.jit
def _attn_chunk_fwd_bwd(
    Q, K, V, DO, O, DQKV, CU,
    s_qh, s_qt,
    s_dot, s_ot, s_gt,
    scale,
    QKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)

    start = tl.load(CU + c).to(tl.int32)
    end = tl.load(CU + c + 1).to(tl.int32)
    L = end - start

    offs = tl.arange(0, BLOCK_L)
    offd = tl.arange(0, D)
    m = offs < L
    rows = start + offs

    qbase = h * s_qh + rows[:, None] * s_qt + offd[None, :]
    q = tl.load(Q + qbase, mask=m[:, None], other=0.0)
    k = tl.load(K + qbase, mask=m[:, None], other=0.0)
    v = tl.load(V + qbase, mask=m[:, None], other=0.0)

    # dO arrives as fp32 (it is grad_output @ out_weight, an fp32 intermediate
    # in the reference).  Split it so the dots below stay accurate.
    dobase = rows[:, None] * s_dot + h * D + offd[None, :]
    do32 = tl.load(DO + dobase, mask=m[:, None], other=0.0)
    doh = do32.to(tl.bfloat16)
    dol = (do32 - doh.to(tl.float32)).to(tl.bfloat16)

    # ---- forward: p = softmax(q k^T * scale) ----------------------------
    s = tl.dot(q, tl.trans(k)) * scale
    s = tl.where(m[None, :], s, -1.0e30)      # finite sentinel: no NaNs
    e = tl.exp(s - tl.max(s, 1)[:, None])
    p = e / tl.sum(e, 1)[:, None]
    p = tl.where(m[:, None], p, 0.0)
    ph = p.to(tl.bfloat16)

    # ---- attention output recompute (feeds grad_out_weight) --------------
    o = tl.dot(ph, v)

    # ---- dV = P^T @ dO ----------------------------------------------------
    pt = tl.trans(ph)
    dv = tl.dot(pt, doh)
    dv = tl.dot(pt, dol, dv)

    # ---- dP = dO @ V^T ----------------------------------------------------
    vt = tl.trans(v)
    dp = tl.dot(doh, vt)
    dp = tl.dot(dol, vt, dp)

    # ---- softmax backward (fp32: this is the cancellation-sensitive step) --
    ds = p * (dp - tl.sum(p * dp, 1)[:, None])
    dsh = ds.to(tl.bfloat16)
    dsl = (ds - dsh.to(tl.float32)).to(tl.bfloat16)

    dq = tl.dot(dsh, k)
    dq = tl.dot(dsl, k, dq) * scale
    dsht = tl.trans(dsh)
    dslt = tl.trans(dsl)
    dk = tl.dot(dsht, q)
    dk = tl.dot(dslt, q, dk) * scale

    obase = rows[:, None] * s_ot + h * D + offd[None, :]
    tl.store(O + obase, o.to(O.dtype.element_ty), mask=m[:, None])

    gbase = rows[:, None] * s_gt + h * D + offd[None, :]
    tl.store(DQKV + gbase, dq, mask=m[:, None])
    tl.store(DQKV + gbase + QKV, dk, mask=m[:, None])
    tl.store(DQKV + gbase + 2 * QKV, dv, mask=m[:, None])


def _torch_attn_fallback(q, k, v, dao, lens, T, H, D, scale, dev):
    """Generic path for chunk lengths beyond the single-tile kernel."""
    qkv = H * D
    o = torch.empty((T, qkv), dtype=torch.bfloat16, device=dev)
    g = torch.empty((T, 3 * qkv), dtype=torch.float32, device=dev)
    do4 = dao.view(T, H, D).transpose(0, 1)  # (H, T, D) fp32
    off = 0
    for L in lens:
        if L <= 0:
            continue
        sl = slice(off, off + L)
        off += L
        qc = q[0, :, sl, :].float()
        kc = k[0, :, sl, :].float()
        vc = v[0, :, sl, :].float()
        dc = do4[:, sl, :].float()
        s = torch.matmul(qc, kc.transpose(-2, -1)) * scale
        p = torch.softmax(s, dim=-1)
        oc = torch.matmul(p, vc)
        dv = torch.matmul(p.transpose(-2, -1), dc)
        dp = torch.matmul(dc, vc.transpose(-2, -1))
        ds = p * (dp - (p * dp).sum(dim=-1, keepdim=True))
        dq = torch.matmul(ds, kc) * scale
        dk = torch.matmul(ds.transpose(-2, -1), qc) * scale
        o[sl] = oc.transpose(0, 1).reshape(L, qkv).to(torch.bfloat16)
        g[sl, 0:qkv] = dq.transpose(0, 1).reshape(L, qkv)
        g[sl, qkv:2 * qkv] = dk.transpose(0, 1).reshape(L, qkv)
        g[sl, 2 * qkv:] = dv.transpose(0, 1).reshape(L, qkv)
    return o, g


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
):
    dev = hidden_states.device
    T, d_model = hidden_states.shape
    H = query_states.shape[1]
    D = query_states.shape[3]
    qkv = H * D
    scale = D ** -0.5

    # ---- 1) dAttnOut = grad_output @ out_weight, kept in fp32 -------------
    dao = _mm(grad_output, out_weight, torch.float32, False)

    # chunk boundaries (the reference reads them on the host too)
    cu_list = cu_seqlens.tolist()
    lens = [cu_list[i + 1] - cu_list[i] for i in range(len(cu_list) - 1)]
    Lmax = max(lens) if lens else 0

    q = query_states.contiguous()
    k = key_states.contiguous()
    v = value_states.contiguous()

    # ---- 2) fused attention recompute + backward --------------------------
    if 0 < Lmax <= 256:
        BLOCK_L = max(16, triton.next_power_of_2(Lmax))
        attn_out = torch.empty((T, qkv), dtype=torch.bfloat16, device=dev)
        dqkv = torch.empty((T, 3 * qkv), dtype=torch.float32, device=dev)
        _attn_chunk_fwd_bwd[(len(lens), H)](
            q, k, v, dao, attn_out, dqkv, cu_seqlens,
            q.stride(1), q.stride(2),
            dao.stride(0), attn_out.stride(0), dqkv.stride(0),
            scale,
            QKV=qkv, D=D, BLOCK_L=BLOCK_L,
            num_warps=8, num_stages=1,
        )
    else:
        attn_out, dqkv = _torch_attn_fallback(q, k, v, dao, lens, T, H, D,
                                              scale, dev)

    # ---- 3) weight / bias gradients ---------------------------------------
    grad_out_weight = torch.mm(grad_output.t(), attn_out)
    grad_out_bias = grad_output.sum(dim=0, dtype=torch.float32).to(torch.bfloat16)

    # gw = [dq|dk|dv]^T @ hidden_states        (3*qkv, d_model)
    gw = _mm(dqkv.t(), hidden_states, torch.bfloat16, True)
    gb = dqkv.sum(dim=0).to(torch.bfloat16)

    # ---- 4) grad wrt hidden_states: one fused GEMM ------------------------
    w_cat = torch.cat((q_weight, k_weight, v_weight), dim=0)   # (3*qkv, d_model)
    grad_hidden_states = _mm(dqkv, w_cat, torch.bfloat16, True)

    return (
        grad_hidden_states,
        gw[0:qkv],
        gb[0:qkv],
        gw[qkv:2 * qkv],
        gb[qkv:2 * qkv],
        gw[2 * qkv:],
        gb[2 * qkv:],
        grad_out_weight,
        grad_out_bias,
    )
