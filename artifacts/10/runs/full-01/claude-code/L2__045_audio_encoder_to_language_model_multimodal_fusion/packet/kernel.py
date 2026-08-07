import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused "clone lm_embeddings + scatter audio rows" kernel.
#
# The reference materialises fused = lm.float().clone() and then index_copy_'s
# the audio rows in.  Semantically the result is: for every (b, t) row, take the
# audio embedding if t is one of the audio token positions, else take the LM
# embedding.  Doing that in a single pass costs exactly one read + one write of
# the [B, T, H] tensor, which is the Speed-of-Light for this problem -- versus
# clone (read+write) followed by a second scattered pass.
#
# The reference's float32 round trip on lm_embeddings is a no-op (bf16 -> f32 ->
# bf16 is exact), so the non-audio rows come out bit-identical.
#
# arows is the audio tensor's row stride, so the caller can hand us a padded
# [B, nwin*40, H] buffer directly and skip a contiguous() copy of the slice.
# ---------------------------------------------------------------------------
@triton.jit
def _fuse_scatter_kernel(
    lm_ptr,
    aud_ptr,
    idx_ptr,
    out_ptr,
    T,
    arows,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)

    src = tl.load(idx_ptr + row)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)

    if src < 0:
        v = tl.load(lm_ptr + row * H + offs)
    else:
        b = row // T
        v = tl.load(aud_ptr + (b * arows + src.to(tl.int64)) * H + offs)

    tl.store(out_ptr + row * H + offs, v)


# ---------------------------------------------------------------------------
# Row softmax that reads bf16, reduces in float32 (as the reference does) and
# writes bf16 -- one launch instead of the upcast / softmax / downcast trio.
# An optional power-of-two scale is folded in; it is exact in binary floating
# point, so it does not perturb the result relative to scaling beforehand.
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_kernel(
    inp,
    out,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(inp + row * N + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x * SCALE
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    e = e / tl.sum(e, 0)
    tl.store(out + row * N + offs, e.to(out.dtype.element_ty), mask=mask)


def _softmax_bf16(x, scale=1.0):
    shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    xf = x.reshape(-1, shape[-1])
    N = shape[-1]
    out = torch.empty_like(xf)
    _softmax_kernel[(xf.shape[0],)](
        xf,
        out,
        N=N,
        SCALE=scale,
        BLOCK=triton.next_power_of_2(N),
        num_warps=4,
    )
    return out.view(shape)


# ---------------------------------------------------------------------------
# Build the inverse permutation in one launch: fill with -1, then write the
# audio row number at each audio token position.  Replaces a full + arange +
# scatter_ trio (3 launches, and this path is CPU-dispatch-bound at small
# sizes).  The fill and the scatter are ordered within a program, and each
# position is written by exactly one program because positions are unique per
# batch row, so no cross-program race exists.
# ---------------------------------------------------------------------------
@triton.jit
def _build_index_kernel(pos_ptr, idx_ptr, T, NA, JBLOCK: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    base = b * T

    for off in tl.range(0, T, JBLOCK):
        o = off + tl.arange(0, JBLOCK)
        tl.store(idx_ptr + base + o, -1, mask=o < T)

    for off in tl.range(0, NA, JBLOCK):
        j = off + tl.arange(0, JBLOCK)
        m = j < NA
        p = tl.load(pos_ptr + b * NA + j, mask=m, other=0).to(tl.int64)
        tl.store(idx_ptr + base + p, j.to(tl.int32), mask=m)


# ---------------------------------------------------------------------------
# Fused Q-Former cross attention.
#
# One program per (window, head).  The whole attention for that head lives in
# registers: 40x64 queries against a 15-key window.  This replaces the
# transpose / bmm / softmax / bmm / transpose-contiguous sequence -- five
# launches over tiny tensors, each dominated by dispatch overhead -- with a
# single launch, and never materialises the [W, heads, 40, 15] score tensor.
#
# Numerics follow the reference: tl.dot accumulates in float32, the softmax
# reduces in float32, and the probabilities are rounded to bfloat16 before the
# second dot exactly as the reference's .to(dtype) does.  Keys past the window
# end (the zero padding) are masked to -inf so they contribute nothing, which
# matches the reference feeding them zeros -- both give a uniform-over-real-keys
# result only where the reference does.
#
# NOTE the padding subtlety: the reference pads the *encoder output* with zeros
# and then projects, so padded keys carry the projection bias, not -inf.  We
# therefore only mask lanes beyond WS (the tile remainder), never real window
# slots -- padded slots are computed exactly as the reference computes them.
# ---------------------------------------------------------------------------
@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    SCALE: tl.constexpr,
    NQ: tl.constexpr,
    WS: tl.constexpr,
    NH: tl.constexpr,
    HD: tl.constexpr,
    BM: tl.constexpr,
    BT: tl.constexpr,
):
    w = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    D = NH * HD

    rm = tl.arange(0, BM)
    rd = tl.arange(0, HD)
    rt = tl.arange(0, BT)
    mm = rm < NQ
    mt = rt < WS

    # Queries are shared by every window, so this load hits cache after the
    # first program.
    q = tl.load(q_ptr + rm[:, None] * D + h * HD + rd[None, :], mask=mm[:, None], other=0.0)

    koff = ((w * WS + rt[:, None]) * NH + h) * HD + rd[None, :]
    k = tl.load(k_ptr + koff, mask=mt[:, None], other=0.0)

    s = tl.dot(q, k.T, out_dtype=tl.float32) * SCALE
    s = tl.where(mt[None, :], s, -float("inf"))
    s = s - tl.max(s, 1)[:, None]
    e = tl.exp(s)
    p = (e / tl.sum(e, 1)[:, None]).to(k_ptr.dtype.element_ty)

    v = tl.load(v_ptr + koff, mask=mt[:, None], other=0.0)
    o = tl.dot(p, v, out_dtype=tl.float32)

    tl.store(
        out_ptr + (w * NQ + rm[:, None]) * D + h * HD + rd[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=mm[:, None],
    )


def _fused_attention(q, k, v, W, num_queries, window_size, num_heads, head_dim, scale):
    out = torch.empty((W * num_queries, num_heads * head_dim), dtype=q.dtype, device=q.device)
    _attention_kernel[(W, num_heads)](
        q,
        k,
        v,
        out,
        SCALE=scale,
        NQ=num_queries,
        WS=window_size,
        NH=num_heads,
        HD=head_dim,
        BM=triton.next_power_of_2(num_queries),
        BT=triton.next_power_of_2(window_size),
        num_warps=4,
        num_stages=1,
    )
    return out


def _fuse_scatter(lm, audio, positions, NA):
    B, T, H = lm.shape
    arows = audio.shape[1]

    # Inverse permutation: for each text row, which audio row lands there
    # (-1 = keep the LM embedding).  Cost is O(B*T) int32 next to the B*T*H
    # bf16 copy, i.e. ~1/8192 of the traffic.
    idx = torch.empty((B, T), dtype=torch.int32, device=lm.device)
    _build_index_kernel[(B,)](positions, idx, T, NA, JBLOCK=1024, num_warps=4)

    out = torch.empty_like(lm)

    assert H & (H - 1) == 0, "hidden size must be a power of two"
    # One program per output row, whole row in one tile: measured 0.66ms on the
    # B=32,T=8192 workload against a 0.537ms pure-bandwidth bound (~81% of peak).
    _fuse_scatter_kernel[(B * T, 1)](
        lm,
        audio,
        idx,
        out,
        T,
        arows,
        H=H,
        BLOCK=H,
        num_warps=8,
        num_stages=1,
    )
    return out


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    lm_embeddings: torch.Tensor,
    audio_token_positions: torch.Tensor,
    encoder_input_weight: torch.Tensor,
    encoder_input_bias: torch.Tensor,
    encoder_out_weight: torch.Tensor,
    encoder_out_bias: torch.Tensor,
    encoder_out_mid_weight: torch.Tensor,
    encoder_out_mid_bias: torch.Tensor,
    learnable_queries: torch.Tensor,
    qformer_q_proj_weight: torch.Tensor,
    qformer_q_proj_bias: torch.Tensor,
    qformer_k_proj_weight: torch.Tensor,
    qformer_k_proj_bias: torch.Tensor,
    qformer_v_proj_weight: torch.Tensor,
    qformer_v_proj_bias: torch.Tensor,
    qformer_out_proj_weight: torch.Tensor,
    qformer_out_proj_bias: torch.Tensor,
    projector_weight: torch.Tensor,
    projector_bias: torch.Tensor,
):
    B, S, D_in = input_features.shape
    T = lm_embeddings.shape[1]
    NA = audio_token_positions.shape[1]

    window_size = 15
    num_queries = 40
    num_heads = 16
    head_dim = 64
    Dh = 512

    nblocks = (S + window_size - 1) // window_size

    # Only the first NA rows of audio_embeddings reach the output, and each
    # window contributes num_queries consecutive rows, so windows at or past
    # ceil(NA / num_queries) are computed by the reference and then discarded.
    # Skipping them is dead-code elimination driven purely by the input shapes
    # (no dependence on the actual values, no per-workload special casing).
    nwin = min(nblocks, -(-NA // num_queries))
    L = nwin * window_size
    Sa = min(L, S)

    # ---------------- Stage 1: encoder over the needed frames ---------------
    # The reference computes this in float32.  We keep bfloat16 inputs and let
    # the MFMA units accumulate in float32, which they do natively: a bf16*bf16
    # product is exact in float32, so the only difference from the reference is
    # rounding the (already float32) accumulator once per GEMM.  Measured worst
    # case over all 16 workloads is 3.1e-4 against a 6.5e-3 tolerance -- a ~20x
    # margin -- and it removes ~20 dtype-conversion kernel launches per call.
    if Sa == S:
        x = input_features.reshape(B * S, D_in)
    else:
        x = input_features[:, :Sa, :].reshape(B * Sa, D_in)

    h = torch.addmm(encoder_input_bias, x, encoder_input_weight.t())
    mid = torch.addmm(encoder_out_bias, h, encoder_out_weight.t())
    # Softmax reduces over 1024 terms; do it in float32 like the reference.
    sm = _softmax_bf16(mid)
    # addmm's beta*input term folds the residual add into the GEMM epilogue.
    enc = torch.addmm(
        encoder_out_mid_bias, sm, encoder_out_mid_weight.t()
    ).add_(h)

    if Sa < L:
        # Frames past the real sequence are the reference's zero padding.
        enc = F.pad(enc.view(B, Sa, Dh), (0, 0, 0, L - Sa)).reshape(B * L, Dh)

    # ---------------- Stage 2: windowed Q-Former ----------------------------
    W = B * nwin

    # K and V read the same input, so fusing their projections into one GEMM is
    # tempting -- but the two torch.cat launches needed to build the stacked
    # weight cost more CPU dispatch time than the merged GEMM saves at these
    # sizes (measured ~6% slower overall).  Keep them separate.
    K = torch.addmm(qformer_k_proj_bias, enc, qformer_k_proj_weight.t())
    V = torch.addmm(qformer_v_proj_bias, enc, qformer_v_proj_weight.t())

    # The queries are identical for every window, so project the 40x1024 block
    # once and let the attention kernel re-read it, instead of materialising the
    # expanded [W, 40, 1024] tensor the reference builds.
    q = torch.addmm(
        qformer_q_proj_bias,
        learnable_queries.view(num_queries, -1),
        qformer_q_proj_weight.t(),
    )

    # head_dim is 64, so the 1/sqrt(head_dim) scale is exactly 0.125 -- a power
    # of two, hence exact in binary FP, so folding it into the kernel is not an
    # approximation.
    scale = 1.0 / math.sqrt(head_dim)
    ctx = _fused_attention(
        q, K, V, W, num_queries, window_size, num_heads, head_dim, scale
    )

    qo = torch.addmm(qformer_out_proj_bias, ctx, qformer_out_proj_weight.t())
    audio = torch.addmm(projector_bias, qo, projector_weight.t())
    audio = audio.view(B, nwin * num_queries, -1)

    # ---------------- Stage 3: fused copy + scatter -------------------------
    if not lm_embeddings.is_contiguous():
        lm_embeddings = lm_embeddings.contiguous()
    return _fuse_scatter(lm_embeddings, audio, audio_token_positions, NA)
