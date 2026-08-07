import torch
import triton
import triton.language as tl

from aiter.ops.triton.gated_delta_net import gated_delta_rule as _gdr


# Reuse the tuned component kernels behind AITer's chunk GDR implementation.
_fwd = _gdr.chunk_gated_delta_rule_fwd_opt
_fused_cumsum_kkt = _fwd.__globals__["fused_chunk_local_cumsum_scaled_dot_kkt_fwd"]
_fused_solve = _fwd.__globals__["fused_solve_tril_recompute_w_u"]
_chunk_h = _fwd.__globals__["chunk_gated_delta_rule_fwd_h_opt"]
_chunk_o = _fwd.__globals__["chunk_fwd_o_opt"]


@triton.jit
def _transpose_qk_kernel(q_src, k_src, q_dst, k_dst, n_elements: tl.constexpr,
                         T: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    d = offs % 128
    x = offs // 128
    h = x % 4
    x = x // 4
    t = x % T
    b = x // T
    src = ((b * 4 + h) * T + t) * 128 + d
    tl.store(q_dst + offs, tl.load(q_src + src, mask=mask), mask=mask)
    tl.store(k_dst + offs, tl.load(k_src + src, mask=mask), mask=mask)


@triton.jit
def _prepare_vgb_kernel(v_src, g_src, b_src, v_dst, g_dst, b_dst, gh_dst,
                        rows: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    mask = (row < rows) & (d < 128)
    hi = row % 16
    x = row // 16
    t = x % T
    batch = x // T
    kh = hi // 4
    group = hi % 4
    ho = group * 4 + kh
    src_row = (batch * 16 + ho) * T + t
    val = tl.load(v_src + src_row * 128 + d, mask=mask)
    tl.store(v_dst + row * 128 + d, val, mask=mask)
    scalar_mask = mask & (d == 0)
    gv = tl.load(g_src + src_row + d * 0, mask=scalar_mask).to(tl.float32)
    bv = tl.load(b_src + src_row + d * 0, mask=scalar_mask).to(tl.float32)
    tl.store(g_dst + row + d * 0, gv, mask=scalar_mask)
    tl.store(b_dst + row + d * 0, bv, mask=scalar_mask)
    tl.store(gh_dst + (batch * 16 + hi) * T + t + d * 0, gv, mask=scalar_mask)


@triton.jit
def _finish_kernel(out, qh, raw_g, cum_g, result, n_elements: tl.constexpr,
                   T: tl.constexpr, NC: tl.constexpr, SCALE: tl.constexpr,
                   BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = offs % 128
    x = offs // 128
    ho = x % 16
    x = x // 16
    t = x % T
    b = x // T
    group = ho // 4
    kh = ho % 4
    hi = kh * 4 + group

    oi = ((b * T + t) * 16 + hi) * 128 + v
    gi = (b * T + t) * 16 + hi
    chunk = t // 64
    token = t % 64
    qi = (((((b * NC + chunk) * 4 + kh) * 4 + group) * 64 + token) * 128 + v)
    o = tl.load(out + oi, mask=mask).to(tl.float32)
    z = tl.load(qh + qi, mask=mask).to(tl.float32)
    rg = tl.load(raw_g + gi, mask=mask)
    cg = tl.load(cum_g + gi, mask=mask)
    o += z * (tl.exp(rg) - tl.exp(cg)) * SCALE
    tl.store(result + offs, o, mask=mask)


@torch.no_grad()
def run(query, key, value, g, beta, scale):
    B, HK, T, K = query.shape
    H, V = value.shape[1], value.shape[-1]

    # The reference rounds the normalization result to bf16 before converting
    # it back to fp32.  Keep that exact placement of the rounding operation.
    qn = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    kn = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)

    # AITer's grouped-head kernels use contiguous groups, whereas the
    # reference maps value head h to key head h % HK.  Transposing the 4x4
    # head grid gives the same mapping without copying q and k four times.
    groups = H // HK
    q = torch.empty((B, T, HK, K), dtype=query.dtype, device=query.device)
    k = torch.empty_like(q)
    nq = q.numel()
    _transpose_qk_kernel[(triton.cdiv(nq, 256),)](
        qn, kn, q, k, nq, T, BLOCK=256
    )
    v = torch.empty((B, T, H, V), dtype=value.dtype, device=value.device)
    gate = torch.empty((B, T, H), dtype=torch.float32, device=g.device)
    bet = torch.empty_like(gate)
    raw_head = torch.empty((B, H, T), dtype=torch.float32, device=g.device)
    rows = B * T * H
    _prepare_vgb_kernel[(rows,)](
        value, g, beta, v, gate, bet, raw_head, rows, T, BLOCK=128
    )

    # A_raw and gc reproduce the reference's per-chunk cumulative gate and
    # lower-triangular K K^T matrix.  The second fused kernel performs the
    # triangular recurrence and produces the transformed key/value vectors.
    gc_head, A_raw = _fused_cumsum_kkt(k, bet, gate, use_exp2=False)
    w, u = _fused_solve(A_raw, k, v, bet, raw_head, use_exp2=False)

    # Unlike the conventional GDR wrapper, the specified recurrence uses the
    # raw gate (not its cumsum) when carrying state between chunks.
    h, v_new, _ = _chunk_h(
        k, w, u, g=gate, initial_state=None, output_final_state=False
    )

    gc = gc_head.transpose(1, 2).contiguous()
    out = _chunk_o(q, k, v_new, h, g=gc, scale=scale)

    # _chunk_o used gc for both terms.  Its intra-chunk term is already exact;
    # correct only q @ h from exp(gc) to the reference's exp(raw_gate).
    NC = (T + 63) // 64
    if T != NC * 64:
        qp = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, NC * 64 - T))
    else:
        qp = q
    qp = qp.reshape(B, NC, 64, HK, K).permute(0, 1, 3, 2, 4).unsqueeze(3)
    qh = torch.matmul(qp, h.reshape(B, NC, HK, groups, K, V))

    result = torch.empty((B, T, H, V), dtype=query.dtype, device=query.device)
    n = result.numel()
    _finish_kernel[(triton.cdiv(n, 256),)](
        out, qh, gate, gc, result, n, T, NC, scale, BLOCK=256
    )
    return result
