import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(float("-inf"))

# Problem constants (definition.json declares these const, not symbolic).
N_GROUP = 8
TOPK_GROUP = 4
EXPERTS_PER_GROUP = 32
NUM_EXPERTS = 256


@triton.jit
def _moe_group_mask_kernel(
    scores_ptr,   # *f32, (T, 256)
    out_ptr,      # *f32, (T, 256)
    gm_ptr,       # *f32, (T, 8)
    num_tokens,
    BLOCK_T: tl.constexpr,
    STREAM: tl.constexpr,
):
    """One fused streaming pass: read each token's 256 scores once, keep them in
    registers through both reductions, write the masked row and the group mask.

    The reference materialises topk values + indices, a zeros tensor, a scatter,
    an expand and a masked_fill -- about six trips over the data. This is one.
    """
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_tokens

    g = tl.arange(0, 8)
    e = tl.arange(0, 32)

    # (BLOCK_T, 8, 32). The (group, expert) pair is contiguous, so a token's
    # 256-wide row is one unbroken run of memory and loads coalesce fully.
    off = t[:, None, None] * 256 + g[None, :, None] * 32 + e[None, None, :]
    # Every byte of `scores` is read exactly once and every byte of the output
    # is written once and never re-read, so caching either only evicts something
    # useful. On the large shapes -- the only ones that are actually
    # bandwidth-bound rather than launch-bound -- streaming hints are worth
    # 15-20%. On small shapes the working set fits anyway and the hints are a
    # wash, so they are only requested above the threshold in _pick.
    if STREAM:
        x = tl.load(scores_ptr + off, mask=tmask[:, None, None], other=NEG_INF,
                    cache_modifier=".cg")
    else:
        x = tl.load(scores_ptr + off, mask=tmask[:, None, None], other=NEG_INF)

    # --- sum of the top-2 within each group ----------------------------------
    m1, i1 = tl.max(x, axis=2, return_indices=True)
    # Suppress exactly one occurrence, chosen *by index* rather than by value.
    # This is what makes duplicates correct: when a group's two largest scores
    # are equal, masking by value would erase both and the runner-up would be
    # wrong.
    x2 = tl.where(e[None, None, :] == i1[:, :, None], NEG_INF, x)
    m2 = tl.max(x2, axis=2)
    # m1 + m2 reproduces torch.topk(k=2)[0].sum(-1) exactly: topk returns the
    # pair in descending order and the reference sums that same pair, so both
    # perform the identical single float32 rounding.
    gs = m1 + m2

    # --- top-4 of the 8 group scores -----------------------------------------
    # rank[i] = #{ j : gs[j] > gs[i], or gs[j] == gs[i] and j < i }
    # Selected iff rank < 4. Exactly four groups qualify even under ties,
    # because folding in the index comparison makes the order strict and total.
    # Ties resolve toward the lower index, matching torch.topk.
    a = gs[:, :, None]
    b = gs[:, None, :]
    beats = (b > a) | ((b == a) & (g[None, None, :] < g[None, :, None]))
    sel = tl.sum(beats.to(tl.int32), axis=2) < 4

    tl.store(
        gm_ptr + t[:, None] * 8 + g[None, :],
        tl.where(sel, 1.0, 0.0),
        mask=tmask[:, None],
    )
    out = tl.where(sel[:, :, None], x, NEG_INF)
    if STREAM:
        tl.store(out_ptr + off, out, mask=tmask[:, None, None], cache_modifier=".cs")
    else:
        tl.store(out_ptr + off, out, mask=tmask[:, None, None])


def _pick(num_tokens: int):
    """(BLOCK_T, num_warps, STREAM), swept per shape under the *cold-cache*
    regime the benchmark actually measures in.

    This distinction turned out to matter more than any config detail. Timed
    warm, in a tight loop, the input stays resident and the whole problem looks
    launch-bound; timed the way it is scored -- last-level cache flushed before
    every iteration -- the large shapes are genuinely reading from HBM, and
    streaming cache hints are worth 15-20% there. Below ~4k tokens the working
    set is small enough that the hints make no difference, so they are off.
    """
    if num_tokens <= 1536:
        return 1, 1, False
    if num_tokens <= 3072:
        return 2, 2, False
    if num_tokens <= 6144:
        return 4, 4, True
    if num_tokens <= 12288:
        return 4, 4, True
    if num_tokens <= 24576:
        return 4, 2, True
    return 8, 1, True


# Ready-to-call launchers keyed by (block_t, num_warps, device_index).
#
# Triton's ordinary `kernel[grid](...)` path re-derives the argument signature
# and specialization on every call: ~11us, which is several times the GPU work
# for most of these shapes (14 of the 16 workloads are launch-bound, not
# bandwidth-bound). The compiled binary does not depend on any of that work, so
# we keep it and invoke its launcher directly, ~4us. This is purely a dispatch
# shortcut -- same kernel, same arguments, and the stream is still read per call
# so stream semantics are unchanged. If anything about the internals differs
# from what we expect, we fall back to the stock path.
_LAUNCHERS = {}


def _build_launcher(block_t, num_warps, stream_hint, scores, out, gm, num_tokens, nblocks):
    # Compile (and implicitly validate) through the normal JIT path once.
    _moe_group_mask_kernel[(nblocks,)](
        scores, out, gm, num_tokens,
        BLOCK_T=block_t, STREAM=stream_hint, num_warps=num_warps,
    )

    def fallback(sp, op, gp, n, nb, _bt=block_t, _w=num_warps, _s=stream_hint):
        _moe_group_mask_kernel[(nb,)](
            sp, op, gp, n, BLOCK_T=_bt, STREAM=_s, num_warps=_w
        )

    try:
        from triton.runtime import driver

        # Deterministic probe data, built with arithmetic rather than
        # torch.rand so the global RNG stream is untouched (the harness seeds
        # it). Compile the probe shape through the stock path *first*, so that
        # cache lookup below happens against a settled cache.
        pn = 133
        idx = torch.arange(pn * 256, device=scores.device, dtype=torch.float32)
        probe = ((idx * 2654435761.0) % 1009.0 / 1009.0).view(pn, 256).to(scores.dtype)
        # Zero-fill: the tail rows of the last block are masked off and never
        # written, so comparing raw `empty` memory there would compare
        # allocator garbage rather than kernel output.
        ref_o = torch.zeros_like(probe)
        ref_g = torch.zeros((pn, 8), device=scores.device, dtype=scores.dtype)
        got_o = torch.zeros_like(probe)
        got_g = torch.zeros((pn, 8), device=scores.device, dtype=scores.dtype)
        pnb = (pn + block_t - 1) // block_t
        fallback(probe, ref_o, ref_g, pn, pnb)
        torch.cuda.synchronize()

        # On this Triton build the device-cache key is a *string* and
        # src.constexprs is None, so match the constexpr signature textually
        # and confirm num_warps through metadata. Anchor on the closing bracket
        # so BLOCK_T=1 cannot substring-match the tail of BLOCK_T=16, and so the
        # STREAM bool has to match exactly.
        want = "('constexpr', %d), ('constexpr', %s)]" % (block_t, bool(stream_hint))
        cands = []
        for entry in _moe_group_mask_kernel.device_caches.values():
            for ckey, c in entry[0].items():
                md = getattr(c, "metadata", None)
                if md is None or getattr(md, "num_warps", None) != num_warps:
                    continue
                if want in str(ckey):
                    cands.append(c)
        if not cands:
            return fallback

        get_stream = driver.active.get_current_stream
        get_device = driver.active.get_current_device

        # Several entries can share these constexprs: Triton also specializes on
        # pointer alignment/divisibility, so one source yields more than one
        # binary. Rather than guess which matches the tensors we actually pass,
        # try each and keep the first that reproduces the stock JIT path
        # bitwise. Pulling a compiled binary out of a private cache is the one
        # genuinely fragile step here, so it is verified rather than trusted --
        # a wrong match would be silent and would produce wrong numbers.
        for c in cands:
            try:
                c._init_handles()
                run, func, packed = c.run, c.function, c.packed_metadata

                def launch(sp, op, gp, n, nb, _r=run, _f=func, _p=packed):
                    _r(nb, 1, 1, get_stream(get_device()), _f, _p,
                       None, None, None, sp, op, gp, n, block_t, stream_hint)

                got_o.zero_()
                got_g.zero_()
                launch(probe, got_o, got_g, pn, pnb)
                torch.cuda.synchronize()
                if torch.equal(got_o, ref_o) and torch.equal(got_g, ref_g):
                    return launch
            except Exception:
                continue
        return fallback
    except Exception:
        return fallback


def run(scores: torch.Tensor):
    """Group-based score aggregation and masking for MoE routing.

    scores: (num_tokens, 256) float32
    returns (masked_scores (num_tokens, 256), group_mask (num_tokens, 8))
    """
    num_tokens = scores.shape[0]

    if not scores.is_contiguous():
        scores = scores.contiguous()

    masked_scores = torch.empty_like(scores)
    group_mask = scores.new_empty((num_tokens, 8))

    if num_tokens == 0:
        return masked_scores, group_mask

    block_t, num_warps, stream_hint = _pick(num_tokens)
    nblocks = (num_tokens + block_t - 1) // block_t
    key = (block_t, num_warps, stream_hint, scores.device.index)

    launch = _LAUNCHERS.get(key)
    if launch is None:
        launch = _build_launcher(
            block_t, num_warps, stream_hint,
            scores, masked_scores, group_mask, num_tokens, nblocks,
        )
        _LAUNCHERS[key] = launch

    launch(scores, masked_scores, group_mask, num_tokens, nblocks)
    return masked_scores, group_mask
