import torch
import triton
import triton.language as tl


@triton.jit
def _rope_cos_sin_kernel(
    pos_ptr,          # *int64  [T]
    inv_ptr,          # *fp32   [HD2]
    out_ptr,          # *bf16   [T, HD2*4]
    T,                # number of tokens (batch_size * seq_len)
    scaling,          # fp32 scalar
    BLOCK_T: tl.constexpr,
    HD2: tl.constexpr,   # head_dim // 2
):
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t < T

    pos = tl.load(pos_ptr + t, mask=mask, other=0).to(tl.float32)

    j = tl.arange(0, HD2)
    inv = tl.load(inv_ptr + j)

    # freqs[t, j] = pos[t] * inv_freq[j] in fp32. The reference computes this as
    # a float32 matmul with inner dimension 1, i.e. exactly one multiply and no
    # accumulation, so a plain fp32 multiply reproduces its rounding bit for bit.
    f = pos[:, None] * inv[None, :]

    c = tl.cos(f) * scaling
    s = tl.sin(f) * scaling

    # emb = cat(freqs, freqs) followed by stack([cos, sin], -1) means the
    # flattened trailing dimension of each output row is
    #   [c0,s0, c1,s1, ..., c63,s63,  c0,s0, ..., c63,s63]
    # i.e. one interleaved cos/sin block of length 2*HD2, stored twice.
    cs = tl.interleave(c, s).to(tl.bfloat16)

    k = tl.arange(0, 2 * HD2)
    base = out_ptr + t[:, None] * (4 * HD2) + k[None, :]
    tl.store(base, cs, mask=mask[:, None])
    tl.store(base + 2 * HD2, cs, mask=mask[:, None])


# ---------------------------------------------------------------------------
# Why there is a fast launch path below
#
# For every shipped workload T = batch*seq lies in [256, 34624]. The kernel
# itself takes 3-9 us of GPU time, while Triton's normal Python dispatch costs
# ~10 us per call and the surrounding wrapper another ~4 us. Launch overhead,
# not memory traffic, dominates most of these shapes.
#
# `_Launcher` caches the compiled kernel from a first, ordinary JIT launch and
# afterwards calls its C launcher directly. Same compiled binary, same grid,
# same stream, same arguments -- only the per-call Python specialization and
# argument-binding work is skipped. If anything is unexpected (launch hooks
# installed, a launcher signature we do not recognise, a device we have not
# validated on), `ok` stays False and every call goes through the stock JIT
# path instead.
# ---------------------------------------------------------------------------

_BLOCK_T = 16
_HD2_SUPPORTED = 64          # head_dim = 128, the only value the axes allow


class _Launcher:
    __slots__ = ("run", "func", "meta", "ok")

    def __init__(self, pos, inv, out, T, scaling, hd2):
        self.ok = False
        # Refuse the direct path if any launch hook is registered, so we never
        # bypass profiler/harness instrumentation. In this Triton the knob is a
        # HookChain whose `.calls` list is empty when nothing is installed
        # (the object itself is always truthy, so test the list).
        try:
            from triton import knobs
            for _knob in (knobs.runtime.launch_enter_hook,
                          knobs.runtime.launch_exit_hook):
                if _knob is None:
                    continue
                calls = getattr(_knob, "calls", _knob)
                if calls:
                    return
        except Exception:
            return
        try:
            grid = ((T + _BLOCK_T - 1) // _BLOCK_T,)
            _rope_cos_sin_kernel[grid](
                pos, inv, out, T, scaling,
                BLOCK_T=_BLOCK_T, HD2=hd2, num_warps=4, num_stages=1,
            )
            cache = _rope_cos_sin_kernel.device_caches[pos.device.index][0]
            ck = None
            for v in cache.values():
                ck = v
            if ck is None:
                return
            self.run = ck.run
            self.func = ck.function
            self.meta = ck.packed_metadata

            # Validate the direct path against the JIT path before trusting it.
            ref = torch.empty_like(out)
            _rope_cos_sin_kernel[grid](
                pos, inv, ref, T, scaling,
                BLOCK_T=_BLOCK_T, HD2=hd2, num_warps=4, num_stages=1,
            )
            probe = torch.empty_like(out)
            self.run(
                grid[0], 1, 1, torch.cuda.current_stream().cuda_stream,
                self.func, self.meta, None, None, None,
                pos, inv, probe, T, scaling, None, None,
            )
            torch.cuda.synchronize()
            self.ok = bool(torch.equal(probe, ref))
        except Exception:
            self.ok = False


_launcher = None          # type: _Launcher | None
_launcher_dev = -1

_empty = torch.empty
_bf16 = torch.bfloat16

# `torch.cuda.current_stream().cuda_stream` costs ~2.4 us of pure Python per
# call -- comparable to the kernel itself. `_cuda_getCurrentRawStream` is the
# same handle (verified equal) via a single C call at ~0.06 us; it is the
# accessor torch.compile's own generated code uses. Fall back if absent.
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream
    if _raw_stream(0) != torch.cuda.current_stream().cuda_stream:
        raise RuntimeError
except Exception:                                    # pragma: no cover
    def _raw_stream(idx):
        return torch.cuda.current_stream().cuda_stream


def _fallback(position_ids, inv_freq, scaling):
    pos = position_ids.float()
    freqs = pos[:, :, None] * inv_freq[None, None, :].float()
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.stack(
        [emb.cos() * scaling, emb.sin() * scaling], dim=-1
    ).to(torch.bfloat16)


def run(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    """
    Fused construction of the RoPE cos/sin embedding table.

    position_ids : (batch_size, seq_len) int64
    inv_freq     : (head_dim // 2,) float32
    returns      : (batch_size, seq_len, head_dim, 2) bfloat16

    Single pass over the output: no (batch, hd2, seq) matmul, no `cat`, no
    `stack`, and no fp32 intermediate written to global memory. Each output
    element is computed once and stored once, as bf16.
    """
    global _launcher, _launcher_dev

    batch_size, seq_len = position_ids.shape
    hd2 = inv_freq.shape[0]
    dev = position_ids.device
    T = batch_size * seq_len

    out = _empty((batch_size, seq_len, hd2 * 2, 2), dtype=_bf16, device=dev)

    L = _launcher
    if (
        L is not None
        and L.ok
        and hd2 == _HD2_SUPPORTED
        and dev.index == _launcher_dev
        and T > 0
    ):
        # Hot path: everything below was validated on the first call.
        L.run(
            (T + _BLOCK_T - 1) // _BLOCK_T, 1, 1,
            _raw_stream(_launcher_dev),
            L.func, L.meta, None, None, None,
            position_ids, inv_freq, out, T, attention_scaling, None, None,
        )
        return out

    # ---- cold / unusual path -------------------------------------------
    with torch.no_grad():
        if T == 0:
            return out

        if not position_ids.is_contiguous():
            position_ids = position_ids.contiguous()
        if not inv_freq.is_contiguous():
            inv_freq = inv_freq.contiguous()
        if inv_freq.dtype != torch.float32:
            inv_freq = inv_freq.float()

        scaling = float(attention_scaling)

        # tl.arange requires power-of-two extents.
        if (hd2 & (hd2 - 1)) != 0 or position_ids.dtype != torch.int64:
            return _fallback(position_ids, inv_freq, scaling)

        grid = ((T + _BLOCK_T - 1) // _BLOCK_T,)

        if hd2 == _HD2_SUPPORTED and (L is None or dev.index != _launcher_dev):
            cand = _Launcher(position_ids, inv_freq, out, T, scaling, hd2)
            if cand.ok:
                _launcher = cand
                _launcher_dev = dev.index
                cand.run(
                    grid[0], 1, 1, torch.cuda.current_stream().cuda_stream,
                    cand.func, cand.meta, None, None, None,
                    position_ids, inv_freq, out, T, scaling, None, None,
                )
                return out

        _rope_cos_sin_kernel[grid](
            position_ids, inv_freq, out, T, scaling,
            BLOCK_T=_BLOCK_T, HD2=hd2, num_warps=4, num_stages=1,
        )
        return out
