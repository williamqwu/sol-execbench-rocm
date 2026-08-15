#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Achievable bandwidth vs working-set size on an unlocked CDNA4 part.

Why this exists
---------------
``docs/issues/mi355x-bound-quality.md`` Issue 6 (judgement item V2): the SOLAR
arch config prices **all** traffic at the 8.0 TB/s DRAM peak, while the part has
a 256 MiB (268 MB) Infinity Cache whose ``SRAM_byte_per_cycle`` entry is
literally marked ``[PLACEHOLDER - verify]``.  Issue 2 rests on a hypothesis that
four FlashInfer workloads with a ~270 MB working set are running out of that
cache at an implied 8.70-10.09 TB/s.  Neither question can be decided without an
independent measurement of *what bandwidth is actually reachable, as a function
of working-set size, on this part*.

``scripts/roofline_probe.py --llc-sweep`` already sweeps working-set size, but
with one access pattern (``torch.Tensor.copy_``) and with no clock evidence.  A
single pattern hides the thing that matters -- the achievable ceiling is a
property of the pattern as much as of the size -- and on a part that is never
clock-locked an unbracketed number is unusable (Issue 6's own "a measurement
that never ran" trap).  So this probe:

* sweeps working-set size densely through the 256 MiB knee and well past it;
* runs **five** access patterns and reports each separately, never merged;
* brackets every single measurement with a GFX-clock sample on each side,
  through ``sol_execbench.core.bench.clock_bracket``, the same mechanism the
  T_b sweeps use, and refuses ``SOLEXBENCH_CLOCK_BASIS=locked`` on this part;
* reports the **maximum** over repetitions as well as the median, because a
  roofline needs the fastest the hardware could go, not the typical.

It reads a bandwidth number.  It does not decide a bandwidth model and it does
not touch the arch config -- prime directive 7.

Usage
-----
    SOLEXBENCH_CLOCK_BASIS=unlocked env/solb python scripts/llc_bandwidth_probe.py \
        --gpu 0 --out artifacts/03-MI355X/llc/llc-bandwidth-gpu0.json

Counting convention (stated because every bandwidth number is a ratio and the
denominator is where these go wrong):

* ``bytes_moved`` is the traffic the *kernel* performs -- what a roofline would
  charge it.  A copy is counted as 2 x buffer (one read + one write), a read as
  1 x buffer, a two-stream read as 2 x buffer.
* ``working_set_bytes`` is the distinct footprint that must live in cache for
  the next iteration to hit -- for a copy that is src + dst, i.e. also 2 x
  buffer, but for a different reason, and the two are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from provenance import write_artifact  # noqa: E402
from solexbench_rocm.parts import detect_part  # noqa: E402

MIB = 2 ** 20

# Sizes in MiB of the *primary* buffer. Dense either side of 128 and 256 so the
# knee is located for both the read patterns (working set = 1x buffer) and the
# copy/two-stream patterns (working set = 2x buffer), and out to 8 GiB so the
# far-field DRAM asymptote is measured rather than assumed.
DEFAULT_SIZES_MIB = [
    1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72, 80, 96,
    112, 128, 144, 160, 176, 192, 208, 224, 232, 240, 248, 256, 264, 272, 288,
    320, 352, 384, 448, 512, 640, 768, 1024, 1536, 2048, 4096, 8192,
]


def _coprime_rot(per_pass: int, total: int) -> int:
    """A rotation, in chunks, that cannot collapse the reuse distance.

    THIS IS THE WHOLE MEASUREMENT and getting it wrong reads as hardware.

    Each pass rotates the grid by `rot` chunks: program `pid` reads chunk
    `(pid + p*rot) mod N`. The obvious choice `rot = N / n_passes` divides N,
    so gcd(rot, N) = n_passes: the chunk space splits into `rot` orbits of
    `n_passes` chunks, and the `n_passes` programs that share an orbit spend
    the whole kernel cycling through the SAME few hundred KiB. That is an L1
    hit, not an LLC hit, and it inflates whichever sizes happen to divide
    evenly. Measured, on GPU 7, 64 MiB pure read, changing only
    `--min-launch-mib` (which changes n_passes and therefore rot, and nothing
    else): 9.35, 19.85, 24.16, 27.27 TB/s. A hardware bandwidth does not
    depend on that knob; a rotation artifact does.

    With gcd(rot, N) = 1 every program's orbit is the entire chunk space, each
    pass still touches every chunk exactly once, and a chunk is revisited only
    after a full pass -- which is the definition of "reuse distance = working
    set" this probe needs.
    """
    r = max(1, min(per_pass, total))
    for cand in range(r, min(r + 8192, total) + 1):
        if math.gcd(cand, total) == 1:
            return cand
    for cand in range(r, 0, -1):
        if math.gcd(cand, total) == 1:
            return cand
    return 1


def _sync(dev=None):
    import torch
    torch.cuda.synchronize(dev)


# --------------------------------------------------------------------------
# Kernels. Triton, because the ceiling this probe is looking for needs control
# over vector width and occupancy that ``Tensor.copy_`` does not expose.
# --------------------------------------------------------------------------
def _build_kernels():
    import triton
    import triton.language as tl

    # ------------------------------------------------------------------
    # Three traps are baked into these kernels. Each one was hit while
    # building this probe, each produced a plausible number, and removing
    # any of them silently restores a wrong curve.
    #
    # 1. LOOP-INVARIANT LOADS. N_PASSES exists so that one launch moves
    #    enough bytes that the back-to-back launch gap is not the
    #    measurement -- a small buffer read once is a ~10 us kernel. But a
    #    pass loop over the SAME addresses is loop-invariant, LLVM hoists
    #    the loads out of it, and the kernel reports exactly N_PASSES times
    #    the real bandwidth. Measured: 8 MiB read "116 TB/s", 256 MiB read
    #    "27.4 TB/s" against 7.29 TB/s for the identical single-pass
    #    kernel -- a ratio of 3.76 against N_PASSES=4.
    #
    # 2. REUSE DISTANCE. The first fix for (1) shifted each pass by two
    #    cache lines. That defeats the hoist and is still wrong, and it is
    #    the subtler error: program `pid` then re-read its OWN 16 KiB chunk
    #    N_PASSES times back to back, which is served out of its CU's L1.
    #    The curve it produced was ~35 TB/s FLAT from 8 MiB to 128 MiB --
    #    no dependence on working-set size at all, which is the signature
    #    of measuring the wrong cache. What the LLC question needs is a
    #    reuse distance equal to the WORKING SET: a line must not be
    #    revisited until the whole buffer has been walked. So each pass
    #    ROTATES the whole grid by `pass_rot` elements instead. Every pass
    #    still touches every line exactly once, in a different order, and a
    #    line is re-read only after a full buffer's worth of traffic.
    #    `pass_rot` is a multiple of BLOCK and the rotation is applied to a
    #    SCALAR base, so the modulo costs one integer op per 16 KiB rather
    #    than one per element.
    #
    # 3. THE REDUCTION'S OWN COST. The first version accumulated
    #    `acc += x.to(tl.float32)` over fp16 loads: two vector ops per
    #    2 bytes, i.e. 35e12 ops/s at the 35 TB/s it reported, against a
    #    v_add_f32 issue ceiling of 16384 lanes x 2.4 GHz = 39e12 ops/s.
    #    That kernel was ~89% ALU-bound and its "bandwidth" was the ALU's.
    #    XOR over int32 is one op per 4 bytes -- 8x cheaper per byte -- so
    #    what limits it is the memory system, which is what is being
    #    measured.
    # ------------------------------------------------------------------
    @triton.jit
    def k_read(src, out, stride_elems, span, pass_rot, N_PASSES: tl.constexpr,
               N_ITER: tl.constexpr, BLOCK: tl.constexpr):
        """Grid-stride pure read; one int32 partial per program."""
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for p in range(N_PASSES):
            for j in range(N_ITER):
                base = ((pid * BLOCK).to(tl.int64) + j * stride_elems
                        + p * pass_rot) % span
                acc ^= tl.load(src + base + lane)
        tl.store(out + pid, tl.sum(acc))

    @triton.jit
    def k_read2(src, src2, out, stride_elems, span, pass_rot,
                N_PASSES: tl.constexpr, N_ITER: tl.constexpr,
                BLOCK: tl.constexpr):
        """Two independent read streams -- more MLP, same footprint per byte."""
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for p in range(N_PASSES):
            for j in range(N_ITER):
                base = ((pid * BLOCK).to(tl.int64) + j * stride_elems
                        + p * pass_rot) % span
                acc ^= tl.load(src + base + lane)
                acc ^= tl.load(src2 + base + lane)
        tl.store(out + pid, tl.sum(acc))

    @triton.jit
    def k_copy(src, dst, stride_elems, span, pass_rot, N_PASSES: tl.constexpr,
               N_ITER: tl.constexpr, BLOCK: tl.constexpr):
        """Read + write, the pattern ``roofline_probe.py`` used."""
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        for p in range(N_PASSES):
            for j in range(N_ITER):
                base = ((pid * BLOCK).to(tl.int64) + j * stride_elems
                        + p * pass_rot) % span
                tl.store(dst + base + lane, tl.load(src + base + lane))

    @triton.jit
    def k_local(src, out, pass_shift, N_PASSES: tl.constexpr,
                BLOCK: tl.constexpr):
        """The OTHER extreme: each program re-reads its OWN chunk, N_PASSES
        times, shifted two cache lines each pass so nothing is hoisted.

        Reuse distance is one workgroup's chunk, not the working set, so this
        is served out of the CU's own L1/L2 and it is NOT an LLC measurement.
        It is here because "does any pattern reach 10.09 TB/s?" has to be asked
        of the most favourable pattern that exists, not only of the streaming
        one. Read it as the ceiling a kernel with PERFECT locality could see;
        the footprint that ceiling applies to is BLOCK*4 bytes per workgroup,
        not the buffer, and the two must never be quoted as one number.
        """
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for p in range(N_PASSES):
            acc ^= tl.load(src + (pid * BLOCK).to(tl.int64) + p * pass_shift
                           + lane)
        tl.store(out + pid, tl.sum(acc))

    # Strided and gather are the SAME kernel with a different granule-index
    # rule, so the two curves differ only in the thing under test. Both read
    # exactly the whole buffer per pass -- same footprint, same bytes, same
    # counting as ``read`` -- and differ only in the ORDER the granules are
    # visited. That is the comparison a roofline needs: order alone, holding
    # bytes fixed.
    #
    # int64 throughout: at 8 GiB the int32 form overflows -- granule index x
    # granule size is 2^31 exactly -- and the symptom is not a wrong number but
    # "Memory access fault by GPU node-2", which killed one authoritative sweep
    # at its last size.
    @triton.jit
    def k_granule(src, idx, out, n_gran, span_gran, pass_rot,
                  USE_IDX: tl.constexpr,
                  STEP: tl.constexpr, G: tl.constexpr, NG: tl.constexpr,
                  N_PASSES: tl.constexpr, N_ITER: tl.constexpr):
        pid = tl.program_id(0)
        npr = tl.num_programs(0)
        acc = tl.zeros((NG, G), dtype=tl.int32)
        cols = tl.arange(0, G).to(tl.int64)[None, :]
        rows = tl.arange(0, NG).to(tl.int64)
        for p in range(N_PASSES):
            for j in range(N_ITER):
                i0 = ((pid + j * npr).to(tl.int64) * NG
                      + p * pass_rot) % span_gran
                i = i0 + rows
                if USE_IDX:
                    g = tl.load(idx + i).to(tl.int64)
                else:
                    g = (i * STEP) % n_gran
                acc ^= tl.load(src + g[:, None] * G + cols)
        tl.store(out + pid, tl.sum(tl.sum(acc, axis=1), axis=0))

    return {"read": k_read, "read2": k_read2, "copy": k_copy,
            "local": k_local, "granule": k_granule}


# --------------------------------------------------------------------------
def _time_launcher(launch, reps: int, dev):
    """Median and min wall time of *launch*, on hip events. Seconds."""
    import torch
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        _sync(dev)
        ts.append(s.elapsed_time(e) / 1e3)
    return {"median_s": statistics.median(ts), "min_s": min(ts),
            "max_s": max(ts), "n": len(ts), "all_s": ts}


def _measure(pattern: str, kernels, mib: int, gpu: int, *, block: int,
             num_warps: int, reps: int, target_ms: float, page_kib: int,
             min_launch_mib: float):
    """One (pattern, size) point, clock-bracketed. Returns a record."""
    import torch
    import triton
    from sol_execbench.core.bench.clock_bracket import bracketed

    dev = torch.device(f"cuda:{gpu}")
    ESZ = 4                                   # int32 elements; see the XOR note
    n = mib * MIB // ESZ

    extra = {}
    if pattern in ("gather", "stride"):
        # Same footprint and same byte count as `read`; only the ORDER differs.
        g_elems = (page_kib * 1024 // ESZ) if pattern == "gather" else 32
        n_gran = n // g_elems
        ng = 1 if pattern == "gather" else 8      # granules per program per iter
        progs = max(1, min(65536, n_gran // ng))
        n_iter = max(1, n_gran // (progs * ng))
        covered_gran = progs * ng * n_iter
        pass_bytes = covered_gran * g_elems * ESZ
        n_passes = max(1, min(256, math.ceil(min_launch_mib * MIB / pass_bytes)))
        # Rotate a whole pass at a time, in units of NG granules, with a
        # rotation coprime to the chunk count -- see _coprime_rot.
        n_chunks = progs * n_iter
        pass_rot = _coprime_rot(max(1, n_chunks // max(n_passes, 1)),
                                n_chunks) * ng
        src = torch.randint(-2**31, 2**31 - 1, (n,), dtype=torch.int32,
                            device=dev)
        out = torch.zeros(progs, dtype=torch.int32, device=dev)
        if pattern == "gather":
            idx = torch.randperm(n_gran, device=dev).to(torch.int32)
            use_idx, step = True, 1
        else:
            # An odd multiplier coprime to a power-of-two granule count visits
            # every granule exactly once, so the footprint is the whole buffer.
            idx = torch.zeros(1, dtype=torch.int32, device=dev)
            use_idx, step = False, 33
        bytes_per_launch = covered_gran * g_elems * ESZ * n_passes
        ws_bytes = n * ESZ

        def launch():
            kernels["granule"][(progs,)](src, idx, out, n_gran,
                                         covered_gran, pass_rot,
                                         USE_IDX=use_idx, STEP=step,
                                         G=g_elems, NG=ng, N_PASSES=n_passes,
                                         N_ITER=n_iter, num_warps=num_warps)
        extra = {"granule_bytes": g_elems * ESZ, "n_granules": n_gran,
                 "granules_covered": covered_gran, "programs": progs,
                 "n_iter": n_iter, "n_passes": n_passes, "pass_rot": pass_rot,
                 "rot_chunks": pass_rot // ng, "n_chunks": progs * n_iter,
                 "granules_per_program": ng, "step": step, "elem_bytes": ESZ,
                 "coverage_frac": covered_gran / n_gran}
    elif pattern == "local":
        LOCAL_SHIFT = 128                      # 512 B, two cache lines
        progs = max(1, n // block)
        covered = progs * block
        n_passes = max(1, min(256,
                              math.ceil(min_launch_mib * MIB / (covered * ESZ))))
        src = torch.randint(-2**31, 2**31 - 1,
                            (covered + n_passes * LOCAL_SHIFT,),
                            dtype=torch.int32, device=dev)
        out = torch.zeros(progs, dtype=torch.int32, device=dev)
        bytes_per_launch = covered * ESZ * n_passes
        ws_bytes = covered * ESZ

        def launch():
            kernels["local"][(progs,)](src, out, LOCAL_SHIFT,
                                       N_PASSES=n_passes, BLOCK=block,
                                       num_warps=num_warps)
        extra = {"programs": progs, "n_passes": n_passes, "elem_bytes": ESZ,
                 "per_program_bytes": block * ESZ, "local_shift": LOCAL_SHIFT,
                 "covered_elems": covered,
                 "reuse_distance_bytes": block * ESZ,
                 "note": "reuse distance is the per-workgroup chunk, NOT the "
                         "working set; this is an L1/L2 ceiling"}
    else:
        # Grid-stride patterns: pick a program count, then N_ITER so that the
        # whole buffer is covered exactly once per pass.
        progs = triton.cdiv(n, block)
        n_iter = 1
        max_progs = 262144
        if progs > max_progs:
            n_iter = triton.cdiv(progs, max_progs)
            progs = triton.cdiv(n, block * n_iter)
        stride_elems = progs * block
        if stride_elems * n_iter > n:
            n_iter = max(1, n // stride_elems)
            if stride_elems * n_iter > n:
                progs = n // block
                stride_elems = progs * block
                n_iter = 1
        covered = stride_elems * n_iter        # == span, a multiple of BLOCK
        pass_bytes = covered * ESZ * (2 if pattern in ("read2", "copy") else 1)
        n_passes = max(1, min(256, math.ceil(min_launch_mib * MIB / pass_bytes)))
        # Rotation in elements: a coprime number of BLOCK-sized chunks, so
        # the base stays block-aligned, no lane straddles the span, and the
        # reuse distance cannot collapse -- see _coprime_rot.
        n_chunks = progs * n_iter
        pass_rot = _coprime_rot(max(1, n_chunks // max(n_passes, 1)),
                                n_chunks) * block
        src = torch.randint(-2**31, 2**31 - 1, (covered,), dtype=torch.int32,
                            device=dev)
        out = torch.zeros(progs, dtype=torch.int32, device=dev)
        if pattern == "read":
            bytes_per_launch = covered * ESZ * n_passes
            ws_bytes = covered * ESZ

            def launch():
                kernels["read"][(progs,)](src, out, stride_elems, covered,
                                          pass_rot, N_PASSES=n_passes,
                                          N_ITER=n_iter, BLOCK=block,
                                          num_warps=num_warps)
        elif pattern == "read2":
            src2 = torch.randint(-2**31, 2**31 - 1, (covered,),
                                 dtype=torch.int32, device=dev)
            bytes_per_launch = 2 * covered * ESZ * n_passes
            ws_bytes = 2 * covered * ESZ

            def launch():
                kernels["read2"][(progs,)](src, src2, out, stride_elems,
                                           covered, pass_rot,
                                           N_PASSES=n_passes, N_ITER=n_iter,
                                           BLOCK=block, num_warps=num_warps)
        elif pattern == "copy":
            dst = torch.empty_like(src)
            bytes_per_launch = 2 * covered * ESZ * n_passes
            ws_bytes = 2 * covered * ESZ

            def launch():
                kernels["copy"][(progs,)](src, dst, stride_elems, covered,
                                          pass_rot, N_PASSES=n_passes,
                                          N_ITER=n_iter, BLOCK=block,
                                          num_warps=num_warps)
        else:
            raise ValueError(pattern)
        extra = {"programs": progs, "n_iter": n_iter, "n_passes": n_passes,
                 "elem_bytes": ESZ, "pass_rot": pass_rot,
                 "rot_chunks": pass_rot // block, "n_chunks": progs * n_iter,
                 "covered_elems": covered}

    # Warm up (this is also what puts the working set in cache) and size the
    # timed window: too short and the launch tail dominates, which is exactly
    # what makes the 8 MiB point in roofline_probe.py read 2.4 TB/s.
    for _ in range(3):
        launch()
    _sync(dev)
    t1 = _time_launcher(launch, 3, dev)["median_s"]
    inner = max(1, min(2000, int(round((target_ms / 1e3) / max(t1, 1e-7)))))

    def window():
        for _ in range(inner):
            launch()
        _sync(dev)

    for _ in range(2):
        window()

    res, br = bracketed(window, device=dev, settle=launch, window_iters=inner)
    timing = _time_launcher(window, reps, dev)

    per_launch_median = timing["median_s"] / inner
    per_launch_min = timing["min_s"] / inner
    rec = {
        "pattern": pattern,
        "buffer_mib": mib,
        "buffer_bytes": mib * MIB,
        "working_set_bytes": ws_bytes,
        "working_set_mib": ws_bytes / MIB,
        "bytes_moved_per_launch": bytes_per_launch,
        "inner_launches_per_window": inner,
        "block": block,
        "num_warps": num_warps,
        "per_launch_median_s": per_launch_median,
        "per_launch_min_s": per_launch_min,
        "tbs_median": bytes_per_launch / per_launch_median / 1e12,
        "tbs_max": bytes_per_launch / per_launch_min / 1e12,
        "window_times_s": timing["all_s"],
        "clock": br.as_dict(),
        **extra,
    }
    del src, out
    torch.cuda.empty_cache()
    return rec


def harness_shaped(kernels, gpu: int, mib: int, *, block: int, num_warps: int,
                   warmup: int, rep: int, reps: int):
    """The arbiter for Issue 2, run through the repo's OWN timing path.

    Issue 2's four FlashInfer violations are all ``declared_traffic`` workloads
    whose declared bytes are ~50% input read and ~50% output write, and whose
    kernels beat an 8.0 TB/s bound at an implied 8.24-10.09 TB/s. A sweep of
    warm cache-resident bandwidth cannot settle that on its own, because a
    scored measurement is NOT warm: ``timing.py`` zeroes a
    ``flush_buffer_bytes`` (512 MiB) buffer before every timed iteration, and
    ``ShiftingMemoryPoolAllocator`` re-copies the inputs to a fresh offset
    before that. Both happen OUTSIDE the event pair.

    So this runs a kernel of exactly known traffic -- read ``mib``, write
    ``mib`` -- through ``time_runnable`` itself: same allocator, same flush,
    same event pair, same warmup/rep counts a score uses. Whatever bandwidth
    that reports is what the benchmark's own methodology admits, which is the
    only number the bound has to be a bound on.
    """
    import torch
    import triton
    from sol_execbench.core.bench.timing import time_runnable

    dev = torch.device(f"cuda:{gpu}")
    ESZ = 4
    n = mib * MIB // ESZ
    progs = min(262144, triton.cdiv(n, block))
    n_iter = max(1, n // (progs * block))
    stride_elems = progs * block
    covered = stride_elems * n_iter
    q = torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), dtype=torch.int32, device=dev)
    out = torch.empty_like(q)

    def fn(qq, oo):
        kernels["copy"][(progs,)](qq, oo, stride_elems, covered, block,
                                  N_PASSES=1, N_ITER=n_iter, BLOCK=block,
                                  num_warps=num_warps)

    runs = []
    for _ in range(reps):
        times = time_runnable(fn, inputs=[q], outputs=[out],
                              device=f"cuda:{gpu}", warmup=warmup, rep=rep,
                              return_mode="all")
        runs.append(times)
    flat = sorted(t for r in runs for t in r)
    bytes_moved = 2 * covered * ESZ
    med = statistics.median(flat)
    best = flat[0]
    return {
        "buffer_mib": mib,
        "bytes_moved": bytes_moved,
        "declared_traffic_mb": bytes_moved / 1e6,
        "warmup": warmup, "rep": rep, "reps": reps,
        "median_ms": med, "min_ms": best,
        "p10_ms": flat[max(0, int(0.10 * len(flat)) - 1)],
        "tbs_at_median": bytes_moved / (med / 1e3) / 1e12,
        "tbs_at_min": bytes_moved / (best / 1e3) / 1e12,
        "t_sol_ms_at_8tbs": bytes_moved / 8.0e12 * 1e3,
        "beats_8tbs_bound": (bytes_moved / (med / 1e3) / 1e12) > 8.0,
        "flush_buffer_bytes": __import__(
            "sol_execbench.core.bench.device", fromlist=["x"]
        ).flush_buffer_bytes(f"cuda:{gpu}"),
        "programs": progs, "n_iter": n_iter, "block": block,
        "num_warps": num_warps,
        "note": "read mib + write mib through time_runnable: production "
                "allocator, production LLC flush, production event pair",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patterns", default="read,read2,copy,stride,gather")
    ap.add_argument("--sizes-mib", default="")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--num-warps", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--target-ms", type=float, default=20.0)
    ap.add_argument("--page-kib", type=int, default=16)
    ap.add_argument("--min-launch-mib", type=float, default=1024.0,
                    help="minimum bytes one kernel launch moves, in MiB; "
                         "raised by re-reading the same footprint")
    ap.add_argument("--note", default="")
    ap.add_argument("--harness-shaped", default="",
                    help="comma-separated MiB sizes for the Issue-2 arbiter: "
                         "read N MiB + write N MiB through time_runnable")
    ap.add_argument("--hs-warmup", type=int, default=10)
    ap.add_argument("--hs-rep", type=int, default=50)
    a = ap.parse_args()

    import torch
    from sol_execbench.core.bench.clock_bracket import (
        checked_clock_basis, summarize_brackets,
    )

    dev_name = torch.cuda.get_device_name(a.gpu)
    basis = checked_clock_basis(dev_name)   # raises on a false 'locked' claim
    part = detect_part(a.gpu)
    kernels = _build_kernels()

    sizes = ([int(s) for s in a.sizes_mib.split(",") if s.strip()]
             if a.sizes_mib else DEFAULT_SIZES_MIB)
    patterns = [p.strip() for p in a.patterns.split(",") if p.strip()]

    points = []
    for pattern in patterns:
        for mib in sizes:
            try:
                rec = _measure(pattern, kernels, mib, a.gpu, block=a.block,
                               num_warps=a.num_warps, reps=a.reps,
                               target_ms=a.target_ms, page_kib=a.page_kib,
                               min_launch_mib=a.min_launch_mib)
            except Exception as e:                      # record, never guess
                rec = {"pattern": pattern, "buffer_mib": mib,
                       "error": f"{type(e).__name__}: {e}"}
            points.append(rec)
            if "error" in rec:
                print(f"{pattern:8s} {mib:6d} MiB  ERROR {rec['error'][:90]}",
                      flush=True)
            else:
                c = rec["clock"]
                print(f"{pattern:8s} {mib:6d} MiB  ws={rec['working_set_mib']:8.1f} MiB  "
                      f"med={rec['tbs_median']:6.3f}  max={rec['tbs_max']:6.3f} TB/s  "
                      f"clk={c.get('clock_mhz')} spread={c.get('clock_bracket_spread')}",
                      flush=True)

    hs = []
    for mib in [int(x) for x in a.harness_shaped.split(",") if x.strip()]:
        try:
            r = harness_shaped(kernels, a.gpu, mib, block=a.block,
                               num_warps=a.num_warps, warmup=a.hs_warmup,
                               rep=a.hs_rep, reps=a.reps)
        except Exception as e:
            r = {"buffer_mib": mib, "error": f"{type(e).__name__}: {e}"}
        hs.append(r)
        print(f"HARNESS-SHAPED {mib:6d} MiB in+out  "
              f"{r.get('declared_traffic_mb', float('nan')):.1f} MB declared  "
              f"med={r.get('median_ms')} ms  "
              f"{r.get('tbs_at_median', float('nan')):.3f} TB/s median / "
              f"{r.get('tbs_at_min', float('nan')):.3f} TB/s best", flush=True)

    ok = [p for p in points if "error" not in p]
    payload = {
        "gpu": a.gpu,
        "device_name": dev_name,
        "part": part.name,
        "clock_basis": basis,
        "note": a.note,
        "part_reference": {
            "dram_tbs_spec": part.dram_bytes_per_sec / 1e12,
            "llc_tbs_placeholder": part.llc_bytes_per_sec / 1e12,
            "llc_capacity_bytes": part.llc_capacity,
            "peak_freq_ghz": part.peak_freq_ghz,
        },
        "counting": {
            "bytes_moved": "traffic the kernel performs; copy=2x buffer, "
                           "read=1x buffer, read2=2x buffer, stride charged at "
                           "the 128 B line, gather at the page",
            "working_set": "distinct footprint that must be resident for the "
                           "next launch to hit; copy/read2 = 2x buffer",
        },
        "points": points,
        "harness_shaped": hs,
        "clock_summary": summarize_brackets([p.get("clock") for p in ok]),
        "ceiling": None,
    }
    if ok:
        best = max(ok, key=lambda p: p["tbs_max"])
        payload["ceiling"] = {
            "tbs_max": best["tbs_max"],
            "pattern": best["pattern"],
            "working_set_mib": best["working_set_mib"],
            "buffer_mib": best["buffer_mib"],
        }
    write_artifact(a.out, "llc-bandwidth-probe", payload,
                   extra_provenance={"part": part.name, "gpu": a.gpu,
                                     "clock_basis": basis})
    print(json.dumps(payload["ceiling"], indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
