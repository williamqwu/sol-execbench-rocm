#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the harness's authoritative timing see work on a non-default stream?

`bench_time_with_cuda_events` brackets each timed iteration with two events
recorded on the CURRENT stream:

    start_events[i].record()
    fn(args)
    end_events[i].record()

Work the submission enqueues on a stream of its own is not between them. The
function's own docstring states the contract -- "should only enqueue operations
onto the default stream" -- and nothing enforces it. `hip_events` is the ROCm
default and is the methodology every published score on this board was measured
with (`manifest.methodology == "hip_events"`).

`bench_gpu_time_with_rocprof` synchronizes before stamping the end of each
window precisely so that "work that lands late, or on another stream" stays
inside it, so it is the control: if the two disagree on a side-stream kernel and
agree on a default-stream one, the gap is the hole and not a methodology
difference.

Three probes, each measured three ways (hip_events / rocprof / host wall under a
full device synchronize):

  A  bypass      -- one GEMM, entirely on a side stream, host-synced inside fn
  A0 control     -- the same GEMM on the default stream
  B  L1__054     -- the submitted kernel, which puts its `value` GEMM on a
                    second stream for M >= 1024
  C  L1__054'    -- the same kernel with that one `with` block removed

No claim is made here about WHY a number came out as it did; the script emits
the three readings per probe and the arithmetic is done downstream.

    env/solb python scripts/bounds/side_stream_timing.py \
        --out artifacts/11/side-stream-timing-hole.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from provenance import write_artifact  # noqa: E402
from sol_execbench.core.bench.timing import (  # noqa: E402
    bench_gpu_time_with_rocprof,
    bench_time_with_cuda_events,
)

WARMUP = 10
REP = 50


def host_wall_ms(fn, args, warmup=WARMUP, rep=REP) -> float:
    """Median host wall time per iteration with a full device sync each side.

    Cannot miss a stream: it measures the wall clock across everything the
    device did between two barriers.
    """
    for _ in range(warmup):
        fn(args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(args)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def tracked_now() -> int:
    """Live submission-created streams, or 0 when the defense is not installed."""
    try:
        from sol_execbench.core.bench import streams
        return streams.tracked_count()
    except Exception:
        return 0


def measure(name: str, fn, args) -> dict:
    ev = statistics.median(
        bench_time_with_cuda_events(lambda a: fn(a), warmup=WARMUP, rep=REP,
                                    setup=lambda: args)
    )
    wall = host_wall_ms(fn, args)
    row = {
        "probe": name,
        "hip_events_ms": ev,
        "host_wall_ms": wall,
        "wall_over_events": wall / ev if ev > 0 else None,
    }
    try:
        rp = bench_gpu_time_with_rocprof(lambda a: fn(a), warmup=WARMUP, rep=REP,
                                         setup=lambda: args)
        row["rocprof_ms"] = statistics.median(rp)
        row["rocprof_over_events"] = row["rocprof_ms"] / ev if ev > 0 else None
    except Exception as exc:  # recorded, not swallowed
        row["rocprof_ms"] = None
        row["rocprof_error"] = f"{type(exc).__name__}: {exc}"
    return row


# ---------------------------------------------------------------- probe A / A0

def build_bypass(m=8192, n=8192, k=8192, dtype=torch.float32):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()

    def on_side(_):
        with torch.cuda.stream(stream):
            c = torch.mm(a, b)
        stream.synchronize()
        return c

    def on_default(_):
        return torch.mm(a, b)

    return on_side, on_default, {"m": m, "n": n, "k": k, "dtype": str(dtype)}


# ------------------------------------------------------------------ probe B / C

def build_l1_054(batch_size=16, seq_len=512):
    """The submitted kernel, and the same kernel with the side stream removed.

    Loaded from the artifact rather than retyped, so probe B is the thing that
    was scored. The variant is built by calling the same triton kernel from a
    single-stream `run`, not by editing the file.
    """
    kern_path = (ROOT / "artifacts/10/glm-sweep-2/kernels"
                 / "L1__054_audio_attention_qkv_projection_with_normalization.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("l1_054_submitted", kern_path)
    mod = importlib.util.module_from_spec(spec)
    # triton's @jit needs the source to be a real file on disk; loading by path
    # keeps it one, which exec'ing the text would not.
    spec.loader.exec_module(mod)

    hidden_size, num_heads, head_dim = 1024, 8, 128
    qkv = num_heads * head_dim
    dev, dt = "cuda", torch.float32
    hs = torch.randn(batch_size, seq_len, hidden_size, device=dev, dtype=dt)
    qw = torch.randn(qkv, hidden_size, device=dev, dtype=dt)
    kw = torch.randn(qkv, hidden_size, device=dev, dtype=dt)
    vw = torch.randn(qkv, hidden_size, device=dev, dtype=dt)
    qn = torch.randn(head_dim, device=dev, dtype=dt)
    kn = torch.randn(head_dim, device=dev, dtype=dt)
    eps = 1e-6

    def submitted(_):
        return mod.run(hs, qw, kw, vw, qn, kn, eps)

    def single_stream(_):
        """Identical work, one stream. Same triton kernel, same order."""
        M = batch_size * seq_len
        hidden_2d = hs.view(M, hidden_size)
        value = torch.mm(hidden_2d, vw.t())
        query = torch.mm(hidden_2d, qw.t())
        key = torch.mm(hidden_2d, kw.t())
        query = query.view(batch_size, seq_len, num_heads, head_dim)
        key = key.view(batch_size, seq_len, num_heads, head_dim)
        qs = torch.empty_like(query)
        ks = torch.empty_like(key)
        import triton
        mod._rms_norm_qk_kernel[(M * num_heads,)](
            qs, ks, query, key, qn, kn,
            M * num_heads, head_dim, eps,
            BLOCK_N=triton.next_power_of_2(head_dim), num_warps=4,
        )
        return qs, ks, value.view(batch_size, seq_len, num_heads, head_dim)

    join_stream = torch.cuda.Stream()

    def joined(_):
        """The same overlap, joined the way that keeps it inside the bracket.

        `current_stream().wait_stream(s)` puts the dependency in the DEFAULT
        stream, so the end event cannot execute until the side stream's work
        has finished. Host-side `s.synchronize()` blocks the host instead,
        which the deep launch queue absorbs. If this reads like C while
        `submitted` reads low, the difference is the join, not the overlap.
        """
        M = batch_size * seq_len
        hidden_2d = hs.view(M, hidden_size)
        join_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(join_stream):
            value = torch.mm(hidden_2d, vw.t())
        query = torch.mm(hidden_2d, qw.t())
        key = torch.mm(hidden_2d, kw.t())
        query = query.view(batch_size, seq_len, num_heads, head_dim)
        key = key.view(batch_size, seq_len, num_heads, head_dim)
        qs = torch.empty_like(query)
        ks = torch.empty_like(key)
        import triton
        mod._rms_norm_qk_kernel[(M * num_heads,)](
            qs, ks, query, key, qn, kn,
            M * num_heads, head_dim, eps,
            BLOCK_N=triton.next_power_of_2(head_dim), num_warps=4,
        )
        torch.cuda.current_stream().wait_stream(join_stream)
        return qs, ks, value.view(batch_size, seq_len, num_heads, head_dim)

    meta = {"batch_size": batch_size, "seq_len": seq_len, "M": batch_size * seq_len,
            "kernel": str(kern_path.relative_to(ROOT))}
    return submitted, single_stream, joined, meta


def build_deferred_sync(m=8192, n=8192, k=8192, dtype=torch.float32):
    """How far does this go? Sync the side stream at the START of the next call.

    Every result is still correct when the caller reads it, because nothing is
    read before the next call's barrier -- but no iteration's bracket contains
    the work of the iteration that enqueued it. This is the adversarial form of
    what `L1__054` does incidentally, and the severity of the finding is
    whichever of the two this turns out to resemble.
    """
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    held: list = []

    def deferred(_):
        if held:
            stream.synchronize()
            held.clear()
        with torch.cuda.stream(stream):
            c = torch.mm(a, b)
        held.append(c)
        return c

    return deferred, {"m": m, "n": n, "k": k, "dtype": str(dtype)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--install-tracking", action="store_true",
                    help="install the stream-join defense first, i.e. measure "
                         "what the eval driver will now do. Run both ways: the "
                         "before/after on the SAME probes is the evidence, and "
                         "the single-stream probes are the control that has to "
                         "not move.")
    args = ap.parse_args()

    if args.install_tracking:
        from sol_execbench.core.bench import streams
        streams.install()

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2

    rows = []

    # Order matters, and getting it wrong misreads the defense as a regression.
    # The tracked-stream set is per PROCESS, so once any probe has built a
    # stream the fence and join fire for every later probe -- including the
    # single-stream controls, which then appear to have moved when nothing
    # about them did. The eval driver runs one submission per process, so a
    # submission that creates no stream never takes that path.
    #
    # Controls first, while nothing is tracked. C runs twice: once clean, once
    # at the end, and the difference between the two is the cost of the fence
    # and join themselves rather than anything about the measurement.
    submitted, single, joined, k_meta = build_l1_054()
    rows.append({**measure("C_L1__054_single_stream_clean", single, None),
                 "shape": k_meta, "streams_tracked_before": tracked_now()})

    side, default, gemm_meta = build_bypass()
    rows.append({**measure("A0_control_default_stream", default, None),
                 "shape": gemm_meta, "streams_tracked_before": tracked_now()})
    rows.append({**measure("A_bypass_side_stream", side, None),
                 "shape": gemm_meta, "streams_tracked_before": tracked_now()})

    deferred, gemm_meta2 = build_deferred_sync()
    rows.append({**measure("D_deferred_sync_side_stream", deferred, None),
                 "shape": gemm_meta2, "streams_tracked_before": tracked_now()})

    rows.append({**measure("B_L1__054_submitted", submitted, None),
                 "shape": k_meta, "streams_tracked_before": tracked_now()})
    rows.append({**measure("E_L1__054_wait_stream_join", joined, None),
                 "shape": k_meta, "streams_tracked_before": tracked_now()})
    rows.append({**measure("C2_L1__054_single_stream_after", single, None),
                 "shape": k_meta, "streams_tracked_before": tracked_now()})

    doc = {
        "question": (
            "Is work enqueued on a non-default stream invisible to the "
            "hip_events timing every published score was measured with?"
        ),
        "methodology_of_record": "hip_events",
        "stream_join_defense_installed": bool(args.install_tracking),
        "warmup": WARMUP,
        "rep": REP,
        "probes": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(rows, indent=2))
    write_artifact(out, "11-side-stream-timing-hole", doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
