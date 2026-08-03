#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 05 runner — derive AMD tolerances for one problem's workloads.

Upstream's tolerances were calibrated by repeated reference probing **on
B200**. MFMA accumulation order, fast-math behaviour and SIMD width all shift
the empirical error distribution on CDNA4, and copying B200's numbers fails in
both directions at once: false failures on correct kernels, and — worse for
benchmark integrity — tolerances loose enough to reward kernels that are wrong
but fast.

Method (mirrors upstream's, so the numbers mean the same thing):

1. Run the reference many times, under perturbation (different seeds, fresh
   allocations, different launch order).
2. Record the empirical error distribution *between runs*.
3. Take the max observed error x 1.25.

Plus the check run-to-run comparison cannot make on its own:

4. Compare against a **float64 CPU golden**. Run-to-run variance says "AMD is
   self-consistent"; it says nothing about whether AMD is self-consistently
   WRONG. A deterministic wrong answer looks perfectly stable.

    python scripts/runners/calibrate_tolerance.py --problem <dir> --out <file>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    ROOT,
    exec_reference,
    load_problem,
    prepare_inputs,
    problem_key,
    run_guarded,
)

GOLDEN_DIR = ROOT / "artifacts" / "golden"


#: Elements per float64 comparison chunk. The comparison promotes to float64
#: and materializes a difference, so a whole-tensor comparison peaks at roughly
#: 4x the output's own size -- which is how a problem with an 18 GiB output
#: OOM'd a 252 GiB GPU while holding only two copies of it. Chunking caps that
#: overhead at a fixed ~2 GiB regardless of output size, and changes no result:
#: a maximum over chunks is the maximum.
_CHUNK = 1 << 26


def _chunks(a, b):
    """(flat a, flat b) chunk pairs, in float64, bounded in size."""
    a = a.detach().reshape(-1)
    b = b.detach().reshape(-1)
    import torch

    n = min(a.numel(), b.numel())
    for i in range(0, n, _CHUNK):
        yield (a[i:i + _CHUNK].to(torch.float64),
               b[i:i + _CHUNK].to(torch.float64))


def _max_abs(a, b) -> float:
    """Max absolute difference between two tensors, ignoring non-finite pairs."""
    import torch

    out = 0.0
    for x, y in _chunks(a, b):
        finite = torch.isfinite(x) & torch.isfinite(y)
        if not bool(finite.any()):
            continue
        diff = torch.where(finite, (x - y).abs(), torch.zeros_like(x))
        out = max(out, float(diff.max()))
    return out


def _max_rel(a, b, atol: float) -> float:
    """Max relative difference, with the SAME denominator floor the harness uses.

    `compute_error_stats` divides by ``clamp(|reference|, min=tolerance.max_atol)``
    -- the tolerance's own atol, not some small epsilon. Using a 1e-8 floor here
    instead produced max_rtol values around 3e8, because any element whose
    reference value is ~0 divides a real difference by ~0. A tolerance derived
    that way is not merely ugly: an rtol of 3e8 accepts literally any output,
    which is the exact failure mode task 05 exists to prevent -- tolerances
    loose enough to reward kernels that are wrong but fast.

    So atol is derived first, and rel is then measured against it, mirroring
    the formula the benchmark actually applies.
    """
    import torch

    out = 0.0
    for x, y in _chunks(a, b):
        finite = torch.isfinite(x) & torch.isfinite(y)
        if not bool(finite.any()):
            continue
        diff = torch.where(finite, (x - y).abs(), torch.zeros_like(x))
        rel = diff / torch.clamp(y.abs(), min=max(atol, 1e-12))
        out = max(out, float(torch.where(finite, rel, torch.zeros_like(rel)).max()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--margin", type=float, default=1.25)
    ap.add_argument("--atol-floor", type=float, default=1e-8,
                    help="denominator floor for relative error, mirroring "
                         "compute_error_stats")
    ap.add_argument("--low-memory", action="store_true",
                    help="never retain more than one seed's outputs; re-run "
                         "the seed loop to measure relative error once atol "
                         "is known. Same derivation, 2x the executions.")
    a = ap.parse_args()

    def body() -> dict:
        import torch

        definition, workloads = load_problem(a.problem)
        key = problem_key(a.problem)

        # Exec the reference once; every probe below calls this same function,
        # which is the point -- this measures the hardware's run-to-run
        # variance, not two different implementations.
        run, ns = exec_reference(definition)

        golden_path = GOLDEN_DIR / f"{key}.pt"
        golden = None
        if golden_path.exists():
            golden = torch.load(golden_path, map_location="cpu", weights_only=False)

        per_workload = []
        for wl in workloads:
            entry: dict = {"workload_uuid": wl.uuid, "axes": dict(wl.axes)}
            try:
                # Run-to-run variance, measured the only way that means
                # anything: TWO EXECUTIONS ON THE SAME INPUTS.
                #
                # The seed loop varies the input DATA, so that the error
                # distribution is sampled across the input space rather than
                # at one arbitrary draw. Within a seed the inputs are
                # identical and only the execution differs (fresh allocations,
                # a different point in the allocator's history, whatever
                # algorithm the library picks this time). Comparing outputs
                # ACROSS seeds would compare answers to different questions --
                # it reported max_abs ~9.8 and max_rel ~2.6e8 on a problem
                # whose actual run-to-run variance is at the last bit.
                first_outputs = None
                max_abs = 0.0
                pairs = []
                for seed in range(a.seeds):
                    torch.manual_seed(seed)
                    inputs = prepare_inputs(definition, wl, ns)
                    with torch.no_grad():
                        out_a = [t.detach().clone() for t in _as_list(run(*inputs))]
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        out_b = [t.detach().clone() for t in _as_list(run(*inputs))]
                    for x, y in zip(out_a, out_b):
                        max_abs = max(max_abs, _max_abs(x, y))
                    if first_outputs is None:
                        first_outputs = out_a
                    # Retaining every seed's outputs costs seeds x 2 x
                    # output_size of device memory, which OOM'd five problems
                    # whose outputs run to tens of GiB (234 GiB of 252 held at
                    # the point of failure). --low-memory keeps only seed 0's
                    # and pays for it with a second pass below.
                    if not a.low_memory:
                        pairs.append((out_a, out_b))
                    else:
                        del out_b
                        if seed:
                            del out_a
                    del inputs
                    torch.cuda.empty_cache()

                base = first_outputs or []
                # atol first, then rel measured against it -- the order the
                # harness's own error formula implies.
                eps = _dtype_floor(base)
                atol = max(max_abs * a.margin, eps["atol"])
                max_rel = 0.0
                if a.low_memory:
                    # Second pass: identical seeds, so identical inputs, and
                    # the same two-executions-per-seed comparison. The
                    # derivation is unchanged; only the memory profile is.
                    for seed in range(a.seeds):
                        torch.manual_seed(seed)
                        inputs = prepare_inputs(definition, wl, ns)
                        with torch.no_grad():
                            out_a = [t.detach().clone()
                                     for t in _as_list(run(*inputs))]
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            out_b = _as_list(run(*inputs))
                        for x, y in zip(out_a, out_b):
                            max_rel = max(max_rel, _max_rel(x, y, atol))
                        del inputs, out_a, out_b
                        torch.cuda.empty_cache()
                else:
                    for out_a, out_b in pairs:
                        for x, y in zip(out_a, out_b):
                            max_rel = max(max_rel, _max_rel(x, y, atol))
                del pairs

                entry.update({
                    "run_to_run": {"max_abs": max_abs, "max_rel": max_rel},
                    "seeds": a.seeds,
                    "executions_per_seed": 2,
                    "deterministic": max_abs == 0.0,
                })

                # Golden comparison: is AMD close to the MATH, not just to
                # itself? Recorded as explicitly not-applicable when absent,
                # never silently skipped.
                if golden is not None and wl.uuid in golden:
                    g = golden[wl.uuid]
                    ga = gr = 0.0
                    for x, y in zip(base, g["outputs"]):
                        yd = y.to(torch.float64)
                        ga = max(ga, _max_abs(x.cpu(), yd))
                        gr = max(gr, _max_rel(x.cpu(), yd, atol))
                    # `mode` decides how a disagreement reads: against a
                    # float64 golden it is a bug; against a native-dtype CPU
                    # golden it may be ordinary low-precision noise. Carried
                    # through so triage never has to guess which it was.
                    entry["vs_golden"] = {
                        "max_abs": ga, "max_rel": gr, "mode": g.get("mode"),
                    }
                else:
                    entry["vs_golden"] = None

                # The derived tolerance. Floors matter: a perfectly
                # deterministic reference measures 0 variance, and a zero
                # tolerance would fail every correct submission that reorders
                # a single accumulation. The floor is the dtype's own epsilon,
                # not a number chosen to make things pass.
                entry["tolerance"] = {
                    "max_atol": atol,
                    "max_rtol": max(max_rel * a.margin, eps["rtol"]),
                    "required_matched_ratio": 0.99,
                    "_derivation": (
                        f"max run-to-run error over {a.seeds} seeds x "
                        f"{a.margin} margin, floored at {base[0].dtype} epsilon"
                    ),
                }
                entry["ok"] = True
            except Exception as e:                  # noqa: BLE001
                entry.update({"ok": False,
                              "error": f"{type(e).__name__}: {e}"})
            per_workload.append(entry)

        return {
            "problem": key,
            "definition": definition.name,
            "margin": a.margin,
            "seeds": a.seeds,
            "low_memory": a.low_memory,
            "golden_available": golden is not None,
            "per_workload": per_workload,
            "n_ok": sum(1 for w in per_workload if w.get("ok")),
            "n_workloads": len(per_workload),
        }

    return run_guarded(a.out, "05-tolerances", body)


def _as_list(out):
    import torch

    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        return [v for v in out.values() if isinstance(v, torch.Tensor)]
    return [t for t in out if isinstance(t, torch.Tensor)]


def _dtype_floor(tensors) -> dict:
    """Tolerance floor: one ulp of the output dtype, AT THE OUTPUT'S SCALE.

    A reference that is bit-exact across runs yields zero measured variance.
    Shipping a zero tolerance would then fail any submission that merely
    reassociates an accumulation -- correct kernels, rejected. So there is a
    floor, and it is derived rather than chosen, so it cannot drift into being
    a number that makes a particular submission pass.

    The scale factor is the point. `eps` is a RELATIVE quantity: bf16's
    0.0078125 means "one ulp at magnitude 1". Using it directly as an
    *absolute* tolerance is a units error, and not a harmless one --

      outputs near 1000: one ulp is 7.8, and a fixed atol of 0.0078 is
                         far tighter than the dtype can even represent
      outputs near 0.001: one ulp is 7.6e-6, and a fixed atol of 0.0078
                          accepts a thousand-fold error

    The second case is the dangerous direction, and it is not hypothetical:
    against upstream's B200 values a fixed bf16 epsilon floor came out 781x
    looser on problems whose measured AMD variance was exactly zero. A
    tolerance that loose is what lets a kernel that is wrong but fast through,
    which is the specific failure task 05 exists to prevent.

    So the floor is one ulp **at the output's own scale**, and the scale used
    is the RMS magnitude, not the maximum. That choice matters:

      max|y|  is dominated by a single outlier. On a tensor spanning 1e-3 to
              1e11, one ulp of the max grants every small element absolute
              slack a thousand times its own value -- blanket permission to be
              wrong wherever the answer happens to be small.
      RMS|y|  is the typical element's magnitude, so the floor means "one ulp
              of a typical element". Elements above it are covered by the rtol
              term, which is proportional by construction.

    rtol floors at eps itself, which is already relative and needs no scaling.
    Together they reproduce the harness's own bound, `atol + rtol*|y|`, at
    about one ulp for a typical element.
    """
    import torch

    if not tensors:
        return {"atol": 0.0, "rtol": 0.0}
    dtype = tensors[0].dtype
    try:
        eps = float(torch.finfo(dtype).eps)
    except TypeError:                                # integer outputs
        return {"atol": 0.0, "rtol": 0.0}

    total_sq, total_n = 0.0, 0
    for t in tensors:
        finite = t[torch.isfinite(t)].to(torch.float64)
        if finite.numel():
            total_sq += float((finite * finite).sum())
            total_n += finite.numel()
    scale = math.sqrt(total_sq / total_n) if total_n else 0.0
    return {"atol": eps * scale, "rtol": eps}


if __name__ == "__main__":
    raise SystemExit(main())
