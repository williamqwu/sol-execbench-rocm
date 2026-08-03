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


def _stats(a, b, atol_floor: float):
    """(max_abs, max_rel) between two tensors, on the device they live on."""
    import torch

    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    diff = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.any()):
        return 0.0, 0.0
    diff = torch.where(finite, diff, torch.zeros_like(diff))
    max_abs = float(diff.max())
    rel = diff / torch.clamp(b.abs(), min=atol_floor)
    max_rel = float(torch.where(finite, rel, torch.zeros_like(rel)).max())
    return max_abs, max_rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--margin", type=float, default=1.25)
    ap.add_argument("--atol-floor", type=float, default=1e-8,
                    help="denominator floor for relative error, mirroring "
                         "compute_error_stats")
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
                # Run-to-run: same inputs, fresh allocations each seed, so any
                # difference is nondeterminism in the kernels themselves.
                outs = []
                for seed in range(a.seeds):
                    torch.manual_seed(seed)
                    inputs = prepare_inputs(definition, wl, ns)
                    with torch.no_grad():
                        out = run(*inputs)
                    outs.append([t.detach().clone() for t in _as_list(out)])
                    del inputs, out
                    torch.cuda.empty_cache()

                base = outs[0]
                max_abs = max_rel = 0.0
                for other in outs[1:]:
                    for x, y in zip(other, base):
                        da, dr = _stats(x, y, a.atol_floor)
                        max_abs, max_rel = max(max_abs, da), max(max_rel, dr)

                entry.update({
                    "run_to_run": {"max_abs": max_abs, "max_rel": max_rel},
                    "seeds": a.seeds,
                    "deterministic": max_abs == 0.0,
                })

                # Golden comparison: is AMD close to the MATH, not just to
                # itself? Recorded as explicitly not-applicable when absent,
                # never silently skipped.
                if golden is not None and wl.uuid in golden:
                    g = golden[wl.uuid]
                    ga = gr = 0.0
                    for x, y in zip(base, g["outputs"]):
                        da, dr = _stats(x.cpu(), y.to(torch.float64), a.atol_floor)
                        ga, gr = max(ga, da), max(gr, dr)
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
                eps = _dtype_floor(base)
                entry["tolerance"] = {
                    "max_atol": max(max_abs * a.margin, eps["atol"]),
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
    """Per-dtype tolerance floor: one ulp of the output dtype.

    A reference that is bit-exact across runs yields zero measured variance.
    Shipping a zero tolerance would then fail any submission that merely
    reassociates an accumulation -- correct kernels, rejected. The floor is
    derived from the dtype rather than chosen, so it cannot drift into being a
    number that makes a particular submission pass.
    """
    import torch

    dtype = tensors[0].dtype if tensors else torch.float32
    try:
        eps = float(torch.finfo(dtype).eps)
    except TypeError:                                # integer outputs
        return {"atol": 0.0, "rtol": 0.0}
    return {"atol": eps, "rtol": eps}


if __name__ == "__main__":
    raise SystemExit(main())
