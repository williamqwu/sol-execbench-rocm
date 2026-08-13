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

import _common  # noqa: E402
from _common import (  # noqa: E402
    INPUT_DEVICE,
    ROOT,
    exec_reference,
    golden_stamp_matches,
    load_problem,
    prepare_inputs,
    problem_key,
    run_guarded,
)

GOLDEN_DIR = ROOT / "artifacts" / "golden"


def golden_contract(key: str) -> dict | None:
    """The golden's input-draw stamp, or None for a golden that has no stamp.

    Read from the sidecar `<key>.meta.json`, never from the `.pt` — the goldens
    run to 143 GB and this has to be cheap enough to do unconditionally.

    Nothing here changes a derived tolerance. It only RECORDS whether the
    golden was drawn from the same generator this run draws from, because the
    alternative is what happened in STATE.md D53: a comparison against a
    different input draw, written into the artifact as `vs_golden`, indexed by
    nothing, read by no one.
    """
    import json

    side = GOLDEN_DIR / f"{key}.meta.json"
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text())
    except (OSError, ValueError):
        return None


def golden_comparability(key: str, golden_loaded: bool) -> tuple:
    """(comparable, stamp, note) for one problem's golden.

    A module-level function, not a few lines inside `main`, so that the
    predicate a test can reach is the predicate the artifact is written from.

    It applies THE SAME check the generator applies when deciding whether a
    golden may be reused (`gen_golden.is_cached` -> `golden_stamp_matches`):
    contract version AND device AND seed. A reader more lenient than its writer
    is D53 one level up — during a partial regeneration the writer redraws a
    stale golden while the reader stamps the very same file `comparable: true`.

    `comparable` additionally requires the `.pt` to have loaded. The goldens run
    to 143 GB and get deleted; the 1 KB sidecar survives, and a stamp with
    nothing under it must not read as a correctness check that was performed.

    Nothing here changes a derived tolerance (prime directive 7). It only
    records, per prime directive 8.
    """
    gc = golden_contract(key)
    comparable = golden_loaded and golden_stamp_matches(gc, INPUT_DEVICE)
    note = None
    if not comparable and (golden_loaded or gc is not None):
        if not golden_loaded:
            note = (
                f"sidecar {key}.meta.json is present but {key}.pt is not: "
                "there is no golden to compare against, comparable=false and "
                "vs_golden is null for every workload."
            )
        elif gc is None:
            note = (
                "golden has no input-draw stamp: it predates the D53 fix and "
                "was drawn on the CPU while this run draws on "
                f"{INPUT_DEVICE!r}. vs_golden is NOT a correctness signal."
            )
        else:
            note = (
                f"golden stamp {gc.get('contract_version')!r}/"
                f"{gc.get('input_device')!r}/seed {gc.get('seed')!r} does not "
                f"match this run's contract "
                f"{_common.GOLDEN_CONTRACT_VERSION!r}/{INPUT_DEVICE!r}/seed "
                f"{_common.GOLDEN_SEED!r}; the two are different random draws "
                "and vs_golden is NOT a correctness signal (STATE.md D53)."
            )
    return comparable, gc, note


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

        # Was the golden drawn from the generator THIS run draws from? Recorded
        # per prime directive 8, not acted on per prime directive 7 -- the
        # derivation below is untouched. An unstamped golden predates the D53
        # fix and its inputs are a different draw entirely; saying so in the
        # artifact is the difference between a check and the appearance of one.
        golden_comparable, gc, golden_note = golden_comparability(
            key, golden is not None)

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
                exact_max_abs = 0.0
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
                        if _is_exact(x):
                            # Measured, but kept OUT of the derived tolerance:
                            # a non-deterministic index is a real finding and
                            # must be recorded, and it must not buy the float
                            # outputs a wider band (D52, reverse direction).
                            exact_max_abs = max(exact_max_abs, _max_abs(x, y))
                        else:
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
                            if not _is_exact(x):
                                max_rel = max(max_rel, _max_rel(x, y, atol))
                        del inputs, out_a, out_b
                        torch.cuda.empty_cache()
                else:
                    for out_a, out_b in pairs:
                        for x, y in zip(out_a, out_b):
                            if not _is_exact(x):
                                max_rel = max(max_rel, _max_rel(x, y, atol))
                del pairs

                # Which outputs the derived band covers and which are held to
                # exact equality, recorded per output so a reader never has to
                # infer it from a dtype string in `_derivation`.
                exact_idx = [i for i, t in enumerate(base) if _is_exact(t)]
                by_dtype = {d["dtype"]: d for d in eps["per_dtype"]}
                entry["outputs"] = [
                    {"index": i,
                     "dtype": str(t.dtype),
                     "comparison": "exact" if _is_exact(t) else "tolerance",
                     # The floor this output's OWN dtype earns, next to the one
                     # it is actually judged by. Equal unless the problem
                     # returns more than one float dtype, in which case the
                     # difference is the whole of D52b.
                     "own_floor": (None if _is_exact(t)
                                   else {k: by_dtype[str(t.dtype)][k]
                                         for k in ("atol", "rtol", "rms")})}
                    for i, t in enumerate(base)
                ]

                entry.update({
                    "run_to_run": {
                        "max_abs": max_abs,
                        "max_rel": max_rel,
                        # Integer/bool outputs, measured separately and never
                        # folded into the band above.
                        "exact_outputs_max_abs": exact_max_abs,
                    },
                    "seeds": a.seeds,
                    "executions_per_seed": 2,
                    "deterministic": max_abs == 0.0 and exact_max_abs == 0.0,
                })

                # Golden comparison: is AMD close to the MATH, not just to
                # itself? Recorded as explicitly not-applicable when absent,
                # never silently skipped.
                if golden is not None and wl.uuid in golden:
                    g = golden[wl.uuid]
                    ga = gr = 0.0
                    g_exact_abs = 0.0
                    for x, y in zip(base, g["outputs"]):
                        yd = y.to(torch.float64)
                        if _is_exact(x):
                            # Same split as run-to-run (D52): an index that
                            # disagrees with the golden by 1 and a weight that
                            # disagrees by 1 are not the same finding, and
                            # maxing them into one number says neither.
                            g_exact_abs = max(g_exact_abs,
                                              _max_abs(x.cpu(), yd))
                            continue
                        ga = max(ga, _max_abs(x.cpu(), yd))
                        gr = max(gr, _max_rel(x.cpu(), yd, atol))
                    # `mode` decides how a disagreement reads: against a
                    # float64 golden it is a bug; against a native-dtype CPU
                    # golden it may be ordinary low-precision noise. Carried
                    # through so triage never has to guess which it was.
                    entry["vs_golden"] = {
                        "max_abs": ga, "max_rel": gr, "mode": g.get("mode"),
                        # Integer/boolean outputs, measured against the golden
                        # too but never folded into the two numbers above.
                        "exact_outputs_max_abs": g_exact_abs,
                        # Whether this number is evidence at all. Without it,
                        # a golden drawn from another generator reads exactly
                        # like a golden that disagrees.
                        "comparable": golden_comparable,
                        "input_device": (gc or {}).get("input_device"),
                        "not_comparable_because": golden_note,
                        # A reference that draws randomness inside `run` cannot
                        # be matched by a CPU golden even with the inputs
                        # aligned; gen_golden.py measures and stamps it.
                        "reference_draws_rng": g.get("reference_draws_rng"),
                    }
                else:
                    entry["vs_golden"] = None

                # The derived tolerance. Floors matter: a perfectly
                # deterministic reference measures 0 variance, and a zero
                # tolerance would fail every correct submission that reorders
                # a single accumulation. The floor is the dtype's own epsilon,
                # not a number chosen to make things pass.
                #
                # The floor names the FLOATING-POINT output dtype it came
                # from, not `base[0].dtype`: those differ exactly when the
                # first output is an index tensor, which is the case D52 got
                # wrong, and the old string ("floored at torch.int64 epsilon")
                # was the only place it showed. When a problem returns more
                # than one float dtype the string names the one whose floor is
                # APPLIED and says what that costs the others (D52b).
                floor_desc = _floor_desc(eps)
                entry["tolerance"] = {
                    "max_atol": atol,
                    "max_rtol": max(max_rel * a.margin, eps["rtol"]),
                    "required_matched_ratio": 0.99,
                    # The per-dtype floors this band was collapsed from, and
                    # the collapse factor. Kept because the applied number
                    # alone cannot say whether an fp32 output is being judged
                    # at bf16's epsilon.
                    "_dtype_floors": eps["per_dtype"],
                    "_floor_over_grant": {"atol": eps["over_grant_atol"],
                                          "rtol": eps["over_grant_rtol"]},
                    # Outputs the band above does NOT apply to. The workload
                    # schema carries one ToleranceSpec for the whole workload
                    # (`Workload.tolerance`), so this cannot be expressed as a
                    # per-output tolerance; the harness enforces it instead, by
                    # comparing integer and boolean outputs exactly whatever
                    # the spec says. Recorded here so the artifact states which
                    # outputs that covers rather than leaving it to be inferred.
                    "_exact_outputs": exact_idx,
                    "_derivation": (
                        f"max run-to-run error over {a.seeds} seeds x "
                        f"{a.margin} margin, {floor_desc}"
                        + (f"; outputs {exact_idx} are integer/boolean and are "
                           "compared for exact equality" if exact_idx else "")
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
            # `available` is not `usable`. Kept as two fields because the old
            # artifacts say only the first, and a reader who assumed it meant
            # the second is how D53 stayed unnoticed for a whole sweep.
            "golden_comparable": golden_comparable,
            "golden_contract": gc,
            "input_device": INPUT_DEVICE,
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


def _is_exact(t) -> bool:
    """True for outputs that must be compared for EXACT equality.

    Integer and boolean outputs -- indices, offsets, masks -- carry no
    representation error, so "within a tolerance" is not a meaningful
    relaxation for one. Zero is the right band for them, and the harness now
    applies exactly that (`compute_error_stats`, AMD: D52).
    """
    return not (t.is_floating_point() or t.is_complex())


def _tolerance_outputs(tensors) -> list:
    """The outputs a derived tolerance applies to: the floating-point ones."""
    return [t for t in tensors if not _is_exact(t)]


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

    **The floor is derived PER FLOAT DTYPE, not per problem (D52b).** A problem
    may return more than one floating-point dtype -- 17 of the 235 do, 16 of
    them scoreable, 396 workloads (counted from the 235 `definition.json`
    files, and every one of them has a bf16 output before its fp32 outputs).
    Deriving one floor from `tensors[0].dtype` gave every fp32 output bf16's
    epsilon, 0.0078125 against 1.1920929e-07: 65536x looser than the dtype can
    justify, and with a bit-exact reference (`max_abs == 0`) the floor IS the
    shipped band. Summing the RMS scale across dtypes was the same leak again,
    in the scale rather than the epsilon.

    So each dtype gets its own `{eps, RMS over that dtype's outputs only}`, and
    all of them are returned in `per_dtype`.

    **What is applied is the element-wise MAX over those floors, because the
    data model cannot carry more than one.** `Workload.tolerance` is a single
    `ToleranceSpec` (`src/sol_execbench/core/data/workload.py:117`) and
    `eval_driver` applies it to every output of the workload; there is no
    per-output tolerance to write a per-output floor into. Of the two ways to
    collapse the per-dtype floors into one number, only the max is safe: the
    min would hold a bf16 output to an fp32 floor, which is exactly the
    unpassable-by-construction failure D52 exists to remove. The max is
    permissive for the tighter dtype instead, which is a bound-quality problem
    (D39's class) rather than a correctness one -- and it is measured, not
    hidden: `over_grant_atol` / `over_grant_rtol` are the applied floor over
    the tightest per-dtype floor, so a reader sees exactly how much slack the
    single-spec limitation buys and on which dtype.

    **Integer and boolean outputs are excluded, and that is the whole of D52.**
    This used to read `torch.finfo(tensors[0].dtype)` inside a
    `except TypeError: return {"atol": 0.0, "rtol": 0.0}`, so a problem whose
    FIRST output happened to be an index tensor got a zero floor for its
    FLOAT outputs too -- and with a bit-exact reference (`max_abs == 0`) the
    shipped tolerance was exactly zero, i.e. bit-identity-with-eager. L2__049
    and Quant__011 both return `(int64 topk_idx, float32 topk_weight)` and
    were unpassable by construction: one fp32 ulp on 4488/16384 elements,
    `mr = 0.726`. The reverse leak was there too and is closed here as well --
    with a float output FIRST no TypeError was raised at all, and the integer
    output's magnitudes were then summed into the RMS scale, inflating the
    float tolerance by whatever an index happens to be worth.

    A tensor list with no floating-point output floors at zero, which is not
    the bug: for an all-integer problem zero IS exact equality, which is the
    comparison those outputs want.
    """
    import torch

    tensors = _tolerance_outputs(tensors)
    if not tensors:
        return {"atol": 0.0, "rtol": 0.0, "dtype": None, "per_dtype": [],
                "over_grant_atol": None, "over_grant_rtol": None}

    # One group per float dtype. Nothing crosses a group: neither the epsilon
    # nor the RMS scale, because mixing either is the D52 leak.
    groups: dict = {}
    for t in tensors:
        groups.setdefault(t.dtype, []).append(t)

    per_dtype = []
    for dtype, ts in groups.items():
        # No try/except: every dtype reaching here is floating-point or
        # complex, so `finfo` cannot raise. Swallowing a TypeError hid D52.
        eps = float(torch.finfo(dtype).eps)
        scale = _rms(ts)
        per_dtype.append({"dtype": str(dtype), "n_outputs": len(ts),
                          "rms": scale, "atol": eps * scale, "rtol": eps})
    per_dtype.sort(key=lambda d: d["dtype"])

    atol_src = max(per_dtype, key=lambda d: d["atol"])
    rtol_src = max(per_dtype, key=lambda d: d["rtol"])
    tightest_atol = min(d["atol"] for d in per_dtype)
    tightest_rtol = min(d["rtol"] for d in per_dtype)
    return {
        "atol": atol_src["atol"],
        "rtol": rtol_src["rtol"],
        # The dtype the APPLIED number came from. With one float dtype these
        # are the same string and say what they always said; with two they are
        # the only place the collapse is visible.
        "dtype": atol_src["dtype"],
        "rtol_dtype": rtol_src["dtype"],
        "per_dtype": per_dtype,
        # How much slack the single-ToleranceSpec collapse hands the tightest
        # dtype. 1.0 when there is only one, and 1.0 is the only value that
        # means "no output is held to another dtype's epsilon".
        "over_grant_atol": (atol_src["atol"] / tightest_atol
                            if tightest_atol > 0 else None),
        "over_grant_rtol": rtol_src["rtol"] / tightest_rtol,
    }


def _floor_desc(eps: dict) -> str:
    """The `_derivation` clause describing where the floor came from.

    A separate function because it is the only place a reader of
    `artifacts/05` learns that a problem with two float dtypes is judged at the
    wider one's epsilon, and a string that is built inline is a string nothing
    tests.
    """
    desc = (
        f"floored at {eps['dtype']} epsilon x output RMS"
        if eps["dtype"] is not None
        else "no floating-point output: exact equality"
    )
    if len(eps["per_dtype"]) <= 1:
        return desc
    others = ", ".join(f"{d['dtype']} {d['atol']:.6g}/{d['rtol']:.6g}"
                       for d in eps["per_dtype"])
    og_a = eps["over_grant_atol"]
    return desc + (
        f" (widest of {len(eps['per_dtype'])} per-dtype floors [{others}]; one "
        f"ToleranceSpec per workload, so the tightest dtype is over-granted "
        + (f"{og_a:.6g}x" if og_a is not None
           else "unboundedly (a dtype's RMS is 0)")
        + f" on atol and {eps['over_grant_rtol']:.6g}x on rtol)"
    )


def _rms(tensors) -> float:
    """RMS magnitude over *tensors*, ignoring non-finite elements.

    Deliberately NOT `t[torch.isfinite(t)]`. Boolean indexing is
    masked_select, and on ROCm 7.2 / torch 2.9.1 masked_select computes a
    garbage allocation size once the tensor has more than 2**32 elements: it
    asks for 16781313 GiB (2**54 + 2**42 + 2**30 bytes) and raises OOM on a
    GPU with 200 GiB free. Reproduced in isolation on a flat
    (2**32 + 1000)-element tensor. That is the whole of STATE.md D13, and it
    cost eight workloads their tolerance before it was found.

    `torch.where` over a bounded chunk computes the same sum of squares and
    never allocates a mask-sized output.
    """
    import torch

    total_sq, total_n = 0.0, 0
    for t in tensors:
        flat = t.detach().reshape(-1)
        for i in range(0, flat.numel(), _CHUNK):
            c = flat[i:i + _CHUNK].to(torch.float64)
            finite = torch.isfinite(c)
            c = torch.where(finite, c, torch.zeros_like(c))
            total_sq += float((c * c).sum())
            total_n += int(finite.sum())
    return math.sqrt(total_sq / total_n) if total_n else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
