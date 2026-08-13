#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate float64 golden references.

Why these exist: run-to-run comparison on AMD tells you a kernel is *stable*,
not that it is *correct*. A deterministically wrong kernel looks perfectly
stable. Golden references separate two very different findings during task 05:

    AMD differs from NVIDIA  -> expected, benign, a numerics difference
    AMD differs from correct -> a bug

Runs each problem's reference in float64 on CPU, per workload, and stores the
outputs keyed by workload uuid — the same key task 05 compares against.

    HIP_VISIBLE_DEVICES=1 python scripts/gen_golden.py --out artifacts/golden

**The inputs are drawn on `_common.INPUT_DEVICE`, not on the CPU** — see
STATE.md D53 and the note on that constant. The arithmetic still happens in
float64 on the CPU, which is the whole point of a golden; only the *draw* moves.
That is not a detail:

    torch.manual_seed(0); torch.randn(n, device="cpu")     mt19937
    torch.manual_seed(0); torch.randn(n, device="cuda:0")  Philox4_32_10

are different numbers. The earlier version of this script drew on the CPU while
`calibrate_tolerance.py` drew on the GPU, so every golden answered a different
question from the measurement it was compared against — and the comparison was
recorded as `vs_golden` and never read, so nothing ever said so. 2302 of the
2331 recorded comparisons (98.756%, over 164 problems) exceed their own derived
atol as a result — recomputed 2026-08-12 from `artifacts/05/*.json`.

Cost, stated rather than hidden, because this changed:

* Input generation now needs a GPU. The reference execution does not, so the
  GPU is busy only for the draw. `--jobs` still defaults to 32: the expensive
  half of the work is float64-on-CPU reference execution, which no GPU is
  involved in, and the timing-card hazard is handled by pinning
  (`HIP_VISIBLE_DEVICES=<a card that is not 0>`), not by serialising. The
  per-worker HIP context footprint has NOT been measured on this part — see
  STATE.md 2026-08-12 D53 and TODO.md; if 32 concurrent contexts turn out not
  to fit, pass `--jobs` explicitly rather than guessing a new default.
* float64 on CPU is 30-100x slower than bf16 on the GPU, and the dataset
  contains GEMMs at n=28672. Workloads above ``--max-elements`` are SKIPPED and
  the skip is recorded per workload, so a missing golden is always visible as a
  decision. Silently dropping the big ones would leave exactly the
  compute-heavy problems -- the ones most likely to accumulate error --
  unchecked.

Every golden gets a sidecar ``<key>.meta.json`` stamping the input device and
seed it was drawn with. The resume/cache check reads that sidecar, so a golden
produced by the old CPU-drawing code is **not** treated as cached: it has no
sidecar, and it will be regenerated rather than silently reused.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

# The contract itself lives in `_common`, next to the device constant, because
# the READER of a golden (`calibrate_tolerance.py`) has to apply exactly the
# same predicate as the writer here. See `_common.golden_stamp_matches`.
# Re-exported under its historic name: this module is where the contract
# version was documented, and callers/tests reach for gen_golden.CONTRACT_VERSION.
from _common import (  # noqa: E402,F401  (scripts/runners is on sys.path)
    GOLDEN_CONTRACT_VERSION as CONTRACT_VERSION,
)
from _common import (  # noqa: E402
    GOLDEN_SEED,
    INPUT_DEVICE,
    golden_contract_stamp,
    golden_stamp_matches,
)


def sidecar_path(out_dir: Path, key: str) -> Path:
    return out_dir / f"{key}.meta.json"


def contract_stamp(input_device: str, seed: int = GOLDEN_SEED) -> dict:
    """The part of a golden's provenance that decides whether it is comparable."""
    return golden_contract_stamp(input_device, seed)


def is_cached(out_dir: Path, key: str, input_device: str) -> bool:
    """True only if the stored golden was drawn under the CURRENT contract.

    A bare `.pt` with no sidecar is a pre-D53 golden: it exists, it is large,
    and it is meaningless. Treating "the file is there" as "the work is done"
    is exactly how 143 GB of unusable goldens survived a resume.
    """
    if not (out_dir / f"{key}.pt").exists():
        return False
    side = sidecar_path(out_dir, key)
    if not side.exists():
        return False
    try:
        meta = json.loads(side.read_text())
    except (OSError, ValueError):
        return False
    return golden_stamp_matches(meta, input_device)


def _elements(definition, workload) -> int:
    """Total input+output elements — a cheap proxy for cost."""
    total = 0
    try:
        shapes = definition.get_input_shapes(workload.axes)
        for shape in shapes.values():
            n = 1
            for d in shape:
                n *= int(d)
            total += n
    except Exception:
        pass
    return total


def _to_cpu(inputs, torch):
    """Move a generated input list to the CPU, leaving scalars alone.

    The DRAW happened on the input device; the arithmetic happens here. Moving
    a tensor between devices is a copy, not a re-randomisation, so the values
    are bit-identical to the ones the tolerance path fed the GPU.
    """
    return [
        t.detach().to("cpu") if isinstance(t, torch.Tensor) else t
        for t in inputs
    ]


def _rng_fingerprint(torch) -> list:
    """Every generator a reference could plausibly draw from, right now.

    `torch.random.get_rng_state()` is the CPU generator ALONE. A reference that
    does `torch.randn(..., device="cuda")` inside `run` consumes no CPU RNG, so
    watching only the CPU state stamps it `reference_draws_rng: false` — which
    is precisely the misreading the flag exists to prevent. Device generators
    are included when CUDA/HIP is initialised; when it is not, the LENGTH of
    this list is itself the signal (see `_rng_changed`), because a reference
    that touches a device generator initialises it.
    """
    states = [torch.random.get_rng_state()]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        states.extend(torch.cuda.get_rng_state_all())
    return states


def _rng_changed(before: list, after: list, torch) -> bool:
    if len(before) != len(after):
        return True          # `run` initialised a device generator that was
                             # not there before it: it drew on that device.
    return any(not torch.equal(b, a) for b, a in zip(before, after))


def gen_one(args) -> tuple[str, dict]:
    problem, out_dir, max_elements, input_device = args
    key = f"{problem.parent.name}__{problem.name}"
    out_file = out_dir / f"{key}.pt"
    report: dict = {"problem": key, "workloads": {},
                    **contract_stamp(input_device)}
    if is_cached(out_dir, key, input_device):
        report["status"] = "cached"
        return key, report

    try:
        import torch

        from _common import exec_reference, load_problem, prepare_inputs

        torch.set_num_threads(1)      # one thread per worker; the pool is the
                                      # parallelism, and oversubscription here
                                      # makes the whole sweep slower.
        definition, workloads = load_problem(problem)
        run, ns = exec_reference(definition)

        goldens: dict = {}
        for wl in workloads:
            n = _elements(definition, wl)
            if max_elements and n > max_elements:
                report["workloads"][wl.uuid] = f"skipped: {n} elements > cap"
                continue
            # Two tiers, and WHICH ONE RAN IS RECORDED, because they are not
            # equally strong evidence:
            #
            #   float64    -- arithmetic ground truth. A disagreement is a bug.
            #   native_cpu -- same dtypes, CPU kernels. Still an independent
            #                 implementation with a different accumulation
            #                 order, so it catches a deterministically wrong
            #                 GPU kernel; but a disagreement here can also be
            #                 ordinary low-precision noise.
            #
            # The fallback exists because many references construct internal
            # tensors at a literal dtype (`torch.zeros(..., bfloat16)`, weights
            # made inside `run`), so promoting only the inputs mixes dtypes and
            # raises. Silently skipping those would drop the fused, multi-step
            # problems -- exactly the ones where error accumulates.
            for mode in ("float64", "native_cpu"):
                try:
                    # Seed and device BOTH match the tolerance path. Either one
                    # alone is not enough: the seed selects the stream, the
                    # device selects the engine, and the engines disagree.
                    torch.manual_seed(GOLDEN_SEED)
                    inputs = prepare_inputs(definition, wl, ns,
                                            device=input_device)
                    inputs = _to_cpu(inputs, torch)
                    if mode == "float64":
                        # Integer and boolean tensors are left alone:
                        # promoting an index tensor changes the program.
                        inputs = [
                            t.to(torch.float64)
                            if isinstance(t, torch.Tensor) and t.is_floating_point()
                            else t
                            for t in inputs
                        ]
                    # A reference that draws its OWN randomness inside `run`
                    # cannot be matched by a CPU golden at all -- `run` here
                    # consumes the CPU generator, `run` there consumes the
                    # device one. That is a residual of computing the truth on
                    # the CPU and it is not fixable by seeding; the only honest
                    # thing is to make it visible, so it is measured and
                    # stamped rather than left to look like agreement.
                    rng_before = _rng_fingerprint(torch)
                    with torch.no_grad():
                        out = run(*inputs)
                    drew_rng = _rng_changed(
                        rng_before, _rng_fingerprint(torch), torch)
                    if isinstance(out, torch.Tensor):
                        out = [out]
                    elif isinstance(out, dict):
                        out = list(out.values())
                    goldens[wl.uuid] = {
                        "mode": mode,
                        "input_device": input_device,
                        "seed": GOLDEN_SEED,
                        "reference_draws_rng": drew_rng,
                        "outputs": [
                            t.detach().cpu()
                            for t in out
                            if isinstance(t, torch.Tensor)
                        ],
                    }
                    report["workloads"][wl.uuid] = (
                        f"ok:{mode}" + (":reference_draws_rng" if drew_rng else "")
                    )
                    break
                except Exception as e:               # noqa: BLE001
                    report["workloads"][wl.uuid] = f"{type(e).__name__}: {e}"

        if goldens:
            out_dir.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, the same convention as `_common.write_result`
            # and for the same reason: a worker killed mid-save must not
            # overwrite a good `.pt` with a truncated one while the previous
            # run's matching sidecar is still on disk, which would read as
            # cached forever. `os.replace` is atomic within a filesystem, and
            # the temp file is created in `out_dir` to keep it on one.
            fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".pt.tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    torch.save(goldens, f)
                os.replace(tmp, out_file)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            # Sidecar LAST: it is the "this .pt is comparable" flag, and a
            # crash between the two must leave the golden looking unfinished,
            # not looking valid.
            sidecar_path(out_dir, key).write_text(json.dumps({
                **contract_stamp(input_device),
                "problem": key,
                "torch": torch.__version__,
                "n_workloads": len(workloads),
                "n_ok": len(goldens),
                "reference_draws_rng": sorted(
                    u for u, g in goldens.items() if g["reference_draws_rng"]),
            }, indent=1))
        report["status"] = "ok" if goldens else "empty"
        report["n_ok"] = sum(
            1 for v in report["workloads"].values() if str(v).startswith("ok")
        )
        report["n_workloads"] = len(workloads)
        return key, report
    except Exception as e:                           # noqa: BLE001
        report["status"] = "failed"
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()[-1500:]
        return key, report


def build_parser() -> argparse.ArgumentParser:
    """The CLI, built by a function so a test can read its defaults.

    `--input-device`'s default is the whole of D53. A test that reads that
    default out of the module it is checking proves nothing; it has to be able
    to get at the parser.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--out", default="artifacts/golden")
    # 32, unchanged from before the D53 fix. The draw moved onto a GPU, but the
    # expensive half of a worker is float64-on-CPU reference execution and the
    # timing-card hazard is a pinning question (HIP_VISIBLE_DEVICES), not a
    # concurrency one. The per-worker HIP context footprint is UNMEASURED on
    # this part -- no GPU was available to the session that asked -- so this
    # number is the status quo ante, not a claim. TODO.md records the debt.
    ap.add_argument("--jobs", type=int, default=32,
                    help="worker processes. Each one opens its own HIP context "
                         "on --input-device; that context's footprint has not "
                         "been measured on MI350X, so lower this if the draw "
                         "OOMs.")
    ap.add_argument("--input-device", default=INPUT_DEVICE,
                    help="device the INPUTS are drawn on. Must match the "
                         "device task 05 draws on (_common.INPUT_DEVICE = "
                         f"{INPUT_DEVICE!r}) or the golden is not comparable "
                         "to anything -- see STATE.md D53.")
    ap.add_argument("--max-elements", type=int, default=64_000_000,
                    help="skip workloads above this input-element count "
                         "(0 = no cap). Skips are recorded, never silent.")
    ap.add_argument("--category", nargs="+",
                    default=["L1", "L2", "Quant", "FlashInfer-Bench"])
    return ap


def main():
    a = build_parser().parse_args()

    jobs = a.jobs
    data, out = Path(a.data), Path(a.out)
    problems = [p for c in a.category for p in sorted((data / c).glob("*"))
                if (p / "definition.json").exists()]
    print(f"{len(problems)} problems, {jobs} workers, "
          f"inputs drawn on {a.input_device} at seed {GOLDEN_SEED}, "
          f"element cap {a.max_elements or 'none'}")
    if a.input_device != INPUT_DEVICE:
        print(f"  WARNING: task 05 draws on {INPUT_DEVICE!r}. Goldens drawn on "
              f"{a.input_device!r} answer a DIFFERENT input draw and their "
              f"`vs_golden` numbers mean nothing (STATE.md D53).")
    if not problems:
        sys.exit("no problems found — check --data")

    reports = {}
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for key, report in ex.map(
            gen_one, [(p, out, a.max_elements, a.input_device) for p in problems]
        ):
            reports[key] = report
            if report.get("status") in ("ok", "cached"):
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {key}: {report.get('error', report.get('status'))}")

    out.mkdir(parents=True, exist_ok=True)
    # A flat problem -> report mapping and NOTHING ELSE, the same shape it has
    # always had: TODO.md describes this file as "the report covers all 235",
    # and any reader that takes `len(...)` or iterates its keys as problems is
    # entitled to that. The contract is not lost by leaving it out -- every
    # per-problem report already carries `contract_stamp` inline (see
    # `gen_one`), which is also the only place it can differ per problem.
    (out / "_report.json").write_text(json.dumps(reports, indent=1))
    print(f"\n{ok} problems with goldens, {fail} without -> {out}")
    print("Missing goldens are not fatal; task 05 falls back to run-to-run "
          "comparison for those and records `vs_golden: null`. The per-workload "
          f"reasons are in {out}/_report.json.")


if __name__ == "__main__":
    main()
