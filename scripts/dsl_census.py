#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# NEW FILE, contributed from the AMDPilot v2 fleet side (issue amdpilotv2#19).
# Reads archived runs and writes a census. Scores nothing, rejects nothing,
# and touches no run's own artifacts.

"""Label an archived sweep against the four DSL rules, and write it down.

This is the thing that has to exist **before** anything in this repository is
allowed to reject a submission for being written in the wrong language. The
rules in `sol_execbench.core.bench.dsl_check` are label-only by design, and the
argument for keeping them that way is empirical: run them over the corpus we
already have, read the rows a human can check, and only then decide whether the
false-positive rate is low enough to gate on. A gate landed before this census
is a gate landed on a guess.

Two input shapes, and they are not equivalent:

* `artifacts/10/<run>/kernels/*.py` — one flat file per problem. Fast, complete,
  and **structurally unable to exercise the sibling-module case**: a submission
  whose Triton lives in `gn_triton.py` beside `kernel.py` arrives here as
  `kernel.py` alone and labels as if the kernel were not there. Every census
  taken from this shape must say so, and this one does, in the artifact.
* `artifacts/10/runs/<run>/<harness>/<problem>/packet/` — the whole submission,
  multiple files. This is the honest input and the only one whose `triton`
  column can be believed.

Usage:

    python3 scripts/dsl_census.py gpt56-220 glm-sweep-2 gpt56-180
    python3 scripts/dsl_census.py --packets runs/full-01/claude-code
    python3 scripts/dsl_census.py --out artifacts/10/scores/dsl-census.json ...

Stdlib only, so it runs on a laptop, in CI and inside the measurement image
without installing this package (which wants torch and pydantic).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_detector():
    """Import `dsl_check` without importing `sol_execbench`.

    The package's `__init__` pulls in pydantic and torch; this script needs
    neither, and a census that only runs where a GPU stack is installed is a
    census nobody runs.
    """
    path = ROOT / "src" / "sol_execbench" / "core" / "bench" / "dsl_check.py"
    spec = importlib.util.spec_from_file_location("dsl_check", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dsl_check"] = module            # dataclasses needs this
    spec.loader.exec_module(module)
    return module


def _commit() -> str:
    """The commit this census was taken at. Unknown stays unknown.

    Written into the artifact beside every number, because a census with no
    commit behind it cannot be cited: the rules move, and a row labelled by an
    earlier version of them is a different measurement.
    """
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _flat_submissions(run: str) -> dict[str, dict[str, str]]:
    """`{problem: {"kernel.py": source}}` from a run's flat kernels directory."""
    root = ROOT / "artifacts" / "10" / run / "kernels"
    if not root.is_dir():
        return {}
    return {path.stem: {"kernel.py": path.read_text(errors="replace")}
            for path in sorted(root.glob("*.py"))}


def _packet_submissions(relative: str) -> dict[str, dict[str, str]]:
    """`{problem: {file: source}}` from a run's multi-file packets."""
    root = ROOT / "artifacts" / "10" / relative
    if not root.is_dir():
        return {}
    out: dict[str, dict[str, str]] = {}
    for problem_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        packet = problem_dir / "packet"
        if not packet.is_dir():
            continue
        sources: dict[str, str] = {}
        for path in sorted(packet.rglob("*")):
            if path.is_file():
                sources[str(path.relative_to(packet))] = \
                    path.read_text(errors="replace")
        if sources:
            out[problem_dir.name] = sources
    return out


def census(name: str, submissions: dict[str, dict[str, str]],
           detector, shape: str) -> dict:
    """One run, labelled. Counts plus one row per problem."""
    rows = []
    counts = {dsl: 0 for dsl in detector.DSLS}
    counts["no_label"] = 0
    unreadable = 0
    for problem, sources in sorted(submissions.items()):
        seen = detector.dsl_labels(sources)
        for label in seen["labels"]:
            counts[label] += 1
        if not seen["labels"]:
            counts["no_label"] += 1
        if seen["unparsed"] or (seen["unread"] and not seen["read"]):
            unreadable += 1
        rows.append({
            "problem": problem,
            "labels": seen["labels"],
            "files": seen["read"],
            "unread": seen["unread"],
            "entry_point_found": seen["entry_point_found"],
            "unparsed": seen["unparsed"],
            "triton_why_not": seen["rules"]["triton"]["why_not"],
            "launch_forms": seen["rules"]["triton"]["launch_forms"],
        })
    return {
        "run": name,
        "shape": shape,
        "problems": len(rows),
        "counts": counts,
        # The census's own unknown. A problem whose sources could not be parsed,
        # or whose only staged file this reader does not parse, is NOT a problem
        # with no DSL in it -- and a rate computed over the whole denominator
        # would quietly say it was.
        "unreadable": unreadable,
        "caveats": ([
            "FLAT SHAPE: one kernel.py per problem. A submission whose kernel "
            "lives in a sibling module arrives here without it and its triton "
            "column is a FLOOR, not a count."] if shape == "flat" else []) + [
            "flydsl and assembly ship without a corpus behind them; see the "
            "dsl_check module docstring.",
            "This census assigns no verdict and rejects nothing."],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsl_census.py", description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+",
                        help="run names under artifacts/10/ (flat shape), or "
                             "paths under artifacts/10/ with --packets")
    parser.add_argument("--packets", action="store_true",
                        help="read multi-file packets rather than the flat "
                             "kernels/ directory. The honest shape.")
    parser.add_argument("--out", default=None,
                        help="write the whole census here as JSON")
    args = parser.parse_args(argv)

    detector = _load_detector()
    shape = "packet" if args.packets else "flat"
    reader = _packet_submissions if args.packets else _flat_submissions

    censuses = []
    for run in args.runs:
        submissions = reader(run)
        if not submissions:
            # Loud, and not a zero row. "I found nothing to read" and "I read a
            # run in which nothing was Triton" are different sentences.
            print(f"{run}: NOTHING READ — no submissions found for this run "
                  f"in the {shape} shape. Not counted.", file=sys.stderr)
            continue
        result = census(run, submissions, detector, shape)
        censuses.append(result)
        counts = result["counts"]
        print(f"{run}: {result['problems']} problems, shape={shape}  " +
              "  ".join(f"{k}={v}" for k, v in counts.items()) +
              f"  unreadable={result['unreadable']}")

    if not censuses:
        print("nothing was read, so nothing is written", file=sys.stderr)
        return 1

    payload = {
        "detector": "sol_execbench.core.bench.dsl_check::dsl_labels",
        "commit": _commit() or None,
        "label_only": True,
        "runs": censuses,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
