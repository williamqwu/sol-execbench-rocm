#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the dataset's *descriptive* metadata into one tracked file.

    python scripts/export_dataset_meta.py     # -> reference/dataset-meta.json

WHY THIS EXISTS
---------------
`data/` is gitignored and does not travel with the repo (CLAUDE.md §8), which
is right for the dataset as a whole: it is 7.5 MB of upstream distribution that
`scripts/materialize_dataset.py` can rebuild from the Hub at any time, and this
repo is not its publisher.

But the *board* needs a small part of it, and only to describe things: what a
problem computes, its inputs and outputs, its axes, its reference source, and
each workload's parameters and position. Without those, a deploy built from a
fresh clone renders every measured number correctly and every problem
description, reference pane, input table, output table, axis table and workload
parameter as blank (STATE.md D49). That is the whole content of five sections
on 235 pages, missing because of a build-environment property no reader can
see.

So the descriptive subset is exported here, once, and tracked. It is:

* **derived, not authored** — every field is copied verbatim from
  `definition.json` / `workload.jsonl`, and `--check` re-derives and compares;
* **descriptive, not measured** — no timing, no bound, no tolerance, no score.
  Nothing in it can affect a number. The measured artifacts stay where they
  are;
* **a fallback, not a source** — `ingest.py` reads `data/` when it is there and
  only falls back to this file, so a machine with the real dataset can never be
  served a stale copy of it.

WHAT IS DELIBERATELY LEFT OUT
-----------------------------
The per-workload `inputs` block (`{"hidden_states": {"type": "random"}, ...}`),
which says how a workload's tensors are generated. It is 40% of the bytes, the
board never renders it, and generating inputs is the harness's job — a machine
that runs the benchmark has the real dataset by definition.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "SOL-ExecBench" / "benchmark"
OUT = ROOT / "reference" / "dataset-meta.json"

# The census, confirmed against the files rather than taken from the paper.
EXPECTED = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}


def collect(dataset: Path) -> dict:
    problems: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for category in sorted(EXPECTED):
        for d in sorted((dataset / category).iterdir()):
            if not (d / "definition.json").is_file():
                continue
            defn = json.loads((d / "definition.json").read_text())
            wls = []
            wl_path = d / "workload.jsonl"
            if wl_path.is_file():
                for line in wl_path.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        # uuid and the axes that vary, in file order. The const
                        # and expr axes are in `axes` above and are merged by
                        # the reader, exactly as it merges them from the real
                        # dataset -- one implementation, not two.
                        wls.append({"uuid": r["uuid"], "axes": r.get("axes") or {}})
            problems[f"{category}__{d.name}"] = {
                "name": defn.get("name"),
                "description": defn.get("description"),
                "hf_id": defn.get("hf_id"),
                "axes": defn.get("axes") or {},
                "inputs": defn.get("inputs") or {},
                "outputs": defn.get("outputs") or {},
                "reference": defn.get("reference"),
                "workloads": wls,
            }
            counts[category] = counts.get(category, 0) + 1
    return {"problems": problems, "counts": counts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="verify the tracked file matches the dataset; write "
                         "nothing and exit non-zero if it does not")
    args = ap.parse_args()

    if not args.dataset.is_dir():
        print(f"no dataset at {args.dataset} — run "
              f"scripts/materialize_dataset.py first", file=sys.stderr)
        return 2

    got = collect(args.dataset)
    if got["counts"] != EXPECTED:
        print(f"census mismatch: found {got['counts']}, expected {EXPECTED}. "
              f"Refusing to write a partial export -- a board built from it "
              f"would be missing problems and would not say so.",
              file=sys.stderr)
        return 1

    n_wl = sum(len(p["workloads"]) for p in got["problems"].values())
    payload = {
        "_note": "Descriptive metadata only, copied verbatim from the "
                 "dataset's definition.json and workload.jsonl. No timing, no "
                 "bound, no tolerance, no score. Tracked so a deploy built "
                 "from a clone can describe a problem; see "
                 "scripts/export_dataset_meta.py.",
        "source": "data/SOL-ExecBench/benchmark (materialize_dataset.py)",
        "counts": got["counts"],
        "n_problems": len(got["problems"]),
        "n_workloads": n_wl,
        "problems": got["problems"],
    }

    # Compact, and NOT `sort_keys`. Deterministic already -- problems are
    # collected in sorted category/name order and workloads in file order -- and
    # sorting would reorder the nested `axes` dict, which is not decoration:
    # the problem page prints a workload's parameter chips in declaration
    # order, the same order upstream lists them in. An alphabetised export
    # renders a different page from the dataset it copies.
    # `generated_utc` is deliberately absent: a timestamp would make every
    # regeneration a change even when the content is identical.
    body = json.dumps(payload, separators=(",", ":")) + "\n"

    if args.check:
        if not args.out.is_file():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        if args.out.read_text() != body:
            print(f"{args.out} does not match the dataset. Regenerate with "
                  f"python {Path(__file__).relative_to(ROOT)}", file=sys.stderr)
            return 1
        print(f"{args.out.relative_to(ROOT)} matches the dataset: "
              f"{len(got['problems'])} problems, {n_wl} workloads")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body)
    print(f"wrote {args.out.relative_to(ROOT)}: {len(got['problems'])} "
          f"problems, {n_wl} workloads, {len(body) / 1e6:.1f} MB "
          f"({datetime.now(timezone.utc).isoformat(timespec='seconds')})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
