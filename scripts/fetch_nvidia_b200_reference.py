#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch NVIDIA's published B200 figures for the 235 upstream problems.

    python scripts/fetch_nvidia_b200_reference.py

Writes `reference/nvidia-b200/published.json`: for every problem, NVIDIA's own
per-workload `baseline_latency_ms` and `sol_ms` as served by the public
SOL-ExecBench site's JSON API.

WHAT THIS IS FOR, AND WHAT IT MUST NEVER BE USED FOR
----------------------------------------------------
It is an *orientation* overlay for the problem pages: the same workload, on the
part the benchmark was designed for, as its authors published it. Reading
"NVIDIA say 0.0201 ms here and we measure 0.0367 ms" tells you something about
the workload — where the shape is, whether a problem is tiny — that no AMD
number tells you on its own.

It is NOT a comparison and NOT a score, and nothing in this repo may compute
with it:

* Different part, different power cap, different sustained clock. B200 at
  NVIDIA's clock and MI350X at F_LOCK 1300 MHz are two measurements of two
  machines; the ratio between them is not a speedup, it is an artefact of two
  configurations nobody controlled against each other.
* NVIDIA's `sol_ms` is *their* roofline, from *their* arch constants. Prime
  directive 2 exists for exactly this value: it must never become an AMD
  T_SOL, never seed one, and never be used to sanity-check one. Every AMD
  bound in `artifacts/03` was derived on AMD and stays that way.
* The numbers are undated snapshots of a live site. They are stamped with the
  URL and the fetch time here, and they go stale silently — which is the other
  reason they may only ever be displayed, never depended on.

The leaderboard renders them off by default, behind a switch, in a column that
says B200 in its heading. `ingest.py` matches them to our workloads by axes and
counts what it could not match rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reference" / "nvidia-b200" / "published.json"

BASE = "https://research.nvidia.com/benchmarks/sol-execbench/api"
# The four upstream categories, which arrive as free-form tags alongside
# topic tags ("attention", "model:..."). A kernel whose tags contain none of
# them cannot be keyed to a problem in this repo and is reported, not guessed.
CATEGORIES = ("L1", "L2", "Quant", "FlashInfer-Bench")


def get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}: {url} ({e})", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def problem_key(kernel: dict) -> str | None:
    """`L1__001_foo` from the tag that names the category plus the kernel name.

    Returns None rather than a guess when no category tag is present: a key
    invented here would attach NVIDIA's numbers to the wrong problem, or to no
    problem at all, and the second failure is the quiet one.
    """
    cats = [t for t in kernel.get("tags", []) if t in CATEGORIES]
    if len(cats) != 1:
        return None
    return f"{cats[0]}__{kernel['name']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--delay", type=float, default=0.15,
                    help="seconds between requests (be a polite client)")
    ap.add_argument("--limit", type=int, default=0, help="fetch only N kernels")
    args = ap.parse_args()

    index = get(f"{BASE}/kernels")["data"]["kernels"]
    print(f"{len(index)} kernels listed", file=sys.stderr)
    if args.limit:
        index = index[: args.limit]

    out: dict[str, dict] = {}
    unkeyed: list[str] = []
    for i, k in enumerate(index, 1):
        key = problem_key(k)
        if key is None:
            unkeyed.append(k["name"])
            continue
        d = get(f"{BASE}/kernels/{k['id']}")["data"]
        out[key] = {
            "nvidia_id": d["id"],
            "gpu_types": d.get("gpu_types"),
            # NVIDIA's own flag: where it is true their SOL column is a
            # placeholder, not a roofline, and we render nothing rather than a
            # number that means nothing.
            "sol_is_dummy": bool(d.get("sol_is_dummy")),
            "baseline_latency_ms": d.get("baseline_latency_ms"),
            "workloads": [
                {"axes": w.get("axes") or {},
                 "baseline_latency_ms": w.get("baseline_latency_ms"),
                 "sol_ms": w.get("sol_ms")}
                for w in (d.get("workloads") or [])
            ],
        }
        if i % 25 == 0:
            print(f"  {i}/{len(index)}", file=sys.stderr)
        time.sleep(args.delay)

    payload = {
        "_note": "NVIDIA's published B200 figures, for display only. Never a "
                 "source for an AMD bound, a tolerance or a score — see the "
                 "module docstring of scripts/fetch_nvidia_b200_reference.py.",
        "source": f"{BASE}/kernels/<id>",
        "site": "https://research.nvidia.com/benchmarks/sol-execbench/",
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_kernels_listed": len(index),
        "n_kernels_keyed": len(out),
        "unkeyed": unkeyed,
        "kernels": out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    n_wl = sum(len(v["workloads"]) for v in out.values())
    n_dummy = sum(1 for v in out.values() if v["sol_is_dummy"])
    print(f"wrote {args.out.relative_to(ROOT)}: {len(out)} problems, "
          f"{n_wl} workloads, {n_dummy} with a dummy SOL, "
          f"{len(unkeyed)} unkeyed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
