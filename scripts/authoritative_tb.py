#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 06, second pass — re-time the selected T_b variants on ONE GPU.

The selection sweep shards across all eight GPUs, which is the only way it
finishes in an evening. But the eight GPUs do not share a clock: at the same
determinism setting they land between 1242 and 1307 MHz, a 5% spread, which is
wider than most of the differences this benchmark exists to measure. A T_b
assembled from eight GPUs would encode that spread into the score scale, and
nothing downstream could see it.

So selection and measurement are separated:

  selection    GPUs 0-7, whole problems (all variants of one problem on ONE
               GPU, so the comparison that picks a winner is internally
               consistent even though GPUs differ)
  measurement  GPU 0 only, serial, re-timing just the variants that won

    python scripts/authoritative_tb.py --gpu 0

Resumable: a problem whose authoritative artifact exists is skipped.

Runner-up policy: `--top-k 2` re-times the two fastest passing variants per
workload rather than one. Cross-GPU selection noise can only ever mis-order
variants that are close, so re-timing the top two and taking the GPU-0 minimum
removes the failure mode entirely at roughly double the cost of the second
pass. It is on by default because an anchor that is 3% too slow inflates every
score measured against it, silently and forever.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def variants_to_retime(doc: dict, top_k: int) -> list[str]:
    """The variants worth re-timing for this problem.

    A variant is eligible only if it passed EVERY workload -- same rule the
    selection pass applies. Fast-but-wrong is not a baseline.
    """
    eligible = {
        name: r["latency_ms_by_workload"]
        for name, r in (doc.get("variants") or {}).items()
        if r.get("ok") and r.get("all_passed")
    }
    winners: set[str] = set()
    uuids = {u for lat in eligible.values() for u in lat}
    for u in uuids:
        ranked = sorted(
            ((lat[u], name) for name, lat in eligible.items() if u in lat),
        )
        for _, name in ranked[:top_k]:
            winners.add(name)
    return sorted(winners)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="artifacts/06/candidates")
    ap.add_argument("--out", default="artifacts/06/authoritative")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--gpu", default="0",
                    help="the ONE GPU every authoritative number comes from")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cand_dir, out_dir = Path(a.candidates), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    runner = ROOT / "scripts" / "runners" / "time_tb_candidates.py"

    plan: list[tuple[Path, list[str]]] = []
    no_winner: list[str] = []
    for f in sorted(cand_dir.glob("*.json")):
        doc = json.loads(f.read_text())
        key = doc.get("problem") or f.stem
        category, name = key.split("__", 1)
        names = variants_to_retime(doc, a.top_k)
        if not names:
            # No variant passed every workload -> this problem has no anchor.
            # It goes to triage, not to the manifest with a guessed one.
            no_winner.append(key)
            continue
        plan.append((Path(a.data) / category / name, names))

    pending = [(p, v) for p, v in plan
               if not (out_dir / f"{p.parent.name}__{p.name}.json").exists()]

    print(f"candidates   {len(list(cand_dir.glob('*.json')))} problems")
    print(f"re-time      {len(plan)} problems, {len(pending)} pending, "
          f"top-{a.top_k} variants each")
    print(f"no winner    {len(no_winner)}"
          + (f"  (first: {no_winner[:3]})" if no_winner else ""))
    print(f"gpu          {a.gpu}  (authoritative, exclusive)")
    if a.dry_run:
        for p, v in pending[:10]:
            print(f"  {p.parent.name}/{p.name}: {v}")
        return 0

    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(a.gpu))
    start, ok, failed = time.time(), 0, 0
    for n, (problem, names) in enumerate(pending, 1):
        key = f"{problem.parent.name}__{problem.name}"
        out_file = out_dir / f"{key}.json"
        cmd = [sys.executable, str(runner), "--problem", str(problem),
               "--out", str(out_file), "--iterations", str(a.iterations),
               "--warmup", str(a.warmup)]
        for v in names:
            cmd += ["--only-variant", v]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=a.timeout, env=env)
            good = r.returncode == 0
        except subprocess.TimeoutExpired:
            out_file.write_text(json.dumps(
                {"problem": key, "ok": False,
                 "error": f"timeout after {a.timeout}s", "gpu": a.gpu}, indent=1))
            good = False
        ok, failed = ok + good, failed + (not good)
        elapsed = time.time() - start
        eta = elapsed / n * (len(pending) - n)
        print(f"[{n}/{len(pending)}] {'ok' if good else 'FAIL':<4} {key}  "
              f"({','.join(names)})  eta {eta/60:.0f}m", flush=True)

    print(f"\ndone: {ok} ok, {failed} failed, {(time.time()-start)/60:.1f} min")
    if no_winner:
        (out_dir / "no-winner.json").write_text(json.dumps(
            {"_note": "No variant passed every workload, so these have no T_b "
                      "anchor. Triage before the manifest is cut -- a problem "
                      "without an anchor is not scoreable and must be counted "
                      "as such.",
             "problems": no_winner}, indent=1))
        print(f"{len(no_winner)} problems have no passing variant; "
              f"see {out_dir/'no-winner.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
