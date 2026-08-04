#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 06, second pass — re-time the selected T_b variants on ONE GPU.

The selection sweep shards across all eight GPUs, which is the only way it
finishes in an evening. But the eight GPUs do not share a clock. On MI350X they
land between 1242 and 1307 MHz at the same determinism setting; on MI355X the
spread is far worse -- 1316 to 1647 MHz at `--setperfdeterminism 1650`, 25% --
because only GPUs 0 and 1 reach the requested clock and the other six sit at
~0.80x it. That is wider than most of the differences this benchmark exists to
measure. A T_b assembled from eight GPUs would encode the spread into the score
scale, and nothing downstream could see it.

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
sys.path.insert(0, str(ROOT / "scripts" / "runners"))

from _common import CLOCK_VIOLATION_EXIT, write_result  # noqa: E402


def variants_to_retime(doc: dict, top_k: int, within: float = 0.25) -> list[str]:
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
        if not ranked:
            continue
        best = ranked[0][0]
        for i, (ms, name) in enumerate(ranked):
            # Always the top k; beyond that, anything close enough that
            # selection-pass noise could have mis-ordered it. Selection ran on
            # eight GPUs whose clocks span 5%, and for part of the sweep two
            # problems could share one GPU (see STATE.md D11), so "close" has
            # to mean a real band and not a rounding epsilon.
            if i < top_k or (best > 0 and ms <= best * (1 + within)):
                winners.add(name)
    return sorted(winners)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="artifacts/06/candidates")
    ap.add_argument("--out", default="artifacts/06/authoritative")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated TORCH indices to shard across. All must "
                         "hold the same clock -- see artifacts/01/"
                         "equalized-clocks.json and `clock_calibrate.py equalize`. "
                         "Defaults to --gpu for the single-GPU behaviour.")
    ap.add_argument("--gpu", default="0",
                    help="the ONE GPU every authoritative number comes from")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--within", type=float, default=0.25,
                    help="also re-time any variant within this fraction of "
                         "the fastest, beyond the top k")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # Before any GPU time is spent. A pass measured at the wrong setpoint is
    # not recoverable after the fact: T_b is a wall-clock time and the artifact
    # records the clock it was *supposed* to run at.
    if not a.dry_run:
        from provenance import assert_clock_lock
        assert_clock_lock()

    cand_dir, out_dir = Path(a.candidates), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    runner = ROOT / "scripts" / "runners" / "time_tb_candidates.py"

    plan: list[tuple[Path, list[str]]] = []
    no_winner: list[str] = []
    for f in sorted(cand_dir.glob("*.json")):
        doc = json.loads(f.read_text())
        # The filename is authoritative for the key. A timeout or crash
        # artifact is written by the shard runner, not the problem runner, and
        # records `problem` as the bare directory name with no category on it.
        key = f.stem
        category, name = key.split("__", 1)
        names = variants_to_retime(doc, a.top_k, a.within)
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
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else [int(a.gpu)]
    print(f"gpus         {gpus}  "
          + ("(authoritative, exclusive)" if len(gpus) == 1
             else f"(sharded {len(gpus)}-way; every card must hold the same clock)"))
    if a.dry_run:
        for p, v in pending[:10]:
            print(f"  {p.parent.name}/{p.name}: {v}")
        return 0

    # A GPU is borrowed and returned rather than assigned by `i % len(gpus)`.
    # Those are not the same constraint: with the modular form, two problems whose
    # indices are congruent can be in flight at once and share a card while another
    # idles, each inflating the other's timing, and the artifact records the device
    # it was TOLD to use either way. That is deviation D11, and it cost an unknown
    # subset of 176 artifacts once already.
    import queue
    import threading

    pool: queue.Queue[int] = queue.Queue()
    for g in gpus:
        pool.put(g)
    lock = threading.Lock()
    counters = {"ok": 0, "failed": 0, "n": 0}
    aborted: list[str] = []

    start, ok, failed = time.time(), 0, 0

    def run_one(problem: Path, names: list[str]) -> None:
        key = f"{problem.parent.name}__{problem.name}"
        out_file = out_dir / f"{key}.json"
        gpu = pool.get()
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, str(runner), "--problem", str(problem),
               "--out", str(out_file), "--iterations", str(a.iterations),
               "--warmup", str(a.warmup)]
        for v in names:
            cmd += ["--only-variant", v]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=a.timeout, env=env)
            good = r.returncode == 0
            if r.returncode == CLOCK_VIOLATION_EXIT:
                # The node drifted off F_LOCK mid-pass. Every artifact after
                # this one would be wrong the same way, so the pass stops here.
                # The artifact is moved aside rather than left in place: left in
                # place it counts as done and the problem is never re-timed.
                moved = out_file.with_suffix(".clock-violation.json")
                out_file.replace(moved)
                print(f"\nABORTING on GPU {gpu}: {key} — that card is not at "
                      f"F_LOCK.\n"
                      f"  evidence  {moved}\n"
                      f"  {key} is pending again, not recorded as failed.\n"
                      f"  Re-apply the lock (clock_calibrate.py lock-equalized), "
                      f"then re-run.", flush=True)
                with lock:
                    aborted.append(key)
        except subprocess.TimeoutExpired:
            # Through write_result, not json.dump: the runner was killed before
            # it could stamp its own artifact, and an artifact with no
            # provenance block cannot be told apart from one written by an
            # unknown commit against an unknown stack. It carries no
            # measured_clock -- the monitor died with the subprocess -- and says
            # so rather than leaving the field absent.
            write_result(out_file, "06-tb-candidates", {
                "problem": key, "ok": False,
                "error": f"timeout after {a.timeout}s",
                "gpu": gpu,
                "measured_clock": None,
            })
            good = False
        finally:
            pool.put(gpu)

        with lock:
            counters["n"] += 1
            counters["ok"] += good
            counters["failed"] += (not good)
            n_done = counters["n"]
            elapsed = time.time() - start
            eta = elapsed / n_done * (len(pending) - n_done)
            print(f"[{n_done}/{len(pending)}] {'ok' if good else 'FAIL':<4} "
                  f"gpu{gpu} {key}  ({','.join(names)})  eta {eta/60:.0f}m",
                  flush=True)

    # Concurrency is bounded by the pool, so the executor is sized to it: a wider
    # pool of threads would simply queue on `pool.get()`.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [ex.submit(run_one, p_, v_) for p_, v_ in pending]
        for f in futures:
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  worker error: {type(exc).__name__}: {exc}", flush=True)

    ok, failed = counters["ok"], counters["failed"]
    print(f"\ndone: {ok} ok, {failed} failed, {(time.time()-start)/60:.1f} min")
    if aborted:
        print(f"{len(aborted)} problem(s) aborted on a clock violation: "
              f"{aborted[:5]}")
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
