#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score an agent run authoritatively, on an idle GPU 0.

    python scripts/agent_score.py --run artifacts/10/<run-id>

The agents optimized on GPUs 1-7 while seven other agents hammered the node.
Nothing measured under those conditions is a scoring number (CLAUDE.md s4), so
every surviving kernel is re-timed here, one at a time, on GPU 0, at the same
iteration count task 06 used for T_b (50 iterations, 10 warmup). Only that
number is scored.

The re-time is also the honesty check. A kernel that was fast in the agent's
sandbox and is not fast here was measuring contention, not itself; a kernel
that passed there and fails here was passing a noisier bar. Both outcomes are
recorded per workload rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402

# Load `sol_score` from its file rather than importing the package. This runs
# on the host python, which has no pydantic, and `import sol_execbench` pulls
# the whole data-model package in through __init__. Reimplementing the formula
# here instead would let the scorer silently drift from the harness, which is
# the one thing worth avoiding.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_sol_score", ROOT / "src" / "sol_execbench" / "sol_score.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sol_score = _mod.sol_score

MANIFEST = ROOT / "artifacts" / "09" / "manifest-v1.json"
DATASET = ROOT / "data" / "SOL-ExecBench" / "benchmark"


def bounds() -> dict:
    m = json.loads(MANIFEST.read_text())
    out = {}
    for key, p in m["problems"].items():
        for uuid, w in p.get("workloads", {}).items():
            if w.get("scoreable") and w.get("t_sol_ms") and w.get("t_b_ms"):
                out[(key, uuid)] = (w["t_sol_ms"], w["t_b_ms"])
    return out


SCRATCH = Path(os.environ.get("SOLEXBENCH_SCRATCH", "/var/tmp/solbench"))

# `run.json.harness` -> what to call it on the board. A harness not listed here
# is printed as it named itself rather than mapped to a default: an unknown
# harness is a fact about the run, and the old default ("Claude Code agent")
# was applied to every run this script ever scored, including one that was not.
HARNESS_NAMES = {
    "claude-code": "Claude Code agent",
    "codex": "codex-cli agent",
    "codex-cli": "codex-cli agent",
}


def retime(problem_key: str, kernel: Path, out: Path, gpu: int,
           iterations: int, warmup: int, timeout: int) -> dict:
    """One kernel, through env/solb, pinned to `gpu`. Returns the eval payload.

    `--out` must name a path the *container* can write. Only two trees are
    bind-mounted: the repo at /work, and SOLEXBENCH_SCRATCH at its own
    absolute path. A host path outside those (a run directory under $HOME, say)
    resolves inside the container to a directory the unprivileged user cannot
    create, and the runner dies before writing anything. So the artifact is
    written to scratch and copied out afterwards, which works wherever the run
    directory lives.
    """
    cat, name = problem_key.split("__", 1)
    staged = SCRATCH / "retime" / f"{problem_key}.json"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        staged.unlink()

    # `--timeout` reaches the INNER evaluation, not just the subprocess call.
    # It did not, and the gap was invisible: `agent_eval.py --timeout` defaults
    # to 1200s, so every evaluation was capped at twenty minutes no matter what
    # was passed here, and raising this script's `--timeout` to 2400 bought
    # nothing. FlashInfer-Bench__014 is the case -- a paged-prefill problem
    # whose re-time genuinely needs longer, recorded as
    # `TimeoutExpired ... after 1200 seconds` with 0 workloads, which reads as
    # a broken kernel rather than a scorer that was not given enough time.
    #
    # The inner cap is the smaller of the two on purpose. Whichever fires
    # first decides what the artifact says, and the inner one writes a real
    # result document; the outer one can only kill the process and leave
    # `retime()` to synthesise "produced no artifact".
    inner = max(60, timeout - 120)
    cmd = [
        str(ROOT / "env" / "solb"), "python", "/work/scripts/agent_eval.py",
        "--problem", f"/work/data/SOL-ExecBench/benchmark/{cat}/{name}",
        "--kernel", str(kernel),
        "--out", str(staged),
        "--iterations", str(iterations), "--warmup", str(warmup),
        "--timeout", str(inner),
        "--quiet",
    ]
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "HIP_VISIBLE_DEVICES": str(gpu),
        "SOLEXBENCH_WORKLOADS_ROOT": "/work/artifacts/05/workloads",
    }
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout)
        rc, err = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        rc, err = -1, f"timed out after {timeout}s"

    if staged.exists():
        payload = json.loads(staged.read_text())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(staged.read_text())
        return payload
    # A silent empty result reads exactly like "the kernel failed", which is a
    # different and much less alarming statement than "the runner never ran".
    return {"ok": False,
            "error": f"runner produced no artifact (rc={rc})",
            "stderr_tail": (err or "")[-3000:],
            "per_workload": [], "workloads": 0, "passed": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=lambda p: Path(p).resolve(),
                    help="artifacts/10/<run-id> (must contain run.json)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--reuse-retimed", action="store_true",
                    help="re-derive scores from existing retimed/*.json "
                         "without touching the GPU")
    ap.add_argument("--exclude-from-leaderboard", action="store_true",
                    help="score and record, but do not list on the board "
                         "(for validation runs)")
    a = ap.parse_args()

    run = json.loads((a.run / "run.json").read_text())
    b = bounds()
    retimed_dir = a.run / "retimed"
    retimed_dir.mkdir(parents=True, exist_ok=True)

    # The sandboxes live in /var/tmp and will be swept. A score whose kernel no
    # longer exists cannot be reproduced or disputed, so the source is copied
    # next to the number that came from it.
    kernels_dir = a.run / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    per_problem: dict[str, dict] = {}
    t0 = time.time()

    for key, sess in sorted(run["sessions"].items()):
        sandbox = Path(sess.get("sandbox", ""))
        kernel = sandbox / "kernel.py"
        reference = sandbox / "reference.py"
        rec: dict = {"problem": key, "gpu": a.gpu}

        if not kernel.exists():
            rec["skipped"] = "no kernel.py in sandbox"
            per_problem[key] = rec
            print(f"[{key}] SKIP: {rec['skipped']}")
            continue

        unchanged = reference.exists() and kernel.read_text() == reference.read_text()
        rec["kernel_unchanged_from_reference"] = unchanged
        saved = kernels_dir / f"{key}.py"
        saved.write_text(kernel.read_text())
        # Repo-relative when it is in the repo, absolute otherwise: a run
        # directory may legitimately live outside the tree (a scratch
        # experiment), and `relative_to` raises rather than falling back.
        try:
            rec["kernel_saved"] = str(saved.relative_to(ROOT))
        except ValueError:
            rec["kernel_saved"] = str(saved)
        # Anything else the agent left behind that the kernel might import.
        extra = sorted(p.name for p in sandbox.glob("*")
                       if p.is_file() and p.suffix in (".hip", ".cu", ".cpp", ".h")
                       and p.name not in ("kernel.py", "reference.py"))
        if extra:
            side = kernels_dir / key
            side.mkdir(exist_ok=True)
            for nm in extra:
                (side / nm).write_text((sandbox / nm).read_text())
            rec["kernel_side_files"] = extra

        existing = retimed_dir / f"{key}.json"
        if a.reuse_retimed and existing.exists():
            # Re-deriving scores from a completed re-time must not need the GPU
            # again: the timing is the expensive part and it does not change.
            ev = json.loads(existing.read_text())
            print(f"[{key}] reusing re-time from {existing.name}", flush=True)
        else:
            print(f"[{key}] re-timing on GPU {a.gpu} ...", flush=True)
            ev = retime(key, kernel, existing, a.gpu,
                        a.iterations, a.warmup, a.timeout)
        rec["ok"] = ev.get("ok")
        rec["error"] = ev.get("error")

        scored = flagged = passed = violations = 0
        for w in ev.get("per_workload", []):
            uuid = w.get("workload_uuid")
            status = w.get("status")
            bound = b.get((key, uuid))
            is_flag = status == "REWARD_HACK"
            score = None
            violated = False
            if status == "PASSED" and bound and w.get("latency_ms"):
                t_sol, t_b = bound
                score = sol_score(w["latency_ms"], t_b, t_sol)
                scored += 1
                passed += 1
                # A hard invariant, not a threshold. T_SOL is the time this
                # workload would take if it were limited only by the arithmetic
                # it must do and the bytes it must move, so nothing can beat
                # it: S > 1 means the BOUND is wrong, never that the kernel is
                # superhuman. The T_SOL <= T_b gate cannot catch this, because
                # a bound that over-counts traffic is under-cut by the
                # reference too -- only a kernel that avoids the traffic
                # exposes it.
                violated = w["latency_ms"] < t_sol
                violations += int(violated)
            results.append({
                "problem": key, "workload_uuid": uuid, "status": status,
                "latency_ms": w.get("latency_ms"), "score": score,
                "flagged": is_flag,
                "bound_violation": violated,
                "note": f"authoritative gpu{a.gpu}, {a.iterations} iters"
                        + (" -- FASTER THAN T_SOL: bound is invalid" if violated else ""),
            })
            flagged += int(is_flag)

        if violations:
            print(f"[{key}] !! {violations}/{scored} workloads came in FASTER than "
                  f"T_SOL. The bound for this problem is wrong; its scores are "
                  f"not usable.", flush=True)

        rec["bound_violations"] = violations
        rec.update({"workloads": ev.get("workloads", 0), "passed": passed,
                    "scored": scored, "flagged": flagged,
                    "geomean_speedup": ev.get("geomean_speedup"),
                    "stderr_tail": ev.get("stderr_tail")})
        per_problem[key] = rec
        if not ev.get("ok") and ev.get("workloads", 0) == 0:
            # Distinguish "this kernel scored nothing" from "nothing ran".
            print(f"[{key}] RUNNER FAILED: {ev.get('error')}", flush=True)
            for ln in (ev.get("stderr_tail") or "").strip().splitlines()[-6:]:
                print(f"    | {ln}", flush=True)
        else:
            print(f"[{key}] {passed}/{ev.get('workloads', 0)} passed, "
                  f"{scored} scored, {flagged} flagged, "
                  f"speedup={ev.get('geomean_speedup') or float('nan'):.2f}x",
                  flush=True)

    scores = [r["score"] for r in results if r["score"] is not None]
    # Headline mean excludes workloads whose bound is provably wrong. Averaging
    # them in would let a defective bound raise the score of a whole run.
    clean = [r["score"] for r in results
             if r["score"] is not None and not r.get("bound_violation")]
    violated_problems = sorted({r["problem"] for r in results
                                if r.get("bound_violation")})
    sessions = run.get("sessions", {})
    total_cost = sum((s.get("session", {}) or {}).get("total_cost_usd") or 0
                     for s in sessions.values())

    # How much of the benchmark this run was pointed at, and whether its
    # sessions chose when to stop. Both read from artifacts; absent when
    # nothing recorded them, rather than defaulted to a flattering value.
    n_problems = run.get("n_problems") or len(sessions)
    n_scoreable = len({k for k, _ in b}) or None
    effort = {}
    cost_report = a.run / "cost-report.json"
    if cost_report.exists():
        try:
            effort = json.loads(cost_report.read_text())
        except json.JSONDecodeError:
            effort = {}
    cap_seconds = effort.get("wall_cap_seconds")
    capped = sum(1 for p in (effort.get("per_problem") or []) if p.get("capped"))

    payload = {
        **stamp("10-agent-scored"),
        "run_id": run.get("run_id"),
        "model": run.get("model"),
        # The harness, from the run, not "Claude Code" asserted. glm-sweep-2 is
        # codex-cli, and calling it Claude Code would both misattribute it and
        # give it a name indistinguishable from `agent-glm-run1`, which really
        # is Claude Code on the same model -- two different harnesses under one
        # label on a board whose whole job is telling runs apart.
        "display_name": f"{HARNESS_NAMES.get(run.get('harness'), run.get('harness') or 'Agent')} "
                        f"({run.get('model')})",
        "author": "claude-code",
        # Described, not asserted. This used to end "Coverage is deliberately
        # partial -- this is a cost study, not a full-benchmark submission",
        # hardcoded, on every run it ever wrote. True of the three pilots it
        # was written for and false of the first 192-problem sweep, where it
        # would have told the reader to discount a run that was trying to cover
        # the benchmark. The coverage sentence is now derived from the coverage,
        # and the harness's own account of itself is quoted when it left one.
        "notes": (
            f"{n_problems} of {n_scoreable} scoreable problems"
            + (f" ({100 * n_problems / n_scoreable:.0f}%)"
               if n_scoreable else "")
            + f". Agents optimized on GPUs {run.get('gpus_used_by_agents')}; "
            f"every score here is a re-time on an idle GPU {a.gpu} at "
            f"{a.iterations} iterations, the same settings T_b was measured at."
            + (f" {capped} of them were stopped by the harness's "
               f"{cap_seconds:g}s wall-clock cap rather than by the agent "
               f"deciding it was done." if capped else "")
            + (f" {run['note']}" if run.get("note") else "")),
        "leaderboard": not a.exclude_from_leaderboard,
        "authoritative_gpu": a.gpu,
        "iterations": a.iterations, "warmup": a.warmup,
        "total_cost_usd": total_cost,
        "wall_seconds_total": run.get("wall_seconds_total"),
        "retime_seconds": time.time() - t0,
        "per_problem": per_problem,
        "results": results,
        "summary": {
            "problems": len(per_problem),
            "workloads_scored": len(scores),
            "workloads_flagged": sum(1 for r in results if r["flagged"]),
            "workloads_bound_violated": len(scores) - len(clean),
            "problems_with_invalid_bound": violated_problems,
            "mean_score": (sum(clean) / len(clean)) if clean else 0.0,
            "mean_score_including_invalid_bounds":
                (sum(scores) / len(scores)) if scores else 0.0,
            "min_score": min(clean) if clean else None,
            "max_score": max(clean) if clean else None,
        },
    }
    (a.run / "scored.json").write_text(json.dumps(payload, indent=1, default=str))
    s = payload["summary"]
    print(f"\nwrote {a.run / 'scored.json'}")
    print(f"  {s['workloads_scored']} workloads scored, "
          f"mean S = {s['mean_score']:.4f}, {s['workloads_flagged']} flagged")
    if s["workloads_bound_violated"]:
        print(f"  EXCLUDED {s['workloads_bound_violated']} workloads with an "
              f"invalid bound (faster than T_SOL) across "
              f"{len(s['problems_with_invalid_bound'])} problem(s): "
              f"{', '.join(s['problems_with_invalid_bound'])}")
        print(f"  including them would report mean S = "
              f"{s['mean_score_including_invalid_bounds']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
