#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a Claude Code agent as a kernel-optimizing baseline, and cost it.

`artifacts/09/agent-baseline.json` records that no agent was ever run on this
node, and why the PyTorch variant set cannot stand in for one. This is the
instrument that closes that gap: it puts a real agent in a sandbox with the
real evaluation harness and measures what it costs in dollars, wall time and
GPUs.

    python scripts/agent_baseline.py --sample 8 --gpus 1,2,3,4,5,6,7

Each problem gets its own sandbox, its own GPU and its own `claude -p` session.
Nothing here writes to `data/` or `artifacts/09/`; the run lands in
`artifacts/10/<run-id>/`.

Three properties are deliberate:

* **GPU 0 is never handed to an agent.** Agent sessions are exploration under
  CLAUDE.md section 4, so they get 1-7. The winning kernels are re-timed on an
  idle GPU 0 afterwards by `agent_score.py`, and only that number is scored.
* **The agent is not shown T_SOL or T_b.** See `agent_eval.py`. It optimizes
  against the hardware, not against the scoring constants.
* **Cost is captured from the CLI's own accounting**, per session, including
  the sessions that fail. A baseline that quietly drops its failures reports
  the cost of the easy problems.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import stamp  # noqa: E402

DATASET = ROOT / "data" / "SOL-ExecBench" / "benchmark"
MANIFEST = ROOT / "artifacts" / "09" / "manifest-v1.json"
WORKLOADS_ROOT = "/work/artifacts/05/workloads"

GATEWAY_URL = "https://llm-api.amd.com/Anthropic"
GATEWAY_KEY = os.environ.get("LLM_GATEWAY_KEY", "")
NTID = os.environ.get("AMD_USER_NTID", "qinwu")


TASK_TEMPLATE = """\
# Optimize a GPU kernel for AMD Instinct MI350X

You are given a working PyTorch reference implementation. **Make it faster
while keeping it numerically correct.** Whatever is in `kernel.py` when you
stop is your submission.

## The hardware

AMD Instinct MI350X, CDNA4, `gfx950`. 256 CUs, 288 GB HBM3E, 8 TB/s.
ROCm 7.2, PyTorch 2.9.1. Clocks are locked at 1300 MHz, so timings are
repeatable — a change in measured latency is a real change.

This is **not** an NVIDIA GPU. A wavefront is 64 lanes, not 32. There are no
tensor cores in the CUDA sense; the matrix engine is MFMA. `torch.compile`,
Triton and HIP C++ all work. Do not assume a CUDA idiom transfers.

## Your loop

```bash
./evaluate          # evaluates kernel.py: correctness + latency vs reference
```

It prints one line per workload: PASS/FAIL, your latency, the reference
latency, and your speedup. **Every workload must PASS.** A faster kernel that
fails one workload scores zero — correctness is a gate, not a trade-off.

Run `./evaluate` as often as you like; it takes seconds to a couple of minutes.
Iterate: measure, change one thing, measure again.

## Rules

* `kernel.py` must define `run(...)` with the **same signature and return type**
  as the reference.
* Compute the real thing. Caching results across calls, returning inputs,
  writing into the output without computing it, or special-casing the specific
  shapes the harness happens to use are all detected and rejected by the
  harness's anti-reward-hack checks. A rejected submission scores zero.
* You may use `torch`, `torch.compile`, Triton, or hand-written HIP via
  `torch.utils.cpp_extension`. `aiter` and `hipblaslt` are installed.
* Numerical tolerance is fixed and generous but real: the harness compares
  against the reference with a per-workload atol/rtol derived on this hardware.
  Reordering a reduction is fine. Dropping precision to bf16 where the
  reference uses fp32 usually is not.

## The problem: `{name}`

{description}

**Axes** (workload dimensions):

{axes}

**Inputs**

{inputs}

**Outputs**

{outputs}

**Workload shapes you will be evaluated on** ({n_workloads} of them):

{shapes}

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
{reference}
```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
"""


EVALUATE_SH = """\
#!/usr/bin/env bash
# Evaluate kernel.py. Correctness first, then latency vs the reference.
set -uo pipefail
cd "$(dirname "$0")"
STAMP=$(date +%s%N)
# Snapshot the exact source being measured, next to the measurement. Without
# this the trajectory is a list of latencies with no way to say what produced
# them -- the agent overwrites kernel.py in place, so only the last version
# survives the session.
cp kernel.py {sandbox}/evals/kernel-${{STAMP}}.py 2>/dev/null || true
exec env HIP_VISIBLE_DEVICES={gpu} \\
     SOLEXBENCH_WORKLOADS_ROOT={workloads_root} \\
     {repo}/env/solb python /work/scripts/agent_eval.py \\
        --problem {problem_rel} \\
        --kernel {sandbox}/kernel.py \\
        --out {sandbox}/evals/eval-${{STAMP}}.json \\
        --iterations {iterations} --warmup {warmup}
"""


def fmt_axes(defn: dict) -> str:
    out = []
    for k, v in (defn.get("axes") or {}).items():
        kind = v.get("type")
        if kind == "const":
            out.append(f"- `{k}` = {v.get('value')} (constant) — {v.get('description','')}")
        elif kind == "expr":
            out.append(f"- `{k}` = `{v.get('expr')}` (derived) — {v.get('description','')}")
        else:
            out.append(f"- `{k}` (varies per workload) — {v.get('description','')}")
    return "\n".join(out) or "- (none)"


def fmt_tensors(d: dict) -> str:
    out = []
    for k, v in (d or {}).items():
        shape = v.get("shape")
        shape_s = "scalar" if shape is None else "[" + ", ".join(map(str, shape)) + "]"
        out.append(f"- `{k}`: {shape_s}, `{v.get('dtype')}` — {v.get('description','')}")
    return "\n".join(out) or "- (none)"


def fmt_shapes(workloads: list[dict], limit: int = 20) -> str:
    out = []
    for w in workloads[:limit]:
        out.append("- " + ", ".join(f"{k}={v}" for k, v in (w.get("axes") or {}).items()))
    if len(workloads) > limit:
        out.append(f"- ... and {len(workloads) - limit} more")
    return "\n".join(out)


def build_sandbox(problem_dir: Path, sandbox: Path, gpu: int,
                  iterations: int, warmup: int) -> dict:
    """Materialize one agent's working directory. Returns task metadata."""
    if sandbox.exists():
        shutil.rmtree(sandbox)
    (sandbox / "evals").mkdir(parents=True)

    defn = json.loads((problem_dir / "definition.json").read_text())
    workloads = [json.loads(l) for l in
                 (problem_dir / "workload.jsonl").read_text().splitlines() if l.strip()]

    reference = defn["reference"]
    (sandbox / "reference.py").write_text(reference)
    (sandbox / "kernel.py").write_text(reference)

    task = TASK_TEMPLATE.format(
        name=defn["name"],
        description=defn.get("description", ""),
        axes=fmt_axes(defn),
        inputs=fmt_tensors(defn.get("inputs")),
        outputs=fmt_tensors(defn.get("outputs")),
        n_workloads=len(workloads),
        shapes=fmt_shapes(workloads),
        reference=reference,
    )
    (sandbox / "TASK.md").write_text(task)

    problem_rel = "/work/" + str(problem_dir.relative_to(ROOT))
    ev = sandbox / "evaluate"
    ev.write_text(EVALUATE_SH.format(
        gpu=gpu, workloads_root=WORKLOADS_ROOT, repo=ROOT,
        problem_rel=problem_rel, sandbox=sandbox,
        iterations=iterations, warmup=warmup))
    ev.chmod(0o755)

    return {"definition": defn["name"], "n_workloads": len(workloads),
            "task_chars": len(task), "reference_chars": len(reference)}


def write_settings(path: Path, model: str) -> Path:
    """Pin the gateway credentials via `--settings`, which is the only way.

    `~/.claude.json` carries an `env` block, and it **overrides the process
    environment** for a Claude Code session. On this host that block sets
    ANTHROPIC_CUSTOM_HEADERS to a different AMD gateway subscription key, so
    every header this script exported was silently discarded -- calls
    authenticated with the settings-file key instead. The failure is invisible:
    the run succeeds, just billed to the wrong key. It was caught by passing a
    deliberately invalid key and watching the session succeed anyway.

    `--settings` takes precedence over `~/.claude.json`, verified the same way:
    an invalid key passed here does make the session fail.

    Written outside the repo with 0600 permissions, because it contains a
    credential.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": GATEWAY_URL,
            "ANTHROPIC_CUSTOM_HEADERS":
                f"Ocp-Apim-Subscription-Key: {GATEWAY_KEY}\nuser: {NTID}",
            "ANTHROPIC_API_KEY": "dummy",
            "ANTHROPIC_MODEL": model,
        }
    }, indent=1))
    path.chmod(0o600)
    return path


def run_agent(sandbox: Path, model: str, budget_usd: float, timeout: int,
              settings: Path, transcript: Path | None = None) -> dict:
    """One `claude -p` session.

    Streamed rather than buffered: a three-hour session at a $100 cap must not
    lose its whole transcript if the process is killed, and the turn-by-turn
    record is the input to any post-hoc analysis of *how* the budget was spent.
    """
    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": GATEWAY_URL,
        "ANTHROPIC_AUTH_TOKEN": GATEWAY_KEY,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_CUSTOM_HEADERS":
            f"Ocp-Apim-Subscription-Key: {GATEWAY_KEY}\nuser: {NTID}",
        # The gateway is inside the AMD network; a proxy would break it.
        "NO_PROXY": "*", "no_proxy": "*",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)

    cmd = [
        "claude", "-p", (sandbox / "TASK.md").read_text(),
        "--settings", str(settings),
        "--model", model,
        "--output-format", "stream-json", "--verbose",
        "--permission-mode", "bypassPermissions",
        "--max-budget-usd", str(budget_usd),
        "--no-session-persistence",
        "--disable-slash-commands",
    ]
    t0 = time.time()
    result: dict = {}
    n_events = 0
    tf = open(transcript, "w") if transcript else None
    try:
        proc = subprocess.Popen(cmd, cwd=sandbox, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for line in proc.stdout:
                if tf:
                    tf.write(line)
                    tf.flush()
                line = line.strip()
                if not line:
                    continue
                n_events += 1
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "result":
                    result = ev
            proc.wait(timeout=max(60, timeout))
        except subprocess.TimeoutExpired:
            proc.kill()
            result = result or {"is_error": True, "timed_out": True,
                                "result": f"session timed out after {timeout}s"}
        if not result:
            result = {"is_error": True,
                      "result": f"no result event (rc={proc.returncode})",
                      "stderr_tail": (proc.stderr.read() or "")[-2000:]}
    except BaseException as e:  # noqa: BLE001
        result = {"is_error": True, "result": f"{type(e).__name__}: {e}"}
    finally:
        if tf:
            tf.close()
    result["wall_seconds"] = time.time() - t0
    result["transcript_events"] = n_events
    if transcript:
        result["transcript"] = str(transcript)
    return result


def gpu_util_sampler(stop: threading.Event, samples: list, period: float = 5.0):
    """Sample per-GPU busy% for the duration of the run.

    The point is the third number the study owes: how much GPU a fleet of
    agents actually uses. An agent spends most of its time thinking, so the
    answer is expected to be low — but expected is not measured.
    """
    while not stop.is_set():
        try:
            out = subprocess.run(["rocm-smi", "--showuse", "--json"],
                                 capture_output=True, text=True, timeout=20)
            d = json.loads(out.stdout)
            row = {"t": time.time()}
            for card, vals in d.items():
                for k, v in vals.items():
                    if "use" in k.lower():
                        try:
                            row[card] = float(v)
                        except (TypeError, ValueError):
                            pass
            samples.append(row)
        except Exception:  # noqa: BLE001 - sampling must never kill the run
            pass
        stop.wait(period)


def select_problems(sample: int, explicit: list[str] | None, seed: int) -> list[str]:
    """Stratified by category and by headroom, deterministically.

    A sample drawn only from L1 would understate cost (L1 problems are the
    small ones), and a sample drawn only from high-headroom problems would
    overstate what an agent can win. Both axes are stratified.
    """
    manifest = json.loads(MANIFEST.read_text())
    problems = manifest["problems"]
    if explicit:
        missing = [p for p in explicit if p not in problems]
        if missing:
            raise SystemExit(f"unknown problem keys: {missing}")
        return explicit

    scoreable = {}
    for key, p in problems.items():
        wl = [w for w in p.get("workloads", {}).values() if w.get("scoreable")]
        if not wl:
            continue
        # Headroom = how much room the agent has between the reference-derived
        # anchor and the physical bound. Median over the problem's workloads.
        ratios = sorted(w["t_b_ms"] / w["t_sol_ms"] for w in wl
                        if w.get("t_sol_ms"))
        if not ratios:
            continue
        scoreable[key] = {"category": p["category"],
                          "headroom": ratios[len(ratios) // 2],
                          "n_workloads": len(wl)}

    by_cat: dict[str, list[str]] = {}
    for k, v in scoreable.items():
        by_cat.setdefault(v["category"], []).append(k)

    # Proportional allocation, at least one per category.
    total = len(scoreable)
    quota = {c: max(1, round(sample * len(ks) / total)) for c, ks in by_cat.items()}
    while sum(quota.values()) > sample:
        quota[max(quota, key=lambda c: quota[c])] -= 1
    while sum(quota.values()) < sample:
        quota[max(quota, key=lambda c: len(by_cat[c]) - quota[c])] += 1

    chosen: list[str] = []
    for cat, keys in sorted(by_cat.items()):
        keys = sorted(keys, key=lambda k: scoreable[k]["headroom"])
        n = quota[cat]
        # Even quantiles of the headroom distribution: low, middle, high.
        for i in range(n):
            idx = int((i + 0.5) * len(keys) / n)
            chosen.append(keys[min(idx, len(keys) - 1)])
    return sorted(set(chosen))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--model", default="Claude-Opus-5")
    ap.add_argument("--budget-usd", type=float, default=8.0,
                    help="hard per-session cost cap")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "10")
    ap.add_argument("--sandbox-root", type=Path,
                    default=Path(os.environ.get("SOLEXBENCH_SCRATCH", "/var/tmp/solbench")) / "agent")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not GATEWAY_KEY:
        raise SystemExit("LLM_GATEWAY_KEY is not set")

    run_id = a.run_id or f"pilot-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    outdir = a.out / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    keys = select_problems(a.sample, a.problems, seed=0)
    print(f"run {run_id}: {len(keys)} problems, model={a.model}, "
          f"gpus={a.gpus}, budget=${a.budget_usd}/session")
    for k in keys:
        print("   ", k)
    if a.dry_run:
        return 0

    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    if 0 in gpus:
        raise SystemExit("GPU 0 is reserved for authoritative timing (CLAUDE.md s4)")
    pool: queue.Queue[int] = queue.Queue()
    for g in gpus:
        pool.put(g)

    # The credential file lives with the sandboxes, never in the repo or the
    # artifact tree, because it carries a key.
    settings = write_settings(a.sandbox_root / run_id / "gateway-settings.json",
                              a.model)
    transcripts = outdir / "transcripts"
    print(f"gateway: {GATEWAY_URL}  key {GATEWAY_KEY[:8]}…  ntid {NTID}  "
          f"(pinned via --settings)")

    stop = threading.Event()
    util_samples: list = []
    sampler = threading.Thread(target=gpu_util_sampler, args=(stop, util_samples),
                               daemon=True)
    sampler.start()

    results: dict[str, dict] = {}
    lock = threading.Lock()

    def worker(key: str):
        gpu = pool.get()
        try:
            cat, name = key.split("__", 1)
            problem_dir = DATASET / cat / name
            sandbox = a.sandbox_root / run_id / key
            t0 = time.time()
            meta = build_sandbox(problem_dir, sandbox, gpu, a.iterations, a.warmup)
            print(f"[{key}] gpu {gpu}: starting", flush=True)
            transcripts.mkdir(parents=True, exist_ok=True)
            session = run_agent(sandbox, a.model, a.budget_usd, a.timeout,
                                settings, transcripts / f"{key}.jsonl")
            rec = {
                "problem": key, "gpu": gpu, "sandbox": str(sandbox),
                **meta,
                "session": session,
                "wall_seconds": time.time() - t0,
                "kernel_changed": (sandbox / "kernel.py").read_text()
                                  != (sandbox / "reference.py").read_text(),
                "n_evals": len(list((sandbox / "evals").glob("*.json"))),
            }
            with lock:
                results[key] = rec
                (outdir / "sessions.json").write_text(json.dumps(results, indent=1))
            print(f"[{key}] gpu {gpu}: done in {rec['wall_seconds']:.0f}s "
                  f"cost=${session.get('total_cost_usd', 0):.2f} "
                  f"evals={rec['n_evals']} changed={rec['kernel_changed']}",
                  flush=True)
        except BaseException as e:  # noqa: BLE001
            import traceback
            with lock:
                results[key] = {"problem": key, "gpu": gpu, "error": str(e),
                                "traceback": traceback.format_exc()}
                (outdir / "sessions.json").write_text(json.dumps(results, indent=1))
            print(f"[{key}] FAILED: {e}", flush=True)
        finally:
            pool.put(gpu)

    threads = [threading.Thread(target=worker, args=(k,)) for k in keys]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    sampler.join(timeout=10)

    payload = {
        **stamp("10-agent-baseline"),
        "run_id": run_id,
        "model": a.model,
        "gateway": GATEWAY_URL,
        "budget_usd_per_session": a.budget_usd,
        "gpus_used_by_agents": gpus,
        "n_problems": len(keys),
        "problems": keys,
        "gateway_key_prefix": GATEWAY_KEY[:8],
        "gateway_ntid": NTID,
        "credentials_pinned_via": "--settings (overrides ~/.claude.json env)",
        "transcripts_dir": str(transcripts),
        "wall_seconds_total": time.time() - t_start,
        "sessions": results,
        "gpu_util_samples": util_samples,
        "eval_settings": {"iterations": a.iterations, "warmup": a.warmup},
        "note": "Agent sessions ran on GPUs 1-7 (exploration). Nothing here is "
                "an authoritative timing; agent_score.py re-times the winning "
                "kernels on an idle GPU 0.",
    }
    (outdir / "run.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {outdir / 'run.json'}")

    total = sum(r.get("session", {}).get("total_cost_usd", 0) or 0
                for r in results.values())
    print(f"total list-price cost: ${total:.2f} over {len(keys)} problems "
          f"in {(time.time() - t_start)/60:.1f} min wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
