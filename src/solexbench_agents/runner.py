# SPDX-License-Identifier: Apache-2.0
"""Drive many (harness, problem) pairs across the GPU pool.

Resumable by construction, because the assumption is that the session dies
mid-sweep -- it will. A completed unit is one that has written
``session.json``; anything without one is redone. A *failed* unit still writes
its session file, so a failure is a result rather than a gap that gets retried
forever (prime directive 1, and the same contract the sweep runners in
``scripts/runners/`` follow).

Concurrency is bounded by the GPU pool, never by arithmetic on a task index.
See ``gpu_pool`` for why that distinction cost 176 artifacts once already.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .gpu_pool import GpuPool
from .harnesses import HARNESSES, AgentSession
from .task_packet import build_packet, build_verify_root

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Unit:
    """One agent run: a harness against a problem."""

    harness: str
    problem_dir: Path

    @property
    def problem_key(self) -> str:
        return f"{self.problem_dir.parent.name}__{self.problem_dir.name}"

    def out_dir(self, run_root: Path) -> Path:
        return run_root / self.harness / self.problem_key


class BudgetExceeded(RuntimeError):
    """Raised to stop a sweep that has spent its cost cap."""


class Budget:
    """A hard cap on spend, checked before each unit starts.

    Enforced before a unit rather than after, because the point is to not spend
    the money. Only harnesses that report cost contribute; a harness that reports
    none cannot be capped this way and the sweep says so rather than pretending
    the cap covers it.
    """

    def __init__(self, max_usd: float | None) -> None:
        self.max_usd = max_usd
        self.spent = 0.0
        self.unpriced_sessions = 0
        self._lock = threading.Lock()

    def add(self, session: AgentSession) -> None:
        with self._lock:
            if session.cost_usd is None:
                self.unpriced_sessions += 1
            else:
                self.spent += session.cost_usd

    def check(self) -> None:
        if self.max_usd is not None and self.spent >= self.max_usd:
            raise BudgetExceeded(
                f"cost cap reached: ${self.spent:.2f} of ${self.max_usd:.2f}"
            )


class Sweep:
    def __init__(
        self,
        *,
        run_root: Path,
        harness_specs: dict[str, dict],
        gpus: list[int],
        max_attempts: int,
        timeout_s: int,
        workloads_root: Path | None,
        author: str = "agent",
        budget_usd: float | None = None,
        resume: bool = True,
        max_transient_retries: int = 2,
        retry_transient: bool = False,
    ) -> None:
        self.run_root = run_root
        self.harness_specs = harness_specs
        self.pool = GpuPool(gpus)
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self.workloads_root = workloads_root
        self.author = author
        self.budget = Budget(budget_usd)
        self.resume = resume
        self.max_transient_retries = max_transient_retries
        self.retry_transient = retry_transient
        # One reduced harness tree for the whole sweep. Agents verify against
        # this instead of the repo, so the tolerance derivations and analytic
        # bounds are not on a path their verify script points at.
        self.verify_root = build_verify_root(
            REPO_ROOT, run_root / "verify-root"
        )
        self._print_lock = threading.Lock()

    def _say(self, msg: str) -> None:
        with self._print_lock:
            print(msg, flush=True)

    def already_done(self, unit: Unit) -> bool:
        """Has this unit produced a result worth keeping?

        A recorded failure normally counts as done -- that is prime directive 1,
        and it is what stops a sweep retrying the same hard problem forever. The
        exception is a failure the *infrastructure* caused: a gateway 403 or a
        dropped stream says nothing about the model, so with ``retry_transient``
        those units are re-run rather than inherited. Without the flag they are
        left alone, so the default behaviour stays "a result is a result".
        """
        if not self.resume:
            return False
        path = unit.out_dir(self.run_root) / "session.json"
        if not path.exists():
            return False
        if not self.retry_transient:
            return True
        try:
            session = json.loads(path.read_text())
        except json.JSONDecodeError:
            return False
        return not (session.get("transient_failure")
                    or session.get("harness_api_error"))

    def run_unit(self, unit: Unit) -> AgentSession | None:
        out_dir = unit.out_dir(self.run_root)
        packet_dir = out_dir / "packet"
        session_path = out_dir / "session.json"

        self.budget.check()

        spec = self.harness_specs[unit.harness]
        harness = HARNESSES[unit.harness](
            timeout_s=self.timeout_s, model=spec.get("model")
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest = build_packet(
                unit.problem_dir,
                packet_dir,
                max_attempts=self.max_attempts,
                author=self.author,
                workloads_root=self.workloads_root,
                verify_root=self.verify_root,
            )
        except Exception as exc:  # noqa: BLE001
            # A problem with no AMD-derived tolerances lands here. Recorded as a
            # session so coverage accounting sees it, rather than skipped.
            session = AgentSession(
                harness=unit.harness, model=spec.get("model"),
                problem=unit.problem_key, packet_dir=str(packet_dir), gpu=-1,
                error=f"packet build failed: {type(exc).__name__}: {exc}",
            )
            session_path.write_text(json.dumps(session.as_dict(), indent=2, default=str))
            self._say(f"  [{unit.harness}] {unit.problem_key}: PACKET FAILED — {exc}")
            return session

        with self.pool.lease() as gpu:
            for attempt in range(1, self.max_transient_retries + 2):
                self._say(f"  [{unit.harness}] {unit.problem_key}: start on GPU {gpu} "
                          f"({manifest['n_workloads']} workloads)"
                          + (f", retry {attempt - 1}" if attempt > 1 else ""))
                session = harness.run(packet_dir, unit.problem_key, gpu,
                                      log_dir=out_dir)
                session.attempt = attempt
                self.budget.add(session)
                if not session.transient_failure:
                    break
                if attempt > self.max_transient_retries:
                    self._say(f"  [{unit.harness}] {unit.problem_key}: gave up after "
                              f"{attempt} infrastructure failures — recorded as one")
                    break
                self._say(f"  [{unit.harness}] {unit.problem_key}: infrastructure "
                          f"failure, not the model — {session.error}")
                # Rebuild so the retry starts from a clean packet rather than
                # inheriting half-written sources and a spent attempt counter.
                build_packet(
                    unit.problem_dir, packet_dir,
                    max_attempts=self.max_attempts, author=self.author,
                    workloads_root=self.workloads_root,
                    verify_root=self.verify_root,
                )
                time.sleep(min(30 * attempt, 120))

        session_path.write_text(json.dumps(session.as_dict(), indent=2, default=str))

        verdict = "solution" if session.produced_solution else "NO SOLUTION"
        best = (f"{session.verify_best_passed}/{session.verify_workloads}"
                if session.verify_best_passed is not None else "never verified")
        cost = f"${session.cost_usd:.2f}" if session.cost_usd is not None else "cost n/a"
        self._say(
            f"  [{unit.harness}] {unit.problem_key}: {verdict}, self-verify {best}, "
            f"{session.verify_attempts} attempt(s), {session.wallclock_s / 60:.1f} min, "
            f"{cost}{' TIMED OUT' if session.timed_out else ''}"
        )
        return session

    def run(self, units: list[Unit]) -> list[AgentSession]:
        pending = [u for u in units if not self.already_done(u)]
        skipped = len(units) - len(pending)
        self._say(
            f"{len(units)} unit(s): {len(pending)} to run, {skipped} already done, "
            f"{self.pool.size} GPU(s) in the pool"
        )
        if not pending:
            return []

        sessions: list[AgentSession] = []
        started = time.time()
        stop = False
        with ThreadPoolExecutor(max_workers=self.pool.size) as ex:
            futures = {ex.submit(self.run_unit, u): u for u in pending}
            for fut in as_completed(futures):
                unit = futures[fut]
                try:
                    session = fut.result()
                except BudgetExceeded as exc:
                    if not stop:
                        stop = True
                        self._say(f"\n  STOPPING: {exc}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    self._say(f"  [{unit.harness}] {unit.problem_key}: "
                              f"runner error {type(exc).__name__}: {exc}")
                    continue
                if session is not None:
                    sessions.append(session)

        self._say(
            f"\n{len(sessions)} session(s) in {(time.time() - started) / 60:.1f} min; "
            f"spend ${self.budget.spent:.2f}"
            + (f", {self.budget.unpriced_sessions} session(s) report no cost"
               if self.budget.unpriced_sessions else "")
        )
        return sessions


def load_deferred(repo_root: Path) -> dict[str, str]:
    """``{problem_key: reason}`` from ``artifacts/deferred.json``.

    The 15 NVFP4 problems are deferred because **their own reference fails on
    ROCm**, before any submission is involved: block-16 FP8-scaled GEMM is
    CUDA-only. There is no solvable task to hand an agent, so including them
    would score models on a problem the platform cannot express and depress every
    rate by a constant nobody could see.
    """
    path = repo_root / "artifacts" / "deferred.json"
    if not path.exists():
        return {}
    try:
        problems = json.loads(path.read_text()).get("problems", {})
    except json.JSONDecodeError:
        return {}
    return {
        key: (entry.get("reason") if isinstance(entry, dict) else str(entry))
        for key, entry in problems.items()
    }


def discover_problems(
    benchmark_dir: Path,
    categories: list[str] | None = None,
    limit_per_category: int | None = None,
    exclude: dict[str, str] | None = None,
) -> tuple[list[Path], list[str]]:
    """``(problem_dirs, excluded_keys)``, sorted, optionally sampled per category.

    ``categories=None`` means all four. Naming them explicitly is the realistic
    way scope silently shrinks -- a missing entry looks exactly like success --
    so the caller records what it asked for and ``scripts/check_coverage.py``
    checks the result against the full set.

    Exclusions are returned rather than silently dropped, so the run config can
    state them. A gap with a recorded reason is a decision; a gap without one is
    a bug.
    """
    cats = categories or ["L1", "L2", "Quant", "FlashInfer-Bench"]
    exclude = exclude or {}
    problems: list[Path] = []
    excluded: list[str] = []
    for cat in cats:
        cat_dir = benchmark_dir / cat
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"no such category directory: {cat_dir}")
        found = []
        for p in sorted(cat_dir.iterdir()):
            if not (p / "definition.json").exists():
                continue
            if f"{cat}__{p.name}" in exclude:
                excluded.append(f"{cat}__{p.name}")
                continue
            found.append(p)
        if limit_per_category is not None:
            # Evenly spaced rather than the first N: problem numbering follows
            # the source model, so the first six of L1 are all from one family
            # and would make a pilot's pass rate a fact about that family.
            if limit_per_category < len(found):
                step = len(found) / limit_per_category
                found = [found[int(i * step)] for i in range(limit_per_category)]
        problems.extend(found)
    return problems, excluded


def preflight(harness_names: list[str]) -> None:
    """Fail before spending anything if a harness CLI is not usable."""
    import shutil
    import subprocess

    missing = []
    for name in harness_names:
        exe = {"claude-code": "claude", "codex": "codex"}[name]
        if shutil.which(exe) is None:
            missing.append(f"{name}: {exe} not on PATH")
            continue
        try:
            proc = subprocess.run([exe, "--version"], capture_output=True,
                                  text=True, timeout=60)
            if proc.returncode != 0:
                missing.append(f"{name}: `{exe} --version` exited "
                               f"{proc.returncode}: {proc.stderr.strip()[:200]}")
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    if missing:
        raise RuntimeError("harness preflight failed:\n  " + "\n  ".join(missing))
    print(f"preflight OK: {', '.join(harness_names)}", file=sys.stderr)
