#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score an agent run authoritatively, on an idle GPU 0.

    python scripts/agent_score.py --run artifacts/10/<run-id> \
        --manifest artifacts/09-MI355X/manifest-v3.json

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

from provenance import detected_part, stamp  # noqa: E402
# The gate's resolver, imported rather than reimplemented. Two answers to "which
# part is this artifact" is the defect this whole change is about; `artifact_part`
# reads `_provenance.part` first and falls back to the torch device names, which
# is how every artifact on disk is attributed today. It is stdlib-only, so it
# imports on the host python this driver runs under.
from verify_artifacts import artifact_part  # noqa: E402
# The one definition of "which millisecond column is the bound" (D63). Imported
# for the same reason `artifact_part` is: a second answer to the same question is
# the defect. Stdlib-only on this path -- verified, since this driver runs on the
# host python with no pydantic.
from bound_headroom import published_bound_ms  # noqa: E402
from tolerance_roots import (  # noqa: E402
    container_tolerance_root,
    recorded_tolerance_root,
)

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

DATASET = ROOT / "data" / "SOL-ExecBench" / "benchmark"


def bounds(manifest: Path) -> tuple[dict, str, dict]:
    """{(problem, uuid) -> (T_SOL, T_b)}, the manifest version, and the basis census.

    The version is returned rather than looked up later because a score is only
    meaningful inside one manifest, and the two must be written together or a
    reader has a number with no way to tell which bounds produced it.

    WHICH T_SOL (D63). This used to read `w["t_sol_ms"]` straight out of the
    record. On a manifest built on the unlocked basis that column is a cycle
    count divided by whichever reference clock the winning tier happened to use,
    not the bound the manifest publishes: measured over
    `artifacts/09-MI355X/manifest-v4.json` the two differ by more than 1% on
    1622 of 3717 scoreable workloads and by more than 30% on 1084. This is the
    submission path, so reading the wrong column here does not merely mis-report
    a score, it computes one -- against a different bound from the one check D
    gates on, `backfill_scores.py` rebases to, and the board serves.
    `published_bound_ms` is the single place that choice is made; it degrades to
    the legacy column (counted, never silently) for the two frozen MI350X
    manifests, which carry `t_sol_ms_published` on 0 of 3717 workloads and were
    measured at one F_LOCK.

    The census travels with the scores for the same reason `ingest.py` writes
    `meta.bound_basis`: a run scored against a legacy column and one scored
    against the published bound are not comparable, and nothing else in the
    artifact would say which happened.
    """
    m = json.loads(manifest.read_text())
    out = {}
    basis_counts: dict[str, int] = {}
    for key, p in m["problems"].items():
        for uuid, w in p.get("workloads", {}).items():
            if not w.get("scoreable") or not w.get("t_b_ms"):
                continue
            t_sol, basis = published_bound_ms(w)
            if not t_sol:
                continue
            basis_counts[basis] = basis_counts.get(basis, 0) + 1
            out[(key, uuid)] = (t_sol, w["t_b_ms"])
    return (out, m.get("manifest_version", "unknown"),
            dict(sorted(basis_counts.items())))


SCRATCH = Path(os.environ.get("SOLEXBENCH_SCRATCH", "/var/tmp/solbench"))


# -- which part are we scoring? ---------------------------------------------
#
# This script re-times kernels on a real card and scores those times against a
# manifest. If the two are different parts, every score is wrong and nothing
# says so: MI355X timings against MI350X bounds raise S on 1996 of the 2078
# MI355X records on disk, mean 0.6377 -> 0.7214. The manifest used to default to
# `artifacts/09/manifest-v1.json` -- MI350X's frozen release manifest -- so on
# the MI355X node the default WAS the defect. There is no default now, and a
# part that cannot be resolved is a refusal rather than a warning: the whole
# point is that this failure is invisible in the output.

def _repo_relative(p: Path) -> str:
    """Repo-relative when it is in the repo, absolute otherwise.

    `relative_to` raises, and raising here would discard the whole aggregation
    after every re-time has been paid for. Now that `--manifest` is required a
    caller can legitimately name one outside the tree.
    """
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(p).resolve())


def _part_claims(doc, where: str) -> dict[str, str]:
    """Every part *claim* in a document, keyed by where it was written.

    Union, never substitution. Two conventions exist in the tree -- a top-level
    `part` (13 artifacts, including `artifacts/03/t_sol.json`, the one part
    check that is currently doing real work) and `_provenance.part` -- plus the
    torch device names as evidence. Replacing one reader with the other kills a
    live guard; reading all of them and refusing when they disagree is the
    detector.
    """
    if not isinstance(doc, dict):
        return {}
    prov = doc.get("_provenance") or {}
    claims: dict[str, str] = {}
    if isinstance(doc.get("part"), str) and doc["part"]:
        claims[f"{where} top-level part"] = doc["part"]
    if isinstance(prov.get("part"), str) and prov["part"]:
        claims[f"{where} _provenance.part"] = prov["part"]
    # `or []` is load-bearing: `detected_part(None)` asks the LOCAL cards, and a
    # document that names no device must yield no claim rather than this host's.
    dev = detected_part((prov.get("torch") or {}).get("devices") or [])
    if dev:
        claims[f"{where} device name"] = dev
    return claims


def _agree(claims: dict[str, str], what: str) -> tuple[str | None, str | None]:
    """`(part, error)`. One distinct claim is an answer; anything else is not."""
    distinct = sorted(set(claims.values()))
    if len(distinct) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(claims.items()))
        return None, f"{what} names more than one part: {detail}"
    if not distinct:
        return None, f"{what} does not say which part it is for"
    return distinct[0], None


def _container_detected_part(gpu: int) -> str | None:
    """The part of the card this run will measure on, asked inside the container.

    The host python has no torch -- a design property of this driver, see the
    module docstring -- so `detected_part()` here returns None on the very node
    the timings come from. The measurement container has torch and can name the
    card, the same way `_foreign_on_card` asks it which processes hold the card.
    Degrades to None: a probe that could not run is not evidence, and the caller
    refuses on None rather than guessing.
    """
    try:
        proc = subprocess.run(
            [str(ROOT / "env" / "solb"), "python", "/work/scripts/provenance.py",
             "--detect-part"],
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()),
                 "HIP_VISIBLE_DEVICES": str(gpu)},
            capture_output=True, text=True, timeout=300)
    except Exception:                                         # noqa: BLE001
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def node_claims(gpu: int, declared: str | None, retimed_dir: Path) -> dict[str, str]:
    """Every claim about the part THIS RUN's timings are taken on.

    Four sources, in the order they become available: the flag, this process,
    the measurement container, and the run's own existing re-times. The last is
    the only *measured* one and it is what `leaderboard/ingest.py:774 run_part()`
    already uses; it also makes `--reuse-retimed` resolvable on a host with no
    docker, which is how five of the seven runs on disk were scored.
    """
    claims: dict[str, str] = {}
    if declared:
        claims["--part"] = declared
    here = detected_part()
    if here:
        claims["this process"] = here
    else:
        container = _container_detected_part(gpu)
        if container:
            claims["measurement container"] = container
    prior: dict[str, str] = {}
    for f in sorted(retimed_dir.glob("*.json")):
        try:
            part = artifact_part(json.loads(f.read_text()))
        except Exception:                                     # noqa: BLE001
            continue
        if part:
            prior[f"retimed/{f.name}"] = part
    if len(set(prior.values())) > 1:
        # Keep them all, so `_agree` reports the split rather than hiding it.
        claims.update(prior)
    elif prior:
        claims["existing retimed/"] = next(iter(prior.values()))
    return claims

# `run.json.harness` -> what to call it on the board. A harness not listed here
# is printed as it named itself rather than mapped to a default: an unknown
# harness is a fact about the run, and the old default ("Claude Code agent")
# was applied to every run this script ever scored, including one that was not.
HARNESS_NAMES = {
    "claude-code": "Claude Code agent",
    "codex": "codex-cli agent",
    "codex-cli": "codex-cli agent",
}


def _foreign_on_card(gpu: int) -> list[str]:
    """Anything on the authoritative card that is not this process tree.

    Runs the check INSIDE the measurement container, because resolving which
    KFD device HIP index *gpu* refers to needs torch and the host interpreter
    has none. A host-side guess would name the wrong card -- `gpu_map.py`
    reports `torch -> rocm-smi {0: 3}` here, so "GPU 0" means two different
    pieces of silicon depending on who is asking.
    """
    try:
        proc = subprocess.run(
            [str(ROOT / "env" / "solb"), "python",
             "/work/scripts/gpu_exclusive.py", "--gpu", str(gpu)],
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()),
                 "HIP_VISIBLE_DEVICES": str(gpu)},
            capture_output=True, text=True, timeout=120)
    except Exception as exc:                              # noqa: BLE001
        return [f"check failed: {type(exc).__name__}: {exc}"]
    if proc.returncode == 0:
        return []
    lines = [ln.strip() for ln in proc.stderr.splitlines() if ln.strip()]
    return lines or [f"check exited {proc.returncode}"]


#: How long to wait for the authoritative card to come free before measuring
#: anyway. Bounded on purpose: blocking forever turns "somebody else is on the
#: card" into "the sweep stopped and nobody knows why", which is a worse failure
#: than a measurement that says on its face that it was not exclusive.
CARD_WAIT_S = float(os.environ.get("SOLEXBENCH_CARD_WAIT_S", "900"))


def _await_exclusive_card(gpu: int, max_wait: float = CARD_WAIT_S) -> tuple[list[str], float]:
    """Wait for the authoritative card, then report what it looked like.

    Returns `(foreign_at_measurement_time, seconds_waited)`.

    Recording that a card was shared is not the same as not sharing it. On
    2026-08-10 the first version of this only recorded, and 52 of 112 re-timed
    problems came back marked non-exclusive -- another team's sglang and
    Megatron container, which comes and goes in bursts rather than holding the
    card. Bursty is the case where waiting works: the card was clean on 30 of
    30 samples taken minutes later.

    Contention makes a measurement SLOWER, so the scores it produces are
    understated rather than inflated. That is the safe direction and it is
    still wrong: T_b was measured on an idle card, and a T_k that was not is
    not comparable to it.
    """
    started = time.time()
    while True:
        foreign = _foreign_on_card(gpu)
        if not foreign:
            return [], round(time.time() - started, 1)
        waited = time.time() - started
        if waited >= max_wait:
            print(f"    card still shared after {waited:.0f}s; measuring anyway "
                  f"and marking it", flush=True)
            return foreign, round(waited, 1)
        if waited == 0 or int(waited) % 60 < 10:
            print(f"    waiting for the authoritative card ({len(foreign)} "
                  f"foreign process(es), {waited:.0f}s)", flush=True)
        time.sleep(10)


def retime(problem_key: str, kernel: Path, out: Path, gpu: int,
           iterations: int, warmup: int, timeout: int,
           tolerance_root: str) -> dict:
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

    # Is the authoritative card ours alone, at the moment this measurement
    # starts? The scheduler reservation binds the fleet's placer and has no
    # authority over a container somebody started by hand -- on 2026-08-10 the
    # sampling guard caught another team's sglang and Megatron work on GPU 0
    # five times, from a container running with /dev/kfd and no device
    # restriction (STATE.md D29). Checked here rather than only observed,
    # because this is the last point where the answer can still change what
    # gets published. Recorded either way: a refusal that leaves no trace is
    # indistinguishable from a run nobody attempted.
    foreign, waited = _await_exclusive_card(gpu)
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
        "SOLEXBENCH_WORKLOADS_ROOT": tolerance_root,
    }
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout)
        rc, err = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        rc, err = -1, f"timed out after {timeout}s"

    if staged.exists():
        payload = json.loads(staged.read_text())
        # Correctness tolerances are calibrated per part.  Keep the selected
        # tree on the measurement itself so reuse cannot silently preserve an
        # old verdict made against another part's gate.
        payload["tolerance_root"] = tolerance_root
        payload["authoritative_card_exclusive"] = not foreign
        payload["authoritative_card_waited_s"] = waited
        if foreign:
            payload["authoritative_card_shared_with"] = foreign
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1))
        return payload
    # A silent empty result reads exactly like "the kernel failed", which is a
    # different and much less alarming statement than "the runner never ran".
    return {"ok": False,
            "error": f"runner produced no artifact (rc={rc})",
            "stderr_tail": (err or "")[-3000:],
            "per_workload": [], "workloads": 0, "passed": 0,
            "tolerance_root": tolerance_root}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=lambda p: Path(p).resolve(),
                    help="artifacts/10/<run-id> (must contain run.json)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="the scoring manifest. Scores are only comparable "
                         "within one version; the version used is stamped into "
                         "scored.json so two runs cannot be compared across "
                         "manifests without somebody noticing. REQUIRED: this "
                         "used to default to artifacts/09/manifest-v1.json, "
                         "which is MI350X's, so running here without the flag "
                         "scored MI355X timings against another part's bounds.")
    ap.add_argument("--part", default=None,
                    help="declare the part this node measures on, for when it "
                         "cannot be detected (no torch here and no container). "
                         "Checked against every piece of evidence there is; a "
                         "declaration the cards contradict is a refusal.")
    ap.add_argument("--reuse-retimed", action="store_true",
                    help="re-derive scores from existing retimed/*.json "
                         "without touching the GPU")
    ap.add_argument("--only", action="append", default=[], metavar="PROBLEM",
                    help="process ONLY these problems and ignore the rest of "
                         "the run. For scoring a sweep that is still going: "
                         "`sbt collect` writes a session for every job it has "
                         "a record of, including ones still running, and their "
                         "sandbox holds whatever the agent has written so far. "
                         "Re-timing one of those measures a kernel mid-edit. "
                         "Repeatable.")
    ap.add_argument("--retime", action="append", default=[], metavar="PROBLEM",
                    help="force a fresh re-time for this problem even under "
                         "--reuse-retimed. Repeatable. For when a harness "
                         "defect invalidated some measurements and not others: "
                         "re-timing the whole run to fix five problems would "
                         "move 215 numbers that had nothing wrong with them, "
                         "and every one of those movements would be noise "
                         "somebody has to explain.")
    ap.add_argument("--exclude-from-leaderboard", action="store_true",
                    help="score and record, but do not list on the board "
                         "(for validation runs)")
    a = ap.parse_args()

    run = json.loads((a.run / "run.json").read_text())
    b, manifest_version, bound_basis = bounds(a.manifest)
    retimed_dir = a.run / "retimed"
    retimed_dir.mkdir(parents=True, exist_ok=True)

    # Fail closed, and BEFORE the first re-time: a refusal after four hours on
    # the card costs four hours, and every piece of evidence this needs is
    # already on disk or one cheap probe away.
    m_part, m_err = _agree(
        _part_claims(json.loads(a.manifest.read_text()), "manifest"), "the manifest")
    n_claims = node_claims(a.gpu, a.part, retimed_dir)
    n_part, n_err = _agree(n_claims, "this node")
    evidence = (", ".join(f"{k}={v}" for k, v in sorted(n_claims.items()))
                or "none")
    for err in (m_err, n_err):
        if err:
            print(f"REFUSING to score: {err}.\n"
                  f"  manifest: {a.manifest}\n"
                  f"  node evidence: {evidence}\n"
                  f"  A score pairs a measured time with a bound; if the two "
                  f"are different parts every score is wrong and nothing in the "
                  f"output says so. Pass --part for the node, or score against "
                  f"a manifest that names its own.", file=sys.stderr)
            return 3
    if m_part != n_part:
        print(f"REFUSING to score: the manifest is {m_part} and this node is "
              f"{n_part}.\n"
              f"  manifest: {a.manifest}\n"
              f"  node evidence: {evidence}\n"
              f"  Scoring {n_part} timings against {m_part} bounds inflates S "
              f"(measured: +0.08 mean over 2078 records) and no check "
              f"downstream can tell.", file=sys.stderr)
        return 3
    try:
        tolerance_root = container_tolerance_root(n_part)
    except ValueError as exc:
        print(f"REFUSING to score: {exc}.", file=sys.stderr)
        return 3
    part_source = ("declared" if "--part" in n_claims else "detected")
    print(f"part {n_part} ({part_source}); manifest {a.manifest.name} "
          f"({manifest_version}); tolerances {tolerance_root}", flush=True)

    # The sandboxes live in /var/tmp and will be swept. A score whose kernel no
    # longer exists cannot be reproduced or disputed, so the source is copied
    # next to the number that came from it.
    kernels_dir = a.run / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    # Consumed as it matches, so a typo in a problem name is an error rather
    # than a silent no-op that looks exactly like "re-timed and nothing moved".
    only = set(a.only)
    unknown_only = only - set(run["sessions"])
    if unknown_only:
        print(f"--only names no session in this run: {sorted(unknown_only)}",
              file=sys.stderr)
        return 2

    force_retime = set(a.retime)
    unknown = force_retime - set(run["sessions"])
    if unknown:
        print(f"--retime names no session in this run: {sorted(unknown)}",
              file=sys.stderr)
        return 2

    results: list[dict] = []
    per_problem: dict[str, dict] = {}
    t0 = time.time()

    for key, sess in sorted(run["sessions"].items()):
        if only and key not in only:
            continue
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
        forced = key in force_retime
        if forced:
            force_retime.discard(key)
        reused = a.reuse_retimed and existing.exists() and not forced
        if reused:
            # Re-deriving scores from a completed re-time must not need the GPU
            # again: the timing is the expensive part and it does not change.
            ev = json.loads(existing.read_text())
            print(f"[{key}] reusing re-time from {existing.name}", flush=True)
        else:
            print(f"[{key}] re-timing on GPU {a.gpu} ...", flush=True)
            ev = retime(key, kernel, existing, a.gpu,
                        a.iterations, a.warmup, a.timeout, tolerance_root)
        measured = artifact_part(ev) if isinstance(ev, dict) else None
        measured_root = recorded_tolerance_root(ev, measured)
        if measured_root != tolerance_root:
            action = "reuse" if reused else "score"
            found = repr(measured_root) if measured_root is not None else "no stamp"
            print(f"REFUSING to {action}: [{key}] tolerance_root is {found}, "
                  f"but {n_part} requires {tolerance_root!r}. Re-time this "
                  f"problem with the current scorer; an unstamped artifact "
                  f"cannot prove which correctness gate produced its verdict.",
                  file=sys.stderr)
            return 5
        # The measurement's own account of the card it ran on, checked against
        # the part resolved before the sweep started. This is the only evidence
        # that can contradict a `--part` a human typed, and it arrives one
        # re-time at a time; refusing here loses the aggregation, never a
        # timing -- every re-time is already on disk and `--reuse-retimed`
        # picks them all up again.
        if measured and measured != n_part:
            print(f"REFUSING to score: [{key}] was measured on {measured} but "
                  f"this run resolved to {n_part} against a {m_part} manifest. "
                  f"The timings are kept in {retimed_dir}; re-run with the "
                  f"matching manifest.", file=sys.stderr)
            return 4
        rec["measured_part"] = measured
        rec["ok"] = ev.get("ok")
        rec["error"] = ev.get("error")
        # Which harness produced this timing. A run can carry timings from two
        # harness versions -- five of glm-sweep-2's problems were re-timed on
        # 2026-08-10 after the side-stream timing defect (STATE.md D38) and the
        # other 215 were not -- and `forced_retime` above only records the
        # invocation that did it. Reading it back per problem means a later
        # re-derivation of scores does not lose the fact.
        rec["retimed_git_sha"] = (
            (ev.get("_provenance") or {}).get("git_sha") if isinstance(ev, dict) else None
        )

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
        # Declared, not detected: this driver runs on the host python, which has
        # no torch, so a detection here is None on the very node the timings
        # came off. `n_part` was resolved from the flag, the container and the
        # re-times themselves, and every one of those agreed.
        **stamp("10-agent-scored", part=n_part),
        "part": n_part,
        "part_source": part_source,
        "part_claims": n_claims,
        "run_id": run.get("run_id"),
        "model": run.get("model"),
        # Which bounds produced these scores. A score is only meaningful inside
        # one manifest version -- v1.1 corrected 1,048 of them -- so the number
        # and the version it was computed against travel together.
        "manifest_version": manifest_version,
        "manifest_path": _repo_relative(a.manifest),
        # The correctness gate paired with these timings and bounds.  This is
        # independently checked on every re-time above because historical
        # MI355X scores were accidentally evaluated through the MI350X tree.
        "tolerance_root": tolerance_root,
        # Which millisecond column each of those bounds came out of
        # (`bound_headroom.published_bound_ms`). {"published": N} is a run scored
        # against the manifest's own published bound; any `legacy_*` count is a
        # run scored against a column with no clock on it, which is not
        # comparable with the first and is why this is written down rather than
        # inferred.
        "bound_basis": bound_basis,
        # Which problems were measured again in THIS invocation. When only some
        # were, the run carries timings from two harness versions and that is
        # worth being able to read off the artifact rather than reconstructing
        # from file mtimes.
        "forced_retime": sorted(a.retime),
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
            # The cap's DURATION is stated only when the run recorded one.
            # pilot8 marks problems `capped` and records no `wall_cap_seconds`,
            # and the sentence used to interpolate that None straight into a
            # `:g` -- a TypeError the moment anything rescored that run, and
            # before the `:g` it would have published "the harness's Nones
            # wall-clock cap". How many were stopped is known; what the cap was
            # is not, and the note now says only the part that is.
            + ((f" {capped} of them were stopped by the harness's "
                + (f"{cap_seconds:g}s " if cap_seconds else "")
                + "wall-clock cap rather than by the agent deciding it was "
                  "done.") if capped else "")
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
