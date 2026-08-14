#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 10 — score harvested agent solutions from a pristine tree.

    python scripts/score_solutions.py --run-id pilot-01

For every session in a run: load the solution the agent left behind, screen it,
re-evaluate it on the **authoritative GPU** one problem at a time, and write one
score record per workload.

Why re-evaluate at all, when ``./verify`` already ran it? Three reasons, and each
of them has bitten this project or its upstream:

1. ``./verify`` ran on a pool GPU with seven busy neighbours. Task 01 measures
   how much that matters; whatever the answer, authoritative timing is pinned to
   one GPU and every timing artifact records which (STATE.md, *Decisions taken*).
2. The agent could have influenced its own verification. It runs as a local
   process with a writable filesystem, so the only defensible position is that
   nothing it produced is trusted to score itself.
3. ``./verify`` is capped at a handful of attempts, so the last thing an agent
   ran is often not the last thing it *wrote*.

Scoring is deliberately serial. Two evaluations sharing a GPU inflate each other
and, as deviation D11 records, nothing in the output would say so.


WHICH CARD A SOLUTION IS SCORED ON, AND WHY IT IS NOT A FREE CHOICE
-------------------------------------------------------------------
On MI350X this script took a single ``--gpu`` and that was correct: the whole
authoritative T_b pass was pinned to GPU 0, so every anchor came off one card
and a T_k measured on that same card was comparable to all of them.

That is no longer how T_b is produced. ``STATE.md`` §4.4 re-times ``T_b`` and
``T_k`` back to back **on the same card**, and as a consequence
``scripts/authoritative_tb.py`` runs 8-way with per-problem card pinning
(``plan[i::8]`` over the sorted plan), stamping a ``card_identity`` block on
every artifact. A problem's anchor therefore lives on a *specific* card. Scoring
its ``T_k`` somewhere else divides a timing taken at one card's clock by a
timing taken at another's -- and on this part the workload-dependent clock
spread is 36.8%, so that is not a rounding error. It is also invisible: the
output looks entirely plausible.

So two things, and the second is the one that matters:

  ``--shard I/N``   the same partition ``authoritative_tb.py`` used, imported
                    from it rather than re-implemented. With ``--shard i/8
                    --gpu i`` on the node that produced the anchors, every
                    solution is re-timed on the card holding its own problem's
                    T_b.
  enforcement       before any solution is scored, the ``card_identity`` on its
                    problem's authoritative artifact is compared against the
                    card this process is actually on. A mismatch -- or an
                    identity that cannot be established on either side -- is
                    **refused and counted**, never scored. Enabling the right
                    layout is not the same as enforcing it; only the refusal
                    makes a cross-card score impossible rather than unlikely.

Enforcement applies exactly when a ``T_b`` is in scope for that problem. Before
the anchor exists there is no card to match, the record is published under the
weaker ``sol_headroom`` basis, and ``scripts/backfill_scores.py`` raises it to
``sol_score_v1`` when the anchor lands -- at which point the same card check is
applied to the raise, with no GPU time spent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402
from solexbench_agents.gpu_pool import AUTHORITATIVE_GPU  # noqa: E402
from solexbench_agents.scoring import (  # noqa: E402
    ScoreBasis,
    clock_lock_state,
    compare_digests,
    headroom_fraction,
    reference_copy_verdict,
    resolve_basis,
    sol_score,
    tree_digest,
)

# The partition and the card probe are IMPORTED from the script that produced
# the anchors, never re-implemented here. Two implementations of "which card
# does problem X belong to" is exactly one more than can be kept in agreement,
# and the failure mode of a disagreement is a silently cross-card score.
from authoritative_tb import card_identity, parse_shard, shard_of  # noqa: E402

# The bound terms and the interval arithmetic, from the one module that owns them.
# `RECLOCK_FIELDS` is imported from the manifest builder for the same reason the
# card probe is imported above: two lists of "the fields needed to re-clock a
# bound" is one more than can be kept in agreement.
from build_manifest import RECLOCK_FIELDS  # noqa: E402
from solexbench_rocm.t_sol_at import (  # noqa: E402
    MissingBoundTerms,
    t_sol_interval,
)

# Likewise the part->path resolver: `verify_artifacts.ArtifactTree` already maps
# (task, part) -> path and already refuses to substitute one part's artifact for
# another's. Duplicating it is how `artifacts/03/t_sol.json` -- MI350X, derived
# at F_LOCK 1300 -- ends up as the default bound on an MI355X node.
from verify_artifacts import (  # noqa: E402
    DEFAULT_PART,
    KNOWN_PARTS,
    ArtifactTree,
    artifact_part,
)


def load_manifest_bounds(path: Path) -> tuple[dict, dict, dict]:
    """(t_sol, t_b, meta) from a frozen scoring manifest.

    Preferred over the raw artifacts, because a score is only meaningful inside
    one manifest version: the README says so, and the manifest is the thing that
    pins `T_SOL`, `T_b` and the tolerances together with the stack that produced
    them. Reading the raw artifacts instead lets a score be assembled from a
    `T_SOL` measured at one clock and a `T_b` measured at another, with nothing
    recording that it happened.

    Returned in the shape ``workload_bound`` already expects, so the caller does
    not care which source it got.

    **The four re-clocking terms come across too**, under the unlocked basis, and
    they have to: ``t_sol_ms`` in the manifest is at the reference clock, and the
    bound this scorer needs is the one evaluated over *this* measurement's own clock
    bracket. Without the terms, `t_sol_at` cannot produce it and the record would
    fall back to a reference-clock bound that belongs to a different frequency —
    plausibly, and only visibly to someone who knew to check.
    """
    if not path.exists():
        return {}, {}, {}
    doc = json.loads(path.read_text())
    problems = doc.get("problems") or {}
    t_sol: dict = {}
    t_b: dict = {}
    for key, entry in problems.items():
        sol_wl, b_wl = {}, {}
        for uuid, w in (entry.get("workloads") or {}).items():
            if w.get("t_sol_ms") is not None:
                sol_wl[uuid] = {"t_sol_ms": w["t_sol_ms"],
                                "t_sol_cycles": w.get("t_sol_cycles"),
                                **{k: w.get(k) for k in RECLOCK_FIELDS}}
            if w.get("t_b_ms") is not None:
                b_wl[uuid] = {"t_b_ms": w["t_b_ms"],
                              "t_b_variant": w.get("t_b_variant")}
        if sol_wl:
            t_sol[key] = {"workloads": sol_wl}
        if b_wl:
            t_b[key] = {"workloads": b_wl}
    meta = {
        "manifest_version": doc.get("manifest_version"),
        "path": str(path),
        "stats": doc.get("stats"),
        "problems_with_t_sol": len(t_sol),
        "problems_with_t_b": len(t_b),
    }
    return t_sol, t_b, meta


def load_bounds(path: Path, part: str | None) -> tuple[dict, dict]:
    """(problems-by-key, meta) from a T_SOL or T_b artifact, or ({}, {}).

    The part check is the whole reason this is a function rather than a
    ``json.load``. ``artifacts/03/t_sol.json`` carries ``part`` in its header,
    and a bound derived at MI350X's F_LOCK of 1300 MHz applied to an MI355X
    measurement would rescale every score by the clock ratio -- plausibly,
    invisibly, and in the direction that flatters the kernel. Prime directive 2.
    """
    if not path.exists():
        return {}, {}
    doc = json.loads(path.read_text())
    meta = {k: v for k, v in doc.items() if k != "problems"}
    if part and doc.get("part") and doc["part"] != part:
        meta["rejected"] = (
            f"artifact was derived on {doc['part']} but this node is {part}; "
            f"not used"
        )
        return {}, meta
    return doc.get("problems", {}) or {}, meta


def workload_bound(bounds: dict, problem_key: str, uuid: str, field: str):
    entry = (bounds.get(problem_key) or {}).get("workloads") or {}
    return (entry.get(uuid) or {}).get(field)


def workload_record(bounds: dict, problem_key: str, uuid: str) -> dict:
    """The whole bound record, which `t_sol_at` needs and `workload_bound` hides."""
    entry = (bounds.get(problem_key) or {}).get("workloads") or {}
    return entry.get(uuid) or {}


def _interval_score(sol_record: dict, bracket: dict | None, t_k, t_b_ms) -> dict:
    """T_SOL and S as intervals over this measurement's own clock bracket.

    Unlocked, the clock moves inside the timed window and the compute half of the
    roofline is a fixed CYCLE count, so T_SOL is a range, not a number. **The
    published bound is the minimum-clock end** -- the largest T_SOL, the tightest
    bound, the one a real measurement can visibly beat if the choice is wrong. The
    reasoning for that direction is in ``solexbench_rocm.t_sol_at`` and is not
    repeated here; what matters at this call site is that ``S`` inherits it.

    S is evaluated at *both* ends and both are recorded beside the published one.
    They bracket it by construction -- the published value is one of the two -- and
    the pair is what tells a reader whether a score of 0.61 means 0.61 or means
    "somewhere in 0.52 to 0.68 depending on what the card was doing".

    Returns ``{}`` when no interval can be formed, rather than a half-populated
    record: the caller then keeps the reference-clock bound it already had, and the
    absence is stated on the record instead of being papered over.
    """
    from sol_execbench.core.bench.clock_bracket import clock_interval

    span = clock_interval(bracket)
    if span is None:
        return {}
    try:
        iv = t_sol_interval(sol_record, *span)
    except (MissingBoundTerms, ValueError):
        # A pre-split bound record, or a nonsense clock. Refused, never inferred:
        # a guessed bound would be wrong only at clocks nobody would think to
        # check, which is the worst place for it.
        return {}
    lo_end, hi_end = iv["t_sol_ms_at_clock_min"], iv["t_sol_ms_at_clock_max"]
    return {
        **iv,
        # S at the two ends of the bound's interval, both against the same T_k and
        # the same T_b. Named by the CLOCK they correspond to, not by magnitude:
        # the min-clock end gives the LARGER T_SOL, and which way that moves S
        # depends on whether T_k is above or below T_b, so a name like "s_low"
        # would be wrong half the time.
        "sol_score_at_clock_min": sol_score(t_k, lo_end, t_b_ms),
        "sol_score_at_clock_max": sol_score(t_k, hi_end, t_b_ms),
    }


# --- which card does this problem's anchor live on? -------------------------

#: The fields that name a physical card, compared all-or-nothing.
#:
#: `hostname` because the same BDF exists on all three nodes; `bdf` because it
#: is the card's position and does not move; `uuid` because a BDF can be
#: re-enumerated across a reboot while the card behind it changes. The torch
#: index is deliberately NOT here -- it is a position in an ordering that
#: changes with HIP_VISIBLE_DEVICES, which is what `authoritative_tb.card_identity`
#: exists to avoid relying on.
CARD_KEYS = ("hostname", "bdf", "uuid")


def anchor_card(tb_dir: Path | None, problem_key: str) -> tuple[dict | None, str]:
    """``(card_identity, note)`` from a problem's authoritative T_b artifact.

    Never guesses. Every way of not finding an identity returns ``None`` and a
    note saying which way it was, because the caller's decision -- refuse -- is
    the same in each case but the triage is not.
    """
    if tb_dir is None:
        return None, "no authoritative T_b directory was given"
    path = Path(tb_dir) / f"{problem_key}.json"
    if not path.exists():
        return None, f"no authoritative T_b artifact at {path}"
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        return None, f"{path} unreadable: {type(exc).__name__}: {exc}"
    card = doc.get("card_identity")
    if not isinstance(card, dict):
        return None, f"{path} carries no card_identity block"
    if not card.get("identified"):
        return None, (f"{path} records a card that could not identify itself: "
                      f"{str(card.get('error'))[:200]}")
    return card, f"from {path}"


def card_verdict(anchor: dict | None, actual: dict | None, *,
                 t_b_in_scope: bool, anchor_note: str = "") -> dict:
    """Is this process allowed to score this problem?

    ``ok`` False means **refuse the record**, not "warn and continue". The whole
    point of §4.4 is that ``T_b`` and ``T_k`` share a card; a score that pairs
    two cards is wrong by a factor nothing downstream can recover, and it looks
    exactly like a correct one.

    When no ``T_b`` is in scope there is nothing to match, so the check is
    ``not_applicable`` rather than passing -- a distinction the summary keeps,
    so that "0 refusals" cannot mean "nothing was checked".
    """
    if not t_b_in_scope:
        return {"ok": True, "state": "not_applicable",
                "reason": "no T_b for this problem in the bounds being used; "
                          "the record is published on a T_b-free basis and "
                          "carries no cross-card claim",
                "anchor": None, "actual": actual}
    if anchor is None:
        return {"ok": False, "state": "unverifiable",
                "reason": f"a T_b is in scope but its card cannot be "
                          f"established ({anchor_note}); refusing rather than "
                          f"assuming it was this one",
                "anchor": None, "actual": actual}
    if not actual or not actual.get("identified"):
        return {"ok": False, "state": "unverifiable",
                "reason": "this process's own card could not identify itself; "
                          f"refusing rather than assuming it matches "
                          f"{anchor.get('bdf')}",
                "anchor": anchor, "actual": actual}
    missing = [k for k in CARD_KEYS
               if not anchor.get(k) or not actual.get(k)]
    if missing:
        return {"ok": False, "state": "unverifiable",
                "reason": f"card identity incomplete on {missing}; a partial "
                          f"match is not a match",
                "anchor": anchor, "actual": actual}
    differing = [k for k in CARD_KEYS if str(anchor[k]) != str(actual[k])]
    if differing:
        return {"ok": False, "state": "mismatch",
                "reason": (
                    "this problem's T_b was measured on "
                    f"{anchor.get('hostname')} {anchor.get('bdf')} "
                    f"(uuid {anchor.get('uuid')}) but this process is on "
                    f"{actual.get('hostname')} {actual.get('bdf')} "
                    f"(uuid {actual.get('uuid')}); differing on {differing}. "
                    "STATE.md 4.4 re-times T_b and T_k on ONE card -- scoring "
                    "across two divides timings taken at two different clocks "
                    "(36.8% workload-dependent spread on this part) and the "
                    "result is indistinguishable from a real speedup. Re-run "
                    "this shard on the card that holds the anchor."),
                "anchor": anchor, "actual": actual}
    return {"ok": True, "state": "matched",
            "reason": f"T_b and T_k on {actual.get('hostname')} "
                      f"{actual.get('bdf')}",
            "anchor": anchor, "actual": actual}


def anchor_shard(tb_dir: Path | None, problem_key: str, count: int) -> int | None:
    """The shard index `authoritative_tb.py` actually used for this problem.

    Read from the artifact rather than recomputed. `plan[i::N]` is a pure
    function of the *plan*, and the plan excludes problems whose candidates had
    no passing variant -- so recomputing the stride from the dataset's problem
    list would drift from the real assignment exactly where a problem was
    dropped. The artifact records what happened; that is what to follow.

    ``None`` when the artifact is absent, unsharded, or was produced under a
    different shard count (a 4-way pass tells you nothing about an 8-way one).
    """
    if tb_dir is None:
        return None
    path = Path(tb_dir) / f"{problem_key}.json"
    if not path.exists():
        return None
    try:
        sh = (json.loads(path.read_text()) or {}).get("shard")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(sh, dict) or sh.get("count") != count:
        return None
    idx = sh.get("index")
    return idx if isinstance(idx, int) else None


def partition_problems(keys: list[str], shard: tuple[int, int] | None,
                       tb_dir: Path | None) -> list[str]:
    """The problems this shard owns: the anchors' own assignment, first.

    Two sources, in priority order, and both are exact partitions:

    1. a problem whose authoritative artifact records ``shard.index`` goes to
       that shard -- the card that holds its T_b, by construction; and
    2. a problem with no such record has no anchor and therefore no card to
       match, so it falls back to `authoritative_tb.shard_of` over the sorted
       remainder -- the same striding convention, imported from the same place,
       so the two never diverge.

    The union over ``I = 0..N-1`` is exactly ``keys``, with nothing duplicated
    and nothing dropped. Coverage is a property of the merged output directory,
    never of one shard (CLAUDE.md 0).
    """
    keys = sorted(keys)
    if shard is None:
        return keys
    index, count = shard
    recorded: dict[str, int] = {}
    for k in keys:
        got = anchor_shard(tb_dir, k, count)
        if got is not None:
            recorded[k] = got
    unrecorded = [k for k in keys if k not in recorded]
    mine = [k for k in keys if recorded.get(k) == index]
    return sorted(mine + shard_of(unrecorded, shard))


def _spec_divergence(packet: Path, authoritative_def: Path,
                     authoritative_wl: Path) -> list[str]:
    """What, if anything, the packet's copy of the problem no longer matches.

    Reported per field rather than as one boolean, because the fields are not
    equally serious: a changed ``reference`` redefines what "correct" means, while
    a changed ``description`` is cosmetic.
    """
    divergence: list[str] = []
    try:
        want = json.loads(authoritative_def.read_text())
        have = json.loads((packet / "definition.json").read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"definition.json unreadable: {type(exc).__name__}: {exc}"]

    for field in sorted(set(want) | set(have)):
        if want.get(field) != have.get(field):
            divergence.append(f"definition.{field}")

    try:
        want_lines = [ln for ln in authoritative_wl.read_text().splitlines()
                      if ln.strip()]
        have_lines = [ln for ln in (packet / "workload.jsonl").read_text().splitlines()
                      if ln.strip()]
        if want_lines != have_lines:
            divergence.append("workload.jsonl")
    except Exception as exc:  # noqa: BLE001
        divergence.append(f"workload.jsonl unreadable: {type(exc).__name__}: {exc}")
    return divergence


def score_one(session_dir: Path, problem_dir: Path, workloads_root: Path | None,
              *, t_sol: dict, t_b: dict, part: str | None, timeout: int,
              gpu: int, lock_clocks: bool = True,
              card_check: dict | None = None) -> dict:
    """Evaluate one harvested solution against the AUTHORITATIVE problem spec.

    The problem — ``definition.json`` and ``workload.jsonl`` — is read from the
    dataset and the tolerance tree, **never from the packet**, and the packet's
    copies are diffed against them and any difference reported.

    This is not defensive decoration. An earlier version loaded the definition
    from the packet, and a submission was found having rewritten the reference
    inside it to work around a broken interpreter (STATE.md D24). Editing the
    reference edits the definition of correctness: in the limit a submission can
    replace the reference with its own kernel and score 100% on every workload.
    Only ``solution.json`` and its sources may come from the packet.
    """
    from _common import evaluate, summarize  # noqa: E402

    from sol_execbench.core import BenchmarkConfig, Definition, Workload  # noqa: E402
    from sol_execbench.core.bench.reward_hack import static_source_screen  # noqa: E402

    from agent_verify import load_solution  # noqa: E402

    # `benchmark_reference=True` because without it reference_latency_ms stays 0.0
    # and there is no speed axis at all -- not even speedup-vs-reference, which is
    # the strongest basis available until T_b exists.
    #
    # `lock_clocks=True` asserts the clock rather than assuming it. Every latency
    # here is quoted at F_LOCK, and an unlocked GPU would produce numbers that
    # look identical and mean something else.
    #
    # It is passed False only on the unlocked basis (STATE.md, MI355X clock
    # methodology), where there IS no F_LOCK to assert and the guarantee is the
    # per-window clock bracket instead -- which `main` verifies is switched on,
    # and which is enforced per record below. That is a different assertion, not
    # a dropped one: with `lock_clocks=True` on an unlocked node the eval driver
    # rejects every workload, so this is also the difference between scoring and
    # not scoring at all.
    config = BenchmarkConfig(benchmark_reference=True, lock_clocks=lock_clocks)

    packet = session_dir / "packet"
    session = json.loads((session_dir / "session.json").read_text())

    # The packet's own manifest is preferred, but it is not always there: in
    # `artifacts/10/runs/full-01` only 13 of 404 harvested packets kept a
    # `.packet.json`, and reading it unconditionally turns 391 sessions into
    # `scorer_error`. The session directory name carries the same key -- `main`
    # already routes on it -- so it is the fallback, recorded as such.
    #
    # When both exist and disagree, that is refused rather than resolved: a
    # packet claiming to be a different problem than the directory it sits in
    # is the D24 failure mode (a submission editing its own definition of
    # correct), and picking a winner would decide it silently.
    dir_key = session_dir.name
    manifest_path = packet / ".packet.json"
    packet_key = None
    if manifest_path.exists():
        try:
            packet_key = (json.loads(manifest_path.read_text())
                          .get("problem", "")).replace("/", "__") or None
        except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
            packet_key = None
            session.setdefault("_packet_manifest_error",
                               f"{type(exc).__name__}: {exc}")
    problem_key = packet_key or dir_key

    result: dict = {
        "problem": problem_key,
        "category": problem_key.split("__", 1)[0],
        "harness": session["harness"],
        "model": session.get("model"),
        "gpu": gpu,
        # Which physical card produced these timings, and how that was checked
        # against the card holding this problem's T_b. Recorded even when the
        # check was not applicable, so a reader can tell "checked and matched"
        # from "nothing to check" without inferring it (prime directive 5).
        "card_check": card_check,
        "clock_basis": "locked" if lock_clocks else "unlocked",
        "packet_manifest": ("present" if packet_key else "missing; problem key "
                            "taken from the session directory name"),
        "agent": {
            "wallclock_s": session.get("wallclock_s"),
            "cost_usd": session.get("cost_usd"),
            "cost_source": session.get("cost_source"),
            "input_tokens": session.get("input_tokens"),
            "output_tokens": session.get("output_tokens"),
            "reasoning_tokens": session.get("reasoning_tokens"),
            "num_turns": session.get("num_turns"),
            "verify_attempts": session.get("verify_attempts"),
            "timed_out": session.get("timed_out"),
            "harness_error": session.get("error"),
        },
        "records": [],
    }

    if packet_key and packet_key != dir_key:
        result["outcome"] = "refused_packet_problem_mismatch"
        result["note"] = (
            f"the packet's .packet.json claims problem {packet_key!r} but it "
            f"sits in the directory for {dir_key!r}. Scoring either one would "
            f"be a guess about which spec this solution was written against.")
        return result

    # Resolved before the early return, so tampering is reported even for a
    # session that produced nothing.
    category, name = problem_key.split("__", 1)
    authoritative_def = problem_dir / "definition.json"
    authoritative_wl = (
        workloads_root / category / name / "workload.jsonl"
        if workloads_root else problem_dir / "workload.jsonl"
    )
    result["spec_source"] = {
        "definition": str(authoritative_def),
        "workloads": str(authoritative_wl),
    }
    result["packet_spec_divergence"] = _spec_divergence(
        packet, authoritative_def, authoritative_wl
    )
    if result["packet_spec_divergence"]:
        # Loud, and it does not stop the scoring: the authoritative spec is what
        # gets used either way, so the honest thing is to score against it and
        # record that the packet disagreed.
        result["spec_tampered"] = True

    if not session.get("produced_solution"):
        result["outcome"] = "no_solution"
        result["note"] = "the agent left no solution.json"
        return result

    definition = Definition(**json.loads(authoritative_def.read_text()))
    raw_workloads = [json.loads(ln) for ln
                     in authoritative_wl.read_text().splitlines()
                     if ln.strip()]
    workloads = [Workload(**w) for w in raw_workloads]

    try:
        solution = load_solution(packet)
    except Exception as exc:  # noqa: BLE001
        result["outcome"] = "invalid_solution"
        result["note"] = f"{type(exc).__name__}: {exc}"
        return result

    result["languages"] = [x.value for x in solution.spec.languages]
    result["n_sources"] = len(solution.sources)
    result["source_bytes"] = sum(len(s.content or "") for s in solution.sources)

    # Reported, not raised: a finding is a result to record, and the sweep must
    # not die on one submission.
    findings = static_source_screen(solution.sources)
    result["static_screen"] = [
        {"path": p, "pattern": pat, "why": why} for p, pat, why in findings
    ]
    copy = reference_copy_verdict(solution.sources, definition.reference)
    result["reference_copy"] = {"kind": copy.kind, "similarity": copy.similarity,
                                "detail": copy.detail}

    if findings:
        # Zero regardless of what it measures. A submission that trips the screen
        # is not a slow kernel, it is a kernel measured under conditions the
        # benchmark does not permit.
        result["outcome"] = "rejected_static_screen"
        return result

    started = time.time()
    try:
        traces = evaluate(definition, workloads, solution, config=config,
                          timeout=timeout)
        summary = summarize(traces)
    except Exception as exc:  # noqa: BLE001
        result["outcome"] = "eval_failed"
        result["note"] = f"{type(exc).__name__}: {exc}"
        result["eval_wallclock_s"] = time.time() - started
        return result

    result["eval_wallclock_s"] = time.time() - started
    result["outcome"] = "evaluated"

    for i, w in enumerate(summary["per_workload"]):
        uuid = w.get("workload_uuid") or (
            raw_workloads[i].get("uuid") if i < len(raw_workloads) else None
        )
        t_k = w.get("latency_ms")
        t_ref = w.get("reference_latency_ms")
        t_sol_ms = workload_bound(t_sol, problem_key, uuid, "t_sol_ms")
        t_b_ms = workload_bound(t_b, problem_key, uuid, "t_b_ms")
        correct = w["status"] == "PASSED"

        # Unlocked basis: a window with NO clock sample at all has no defensible
        # latency, so it makes no timing claim. Same reading
        # `build_manifest.collect_t_b` applies to an anchor -- an unknown clock
        # is not a permissive one -- applied to T_k. The record still exists and
        # still reports correctness; it simply drops to a basis with no clock in
        # it, and says why.
        #
        # **A bracket refused for SPREAD no longer lands here.** It used to: the
        # test was `has_clock_evidence`, which requires a single defensible
        # frequency, and a window that moved from 1607 to 2148 MHz has none. Under
        # the interval methodology it has two, the bound is evaluated at both, and
        # the measurement is published with a width instead of being thrown away.
        # The refusal is still recorded, still counted, and still filterable -- it
        # is a quality label now rather than a gate. What stays a gate is an
        # ABSENT clock, immediately below.
        bracket = w.get("clock_bracket")
        ref_bracket = w.get("reference_clock_bracket")
        clock_refused = None
        clock_refusal_reason = None
        interval: dict = {}
        if not lock_clocks:
            from sol_execbench.core.bench.clock_bracket import (  # noqa: E402
                has_clock_interval,
            )
            for label, br in (("solution", bracket), ("reference", ref_bracket)):
                # The reference arm matters too: `headroom_fraction` and
                # `speedup_vs_reference` divide by T_ref, so a refused
                # reference bracket poisons those exactly as a refused
                # solution bracket poisons T_k.
                if br is None and label == "reference" and t_ref is None:
                    continue
                if not has_clock_interval(br):
                    clock_refused = True
                    clock_refusal_reason = (
                        f"{label} window carries no clock samples at all: "
                        f"{(br or {}).get('clock_bracket_refused_reason') or 'absent'}"
                    )
                    break
            if clock_refused is None:
                clock_refused = False

        if clock_refused:
            basis = ScoreBasis.CORRECTNESS_ONLY
            s = hf = None
        else:
            if not lock_clocks:
                # The bound this T_k is judged against is the one over T_k's OWN
                # window, not the reference-clock figure the manifest carries and
                # not the one derived over the T_b window. Where an interval can
                # be formed, `t_sol_ms` below becomes the published (minimum-clock)
                # end, so every downstream consumer -- S, headroom, the
                # `bound_violation` check -- sees the same bound.
                interval = _interval_score(
                    workload_record(t_sol, problem_key, uuid), bracket, t_k, t_b_ms)
                if interval:
                    t_sol_ms = interval["t_sol_ms_published"]
            basis = resolve_basis(correct=correct, t_k_ms=t_k, t_ref_ms=t_ref,
                                  t_sol_ms=t_sol_ms, t_b_ms=t_b_ms)
            s = sol_score(t_k, t_sol_ms, t_b_ms) if correct else None
            hf = headroom_fraction(t_k, t_ref, t_sol_ms) if correct else None
        if not correct:
            # S is not computed for an incorrect kernel, so neither end of its
            # interval means anything either. Dropped rather than left as a pair of
            # numbers beside a null score.
            interval = {k: v for k, v in interval.items()
                        if not k.startswith("sol_score_at_")}

        tol = raw_workloads[i].get("tolerance", {}) if i < len(raw_workloads) else {}
        record = {
            "workload_index": i,
            "workload_uuid": uuid,
            "status": w["status"],
            "correct": correct,
            "t_k_ms": t_k,
            "t_ref_ms": t_ref,
            "speedup_vs_reference": (t_ref / t_k) if (t_k and t_ref) else None,
            "t_sol_ms": t_sol_ms,
            "t_sol_cycles": workload_bound(t_sol, problem_key, uuid, "t_sol_cycles"),
            "t_b_ms": t_b_ms,
            "sol_score": s,
            "headroom_fraction": hf,
            "score_basis": basis.value,
            # Which card this T_k was measured on and whether it is the card
            # this problem's T_b came off. On every record, not only the
            # refused ones: a score that does not say what it was compared
            # against invites being compared against anything.
            "card_check_state": (card_check or {}).get("state"),
            "clock_bracket": bracket,
            "reference_clock_bracket": ref_bracket,
            "clock_bracket_refused": clock_refused,
            "max_absolute_error": w.get("max_absolute_error"),
            "max_relative_error": w.get("max_relative_error"),
            "allowed_atol": tol.get("max_atol"),
            "allowed_rtol": tol.get("max_rtol"),
            "has_nan": w.get("has_nan"),
            "has_inf": w.get("has_inf"),
            "methodology": w.get("methodology"),
            "failure_log": w.get("log") or "",
        }
        # A kernel faster than its own analytic lower bound is impossible; the
        # bound is wrong. Surfaced per record rather than clamped, because
        # clamping would hide exactly the thing worth fixing (cf. D12).
        if correct and t_k and t_sol_ms and t_k < t_sol_ms:
            record["bound_violation"] = (
                f"t_k {t_k:.6g} ms is below t_sol {t_sol_ms:.6g} ms"
            )
        if clock_refusal_reason:
            record["score_refused"] = clock_refusal_reason
        result["records"].append(record)

    result["workloads"] = summary["workloads"]
    result["passed"] = summary["passed"]
    result["all_passed"] = summary["all_passed"]
    return result


# --- the clock basis in force -----------------------------------------------
#
# Read through the harness's own module so there is exactly one definition of
# what `SOLEXBENCH_CLOCK_BASIS` means. Imported lazily because that module is
# under `src/sol_execbench`, which the dry-run path must be able to skip.

def _clock_basis() -> str:
    from sol_execbench.core.bench.clock_bracket import clock_basis
    return clock_basis()


def _bracketing_enabled() -> bool:
    from sol_execbench.core.bench.clock_bracket import bracketing_enabled
    return bracketing_enabled()


def _bracket_threshold() -> float:
    from sol_execbench.core.bench.clock_bracket import bracket_threshold
    return bracket_threshold()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gpu", type=int, default=AUTHORITATIVE_GPU,
                    help="the card this process times on. With --shard I/N it "
                         "must be I: the partition IS the card assignment")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="score only the problems whose authoritative T_b was "
                         "measured on card I of an N-way pass, following "
                         "authoritative_tb.py's own partition. Coverage is a "
                         "property of the merged output, not of one shard")
    ap.add_argument("--allow-gpu-shard-mismatch", action="store_true",
                    help="permit --shard I/N with --gpu J where I != J. The "
                         "per-problem card check still applies and will refuse "
                         "anything that does not line up")
    ap.add_argument("--part", choices=list(KNOWN_PARTS), default=None,
                    help="which part's artifact tree to read; default is the "
                         "part this node reports. Refused if they disagree")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--benchmark-dir",
                    default=str(ROOT / "data" / "SOL-ExecBench" / "benchmark"),
                    type=Path)
    # The bound paths default to None and are resolved through `ArtifactTree`
    # once the part is known. Spelling `artifacts/03/t_sol.json` here made the
    # MI350X release tree the default on every node, including this one.
    ap.add_argument("--t-sol", default=None, type=Path,
                    help="default: artifacts/03[-<part>]/t_sol.json")
    ap.add_argument("--t-b", default=None, type=Path,
                    help="default: artifacts/06[-<part>]/t_b.json")
    ap.add_argument("--tb-artifacts", default=None, type=Path,
                    help="the authoritative T_b pass output, whose per-problem "
                         "card_identity blocks are what the card check compares "
                         "against. default: artifacts/06[-<part>]/authoritative")
    ap.add_argument("--manifest",
                    help="frozen scoring manifest; supersedes --t-sol/--t-b. "
                         "Preferred, because a score is only meaningful inside "
                         "one manifest version")
    ap.add_argument("--workloads-root", default=None, type=Path,
                    help="AMD-derived tolerances; the authoritative workload "
                         "source. 'none' falls back to the dataset's B200 ones. "
                         "default: artifacts/05[-<part>]/workloads")
    ap.add_argument("--force", action="store_true",
                    help="re-score sessions that already have a score record")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the part, the partition and every problem's "
                         "anchor card, print them, and exit without timing "
                         "anything. Verify a shard here before spending GPU "
                         "hours on it")
    args = ap.parse_args()

    shard = None
    if args.shard is not None:
        try:
            shard = parse_shard(args.shard)
        except ValueError as exc:
            ap.error(str(exc))
        if args.gpu != shard[0] and not args.allow_gpu_shard_mismatch:
            ap.error(
                f"--shard {shard[0]}/{shard[1]} with --gpu {args.gpu}: the "
                f"shard index and the card disagree. The partition is the card "
                f"assignment -- shard i holds the problems whose T_b was "
                f"measured on card i -- so scoring shard i on card j re-times "
                f"every T_k on a card that does not hold its own anchor. Pass "
                f"--allow-gpu-shard-mismatch if the remap is deliberate; the "
                f"per-problem card check still refuses anything that does not "
                f"line up.")

    run_root = ROOT / "artifacts" / "10" / "runs" / args.run_id
    if not run_root.exists():
        sys.exit(f"no such run: {run_root}")
    out_dir = ROOT / "artifacts" / "10" / "scores" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    cfg_path = run_root / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text())

    prov = stamp("10-scoring")["_provenance"]
    # `provenance.stamp()` in this tree does NOT emit a `part` key, so the
    # existing `prov.get("part")` was always None and every part check that
    # depended on it -- including `load_bounds`' -- was dead code. Resolved the
    # way `verify_artifacts` resolves it instead: from the torch device names
    # already in the provenance block. No extra hardware call, and it fails
    # closed (None => "unattributable", never "MI350X").
    detected = artifact_part({"_provenance": prov})
    if args.part and detected and args.part != detected:
        sys.exit(
            f"--part {args.part} but this node reports {detected}. Scoring the "
            f"wrong part's tree pairs a T_SOL derived at another part's clock "
            f"and peak FLOPS with a T_k measured here; every score is then "
            f"rescaled by a ratio nothing records (prime directive 2).")
    part = args.part or detected
    tree = ArtifactTree(part or DEFAULT_PART)
    if part is None:
        print(f"  WARNING: this node did not identify its part, so artifact "
              f"paths fall back to the default tree ({DEFAULT_PART}). Every "
              f"bound artifact's own `part` field is still checked below.",
              file=sys.stderr)

    t_sol_path = args.t_sol or tree.path("03", "t_sol.json")
    t_b_path = args.t_b or tree.path("06", "t_b.json")
    tb_artifacts = args.tb_artifacts or tree.dir("06") / "authoritative"
    if args.workloads_root is None:
        workloads_root = tree.dir("05") / "workloads"
    elif str(args.workloads_root).lower() == "none":
        workloads_root = None
    else:
        workloads_root = args.workloads_root
    print(f"  part         {part or 'UNIDENTIFIED'}  (tree {tree.dir('03').name}, "
          f"{tree.dir('06').name})")

    integrity = compare_digests(config.get("harness_tree_digest"), tree_digest(ROOT))
    if integrity.get("comparable") and not integrity.get("match"):
        print("WARNING: the scoring-critical source tree changed since the sweep "
              "started.", file=sys.stderr)
        for p in integrity.get("changed", []):
            print(f"  changed: {p}", file=sys.stderr)
        print("  This is recorded on the summary. It does not prove tampering -- an "
              "operator edit looks the same -- but a score whose harness moved "
              "mid-run is not comparable with one whose did not.", file=sys.stderr)

    manifest_meta: dict = {}
    if args.manifest and Path(args.manifest).exists():
        manifest_doc = json.loads(Path(args.manifest).read_text())
        # The manifest was NOT part-checked before. It is the preferred bound
        # source and it is passed by hand, which is precisely why a wrong one
        # would not be noticed: `manifest-v1.json` is MI350X and resolves
        # perfectly well from an MI355X node.
        m_part = manifest_doc.get("part") or artifact_part(manifest_doc)
        if part and m_part and m_part != part:
            sys.exit(
                f"REFUSING: manifest {args.manifest} was cut on {m_part} but "
                f"this node is {part}. Its T_SOL and T_b were derived at "
                f"another part's clock and peak FLOPS.")
        t_sol, t_b, manifest_meta = load_manifest_bounds(Path(args.manifest))
        manifest_meta["part"] = m_part
        t_sol_meta = t_b_meta = {"from_manifest": manifest_meta.get("manifest_version")}
        print(f"  bounds from manifest {manifest_meta.get('manifest_version')}: "
              f"T_SOL for {manifest_meta['problems_with_t_sol']} problems, "
              f"T_b for {manifest_meta['problems_with_t_b']}", file=sys.stderr)
    else:
        t_sol, t_sol_meta = load_bounds(t_sol_path, part)
        t_b, t_b_meta = load_bounds(t_b_path, part)
    for label, meta in (("T_SOL", t_sol_meta), ("T_b", t_b_meta)):
        if meta.get("rejected"):
            # Not a downgrade to a weaker basis: a foreign-part bound that got
            # this far was either the resolved default or asked for by name, and
            # silently scoring everything as `correctness_only` instead would
            # look like a run that simply had no bounds.
            sys.exit(f"REFUSING: {label} {meta['rejected']}")
        elif not meta:
            print(f"  {label}: absent — scores fall back to a weaker basis",
                  file=sys.stderr)

    lock_basis = _clock_basis()
    lock = clock_lock_state(ROOT, args.gpu)
    if lock_basis == "locked":
        # Probed before anything is timed, and the harness's env flag is set only
        # if the probe agrees. Refusing here costs a re-run; not refusing produces
        # a scoreboard of numbers taken at an unknown clock, which is unrecoverable.
        if not lock.get("locked"):
            sys.exit(
                f"GPU {args.gpu} is not clock-locked: {lock}\n"
                f"Every latency here is quoted at F_LOCK. Lock it first:\n"
                f"  python scripts/clock_calibrate.py lock --freq-mhz <F_LOCK> --all-gpus\n"
                f"  python scripts/clock_calibrate.py verify --freq-mhz <F_LOCK> --under-load"
            )
        print(f"  clock lock: GPU {args.gpu} at {lock['performance_level']} "
              f"({lock.get('drm_card')})")
    else:
        # Unlocked basis (STATE.md, MI355X clock methodology): there is no
        # F_LOCK on this part to assert, so the guarantee is the per-window
        # bracket. Asserting that it is switched on is the equivalent check --
        # without it, latencies would be taken at an unknown, unrecorded clock,
        # which is the same failure the lock check exists to prevent.
        if not _bracketing_enabled():
            sys.exit(
                f"SOLEXBENCH_CLOCK_BASIS={lock_basis!r} but clock bracketing is "
                f"not enabled, so nothing would record the clock each latency "
                f"was taken at. Refusing.")
        print(f"  clock basis: unlocked, per-window bracket at threshold "
              f"{_bracket_threshold()} (GPU {args.gpu} reads "
              f"{lock.get('performance_level')})")

    sessions = sorted(run_root.glob("*/*/session.json"))
    all_problems = sorted({p.parent.name for p in sessions})
    mine = set(partition_problems(all_problems, shard, tb_artifacts))
    sessions = [p for p in sessions if p.parent.name in mine]

    print(f"{len(sessions)} session(s) over {len(mine)} problem(s) to score on "
          f"GPU {args.gpu} (serial)")
    if shard is not None:
        print(f"shard        {shard[0]}/{shard[1]} of {len(all_problems)} "
              f"problems; coverage is a property of the MERGED {out_dir}, not "
              f"of this shard")
    print(f"anchors      {tb_artifacts}")

    # The card this process is actually on, resolved by the SAME probe that
    # stamped the anchors -- so the comparison is between two identities
    # produced the same way, not between an identity and a torch index.
    # Skipped under --dry-run, which is the whole point of --dry-run: it takes
    # no HIP context on a card that may be running a timing job.
    actual_card = None if args.dry_run else card_identity(str(args.gpu))
    if actual_card and not actual_card.get("identified"):
        print(f"WARNING: card {args.gpu} could not identify itself; every "
              f"problem with a T_b will be refused. "
              f"{str(actual_card.get('error'))[:300]}", file=sys.stderr)
    elif actual_card:
        print(f"card         {actual_card.get('uuid')}  {actual_card.get('bdf')}  "
              f"{actual_card.get('hostname')}")

    def verdict_for(problem_key: str) -> dict:
        anchor, note = anchor_card(tb_artifacts, problem_key)
        return card_verdict(anchor, actual_card, anchor_note=note,
                            t_b_in_scope=bool(t_b.get(problem_key)))

    if args.dry_run:
        # Everything that can be decided without a GPU, decided and printed:
        # the partition, and for each problem whether a T_b is in scope and
        # which card its anchor sits on. An operator verifying a shard needs
        # the whole slice, not a preview.
        n_anchored = 0
        for key in sorted(mine):
            anchor, note = anchor_card(tb_artifacts, key)
            in_scope = bool(t_b.get(key))
            n_anchored += bool(anchor)
            where = (f"{anchor['hostname']} {anchor['bdf']}" if anchor
                     else f"no card ({note})")
            print(f"  {key}: t_b={'yes' if in_scope else 'no '}  {where}")
        print(f"\n{len(mine)} problem(s) in this shard, {n_anchored} with an "
              f"identified anchor card, "
              f"{sum(1 for k in mine if t_b.get(k))} with a T_b in scope")
        print("dry run: no card probed, nothing timed, nothing written")
        return 0

    import os
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if lock_basis == "locked":
        os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = "1"
    else:
        # Never declared on the unlocked basis. The flag means "clocks are
        # locked"; exporting it here would make every latency read as taken at
        # an F_LOCK that does not exist on this part.
        os.environ.pop("SOL_EXECBENCH_CLOCKS_LOCKED", None)

    results = []
    # Counted where the verdict is taken, not reconstructed from the results
    # afterwards: a census assembled from whatever the records happen to carry
    # is a census that can miss the thing it exists to count.
    card_states: dict[str, int] = {}
    refusals: list[dict] = []
    for n, sess_path in enumerate(sessions, 1):
        session_dir = sess_path.parent
        harness, problem_key = session_dir.parent.name, session_dir.name
        dest = out_dir / harness / f"{problem_key}.json"
        if dest.exists() and not args.force:
            prior = json.loads(dest.read_text())
            results.append(prior)
            state = ((prior.get("card_check") or {}).get("state")
                     or "resumed_unrecorded")
            card_states[state] = card_states.get(state, 0) + 1
            print(f"  [{n}/{len(sessions)}] {harness}/{problem_key}: already scored")
            continue

        check = verdict_for(problem_key)
        card_states[check["state"]] = card_states.get(check["state"], 0) + 1
        if not check["ok"]:
            # Refused, recorded, and counted -- not skipped. A skip is
            # indistinguishable from a problem nobody got to; a refusal names
            # itself and shows up in the outcome census.
            result = {"problem": problem_key,
                      "category": problem_key.split("__", 1)[0],
                      "harness": harness, "gpu": args.gpu,
                      "outcome": "refused_card_mismatch",
                      "note": check["reason"], "card_check": check,
                      "records": []}
            refusals.append({"harness": harness, "problem": problem_key,
                             "state": check["state"], "reason": check["reason"]})
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({**stamp("10-scoring"), **result},
                                       indent=2, default=str))
            results.append(result)
            print(f"  [{n}/{len(sessions)}] {harness}/{problem_key}: REFUSED "
                  f"({check['state']}) {check['reason'][:160]}")
            continue

        category, name = problem_key.split("__", 1)
        problem_dir = args.benchmark_dir / category / name
        try:
            result = score_one(session_dir, problem_dir, workloads_root,
                               t_sol=t_sol, t_b=t_b, part=part,
                               timeout=args.timeout, gpu=args.gpu,
                               lock_clocks=(lock_basis == "locked"),
                               card_check=check)
        except Exception as exc:  # noqa: BLE001
            result = {"problem": problem_key, "harness": harness,
                      "outcome": "scorer_error", "card_check": check,
                      "note": f"{type(exc).__name__}: {exc}", "records": []}

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({**stamp("10-scoring"), **result},
                                   indent=2, default=str))
        results.append(result)

        detail = result.get("outcome")
        if detail == "evaluated":
            detail = f"{result['passed']}/{result['workloads']} workloads passed"
            if result["reference_copy"]["kind"] != "distinct":
                detail += f" (reference {result['reference_copy']['kind']} copy)"
        print(f"  [{n}/{len(sessions)}] {harness}/{problem_key}: {detail}")

    summary = {
        "run_id": args.run_id,
        "part": part,
        "part_source": "--part" if args.part else "detected",
        "authoritative_gpu": args.gpu,
        "shard": ({"index": shard[0], "count": shard[1], "gpu": args.gpu,
                   "_note": "COVERAGE IS A PROPERTY OF THE MERGED SCORES "
                            "DIRECTORY, NOT OF ONE SHARD."}
                  if shard else None),
        "scoring_card": actual_card,
        "clock_basis": lock_basis,
        "clock_lock": lock,
        "f_lock_mhz": prov.get("f_lock_mhz"),
        "sessions_scored": len(results),
        "manifest": manifest_meta or None,
        "t_sol": {"path": str(t_sol_path), "used": bool(t_sol), **t_sol_meta},
        "t_b": {"path": str(t_b_path), "used": bool(t_b), **t_b_meta},
        "tb_artifacts": str(tb_artifacts),
        # "0 refused" is only meaningful next to how many were checked, so both
        # are here. `not_applicable` is a third number on purpose: it is the
        # count of records published with no cross-card claim to make, and
        # reading it as "passed the check" would be wrong.
        "card_enforcement": {
            "matched": card_states.get("matched", 0),
            "refused": (card_states.get("mismatch", 0)
                        + card_states.get("unverifiable", 0)),
            "not_applicable": card_states.get("not_applicable", 0),
            "by_state": dict(sorted(card_states.items())),
            "refusals": refusals,
        },
        "clock_bracket_refused_records": sum(
            1 for res in results for r in res.get("records", [])
            if r.get("clock_bracket_refused")
        ),
        "harness_integrity": integrity,
        "workloads_root": str(workloads_root) if workloads_root else None,
        "spec_tampered": [
            {"harness": r.get("harness"), "problem": r.get("problem"),
             "divergence": r.get("packet_spec_divergence")}
            for r in results if r.get("spec_tampered")
        ],
        "outcomes": _count(results, "outcome"),
        "score_bases": _count(
            [r for res in results for r in res.get("records", [])], "score_basis"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps({**stamp("10-scoring"), **summary}, indent=2, default=str)
    )
    print(f"\nwrote {out_dir}/summary.json")
    print(json.dumps(summary["outcomes"], indent=2))
    ce = summary["card_enforcement"]
    print(f"card check: {ce['matched']} matched, {ce['refused']} REFUSED, "
          f"{ce['not_applicable']} not applicable (no T_b in scope)")
    print(f"next: python scripts/build_scoreboard.py --run-id {args.run_id}")
    return 0


def _count(items: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item.get(key))] = counts.get(str(item.get(key)), 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    sys.exit(main())
