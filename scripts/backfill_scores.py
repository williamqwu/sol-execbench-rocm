#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 10 — recompute scores from a newer manifest without re-running any GPU work.

    python scripts/backfill_scores.py --run-id pilot-01 \
        --manifest artifacts/09/manifest-MI355X-v1.json

The two-phase plan this repo is following measures agents *before* `T_b` exists,
under a weaker `score_basis`, and layers the real SOL score in when the anchor
lands. That layering must not re-time anything: `T_k`, `T_ref` and the pass/fail
verdicts were measured on the authoritative GPU at F_LOCK and are still valid.
Only the bounds changed.

Re-running the evaluation instead would cost hours of GPU time to produce the
same latencies, and — worse — the new latencies would be measured on a different
day under different node conditions, so a record's basis and its timing would
come from different runs. Recomputing arithmetic is both cheaper and more
correct.

Every record keeps its previous basis under ``score_basis_history``, so a
strengthened score is visibly a strengthened score rather than a silently
different number.


TWO PROVENANCE BLOCKS, BECAUSE THERE ARE TWO EVENTS
---------------------------------------------------
A backfilled score file is the product of two runs, and prime directive 5 is
owed an answer for each: the **measurement** (which host, which cards, which
ROCm, which F_LOCK produced ``T_k``) and the **rebase** (which manifest, which
anchor tree, which git SHA recomputed ``S``). One key cannot hold both.

So the original ``_provenance`` is left exactly where the scorer wrote it and a
second block, ``_backfill_provenance``, is added beside it. Purely additive:
every existing reader of ``_provenance`` -- ``_assert_comparable`` below,
``verify_artifacts``'s part and clock resolution, ``leaderboard/ingest.py``'s
``created_utc``/``provenance_json``/``run_part`` -- goes on reading the
measurement's block and sees the same value it saw before.

This file used to write ``json.dumps({**stamp('10-backfill'), **doc})``. ``doc``
already carries the measurement's ``_provenance`` and is spread SECOND, so the
freshly computed stamp was overwritten by the old one and discarded --
silently, on every file, since the line was written. 417 score files were
rebased against manifest v4 on 2026-08-15 and every one of them attested to a
run from the previous day, with only ``backfilled_from_manifest`` to hint
otherwise. That is worse than an unstamped artifact, because an unstamped one
announces itself. ``summary.json`` was worse again: its write path never called
``stamp()`` at all.

Letting the fresh stamp win instead would have been the one-line fix and is the
wrong one: it would erase the host and cards ``T_k`` was measured on, which is
the provenance that cannot be recovered by re-running anything.


THE CARD THE ANCHOR CAME OFF IS PART OF WHETHER THE RAISE IS LEGAL
-------------------------------------------------------------------
Recomputing arithmetic is cheap, but it is only *correct* while the ``T_b`` being
pulled in is comparable to the ``T_k`` already on disk. ``STATE.md`` §4.4 makes
that a per-problem property: T_b and T_k must have been measured on the same
card, because the authoritative pass now runs 8-way with per-problem card
pinning and the cards do not share a clock.

So the raise is guarded the same way ``score_solutions.py`` guards the original
scoring: for every record that would gain a ``T_b``, the ``card_identity`` on
that problem's authoritative artifact is compared against the card the score
file records having been measured on. A mismatch, or an identity that cannot be
established on either side, **refuses the raise and counts it** -- the record
keeps its weaker basis and says why, which is a recoverable state. A raise that
pairs two cards is not recoverable, because nothing downstream can tell.

The pre-§4.4 regime is untouched: a manifest that declares one
``authoritative_gpu`` at one ``f_lock_mhz`` was produced under the single-card
rule, ``_assert_comparable`` is the proof for it, and MI350X backfills take
exactly the path they took before.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402
from solexbench_agents.scoring import (  # noqa: E402
    headroom_fraction,
    resolve_basis,
    sol_score,
)

from score_solutions import (  # noqa: E402
    _interval_score,
    anchor_card,
    card_verdict,
    load_manifest_bounds,
    workload_bound,
    workload_record,
)
from verify_artifacts import (  # noqa: E402
    DEFAULT_PART,
    KNOWN_PARTS,
    ArtifactTree,
    artifact_part,
)


#: Where the rebase records itself. Named once, so the writer here and any
#: reader that wants to know when a score was last recomputed spell it the same
#: way, and so it is greppable next to ``_provenance``.
BACKFILL_PROVENANCE_KEY = "_backfill_provenance"


def backfill_provenance(*, manifest: Path, manifest_version: str | None,
                        part: str | None, tb_trees: list[Path],
                        run_id: str) -> dict:
    """The block that says who rebased these scores, and against what.

    Computed ONCE per invocation rather than per file: it is one event, so
    every file it touches should carry the same timestamp, and ``stamp()``
    enumerates torch devices on each call -- 417 of those is a minute of
    nothing.

    ``part`` is *declared*, with ``allow_cross_part``, because the backfill is
    arithmetic: it re-derives ``S`` from a manifest and an anchor tree and never
    touches a GPU, so the part is a property of the artifacts being rebased and
    not of the host doing the rebasing. Refusing to run an MI355X backfill on an
    MI350X node would be refusing a legitimate operation, and stamping it with
    the host's part would be a false statement that outranks its own evidence
    (see ``provenance.stamp``). ``part_detected`` records the host regardless.
    """
    return stamp(
        "10-backfill",
        {
            "run_id": run_id,
            "manifest": str(manifest),
            "manifest_version": manifest_version,
            "tb_artifacts": [str(t) for t in tb_trees],
            "rebase_only": True,
            "note": (
                "This artifact has two provenance blocks. `_provenance` is the "
                "MEASUREMENT: the host, cards, ROCm and clock that produced "
                "`t_k_ms`, which no rebase may overwrite. This block is the "
                "REBASE: which manifest and anchor tree recomputed the bounds "
                "and `sol_score` from those unchanged timings. No timing was "
                "re-measured."
            ),
        },
        part=part,
        allow_cross_part=True,
    )["_provenance"]


def backfill_record(record: dict, problem_key: str, t_sol: dict,
                    t_b: dict, *, card_check: dict | None = None
                    ) -> tuple[dict, bool]:
    """Return (record, changed). Timings are never touched, only the bounds.

    ``card_check`` is the verdict from :func:`score_solutions.card_verdict` for
    this problem. When it refuses, the incoming ``T_b`` is dropped -- not
    applied with a warning -- so the record keeps whatever basis it already had
    and gains a ``t_b_refused`` field saying why. ``None`` means the caller has
    established that the single-card regime applies and there is nothing
    per-problem to check.
    """
    uuid = record.get("workload_uuid")
    new_sol = workload_bound(t_sol, problem_key, uuid, "t_sol_ms")
    new_cycles = workload_bound(t_sol, problem_key, uuid, "t_sol_cycles")
    new_b = workload_bound(t_b, problem_key, uuid, "t_b_ms")

    refused = None
    if new_b is not None and card_check is not None and not card_check["ok"]:
        refused = card_check["reason"]
        new_b = None

    old_basis = record.get("score_basis")
    t_k = record.get("t_k_ms")
    t_ref = record.get("t_ref_ms")
    correct = bool(record.get("correct"))

    # The bound this T_k is judged against is T_SOL over T_K'S OWN clock window,
    # not the reference-clock column the manifest carries. `score_solutions.py`
    # has done this since the unlocked basis landed; this script did not, and
    # took `t_sol_ms` straight out of the manifest. On MI355X manifest-v4 that
    # column differs from the published bound by more than 1% on 1622 of 3717
    # scoreable workloads and by more than 30% on 1084 -- the 2.4/1.8 ratio of
    # D63 -- in BOTH directions, so it is not a conservative simplification.
    #
    # `_interval_score` is imported from the scorer rather than reimplemented:
    # two scorers that can disagree about the same bound is the failure mode this
    # whole file exists downstream of, and a divergence here would show up as a
    # score that changes when a run is backfilled rather than re-scored.
    #
    # `{}` comes back when no interval can be formed -- the locked basis, a
    # record with no bracket, a bound predating the term split -- and then the
    # reference-clock column is what there is, exactly as before. Every MI350X
    # record takes that path: `manifest-v1` and `-v1.2` carry
    # `t_sol_ms_published` on 0 of their 3717 scoreable workloads.
    #
    # NOT SO ON MI355X, AND THIS IS AN OPEN GAP. 137 of the 2185 records under
    # `artifacts/10/scores` carry no usable clock bracket, so they take the
    # fallback -- and there the manifest DOES publish a bound. The gate
    # disagrees with this line about them: `verify_artifacts._bound_for` ends
    # `return (w.get("t_sol_ms_published") or w["t_sol_ms"]), False`, so check D
    # judges those 137 against the published bound while the score stored beside
    # them is against the legacy reference-clock column, which on manifest-v4
    # differs from it on 3685 of 3717 workloads.
    #
    # Not closed here, deliberately. Closing it changes which bound a PUBLISHED
    # score is computed against, which is a methodology change and not an
    # auditor's edit (CLAUDE.md prime directive 7), and it cannot be done in
    # this file alone: `load_manifest_bounds` carries only `RECLOCK_FIELDS` into
    # the bounds map, and `t_sol_ms_published` / `t_sol_cycles_published` are not
    # in it. The one-line remedy is to add them there -- `scripts/build_manifest.py`
    # and `scripts/score_solutions.py`, whose owners decide the shape.
    interval = _interval_score(workload_record(t_sol, problem_key, uuid),
                               record.get("clock_bracket"), t_k, new_b)
    if interval:
        new_sol = interval["t_sol_ms_published"]
        new_cycles = interval["t_sol_cycles_published"]

    basis = resolve_basis(correct=correct, t_k_ms=t_k, t_ref_ms=t_ref,
                          t_sol_ms=new_sol, t_b_ms=new_b)
    s = sol_score(t_k, new_sol, new_b) if correct else None
    hf = headroom_fraction(t_k, t_ref, new_sol) if correct else None

    changed = (
        new_sol != record.get("t_sol_ms")
        or new_b != record.get("t_b_ms")
        or basis.value != old_basis
        or refused != record.get("t_b_refused")
    )
    if not changed:
        return record, False

    history = list(record.get("score_basis_history") or [])
    if old_basis and old_basis != basis.value:
        history.append(old_basis)

    record.update(
        t_sol_ms=new_sol,
        t_sol_cycles=new_cycles,
        # The interval the published bound was taken from, so the record says
        # which clock its own bound is at rather than needing the manifest and
        # the bracket to be re-read together (D63).
        **{k: v for k, v in interval.items() if not k.startswith("sol_score_at_")
           or correct},
        t_b_ms=new_b,
        t_b_variant=workload_bound(t_b, problem_key, uuid, "t_b_variant"),
        sol_score=s,
        headroom_fraction=hf,
        score_basis=basis.value,
        score_basis_history=history,
    )
    if refused:
        record["t_b_refused"] = refused
    else:
        record.pop("t_b_refused", None)
    # Re-derived rather than carried over: a bound that just changed may now be
    # the one being violated, or may have stopped being violated.
    record.pop("bound_violation", None)
    if correct and t_k and new_sol and t_k < new_sol:
        record["bound_violation"] = (
            f"t_k {t_k:.6g} ms is below t_sol {new_sol:.6g} ms"
        )
    return record, True


def _assert_comparable(score_files: list[Path], manifest: Path) -> None:
    """Refuse to pair a T_b with a T_k that is not comparable to it.

    The docstring above claims T_k "was measured on the authoritative GPU at F_LOCK
    and is still valid, only the bounds changed". S divides one by the other, so that
    claim is load-bearing, and it is only true while both sides come from the same
    card at the same clock. Nothing checked it.

    It has since stopped being true here. The pilot's T_k was measured on **GPU 0 at
    F_LOCK 1640**, before D30 established that GPU 0 does not hold a setpoint under
    load -- it runs 1657 MHz alone and 1410 with the node busy. The T_b in
    manifest-MI355X-v1 was measured on **GPU 1 at 1650**. Backfilling one into the
    other silently rescales every score by the ratio between two cards' real clocks,
    and the output looks entirely plausible: this is D26 again, one layer further in.

    So the guard `collect_t_b()` applies when *gathering* T_b is applied here when
    *pairing* it. Refuses rather than warns, because the failure is invisible
    downstream.
    """
    m = json.loads(manifest.read_text())
    m_prov = m.get("_provenance") or {}
    m_lock = m_prov.get("f_lock_mhz")
    m_gpu = m_prov.get("authoritative_gpu")

    seen: dict[tuple, int] = {}
    for p in score_files:
        prov = (json.loads(p.read_text()).get("_provenance") or {})
        seen[(prov.get("f_lock_mhz"), prov.get("authoritative_gpu"))] = \
            seen.get((prov.get("f_lock_mhz"), prov.get("authoritative_gpu")), 0) + 1

    bad = {k: n for k, n in seen.items()
           if (k[0] is not None and m_lock is not None and k[0] != m_lock)
           or (k[1] is not None and m_gpu is not None and k[1] != m_gpu)}
    if bad:
        lines = "\n".join(
            f"    {n} record file(s) measured at F_LOCK {lk} on GPU {g}"
            for (lk, g), n in sorted(bad.items(), key=lambda kv: -kv[1]))
        sys.exit(
            f"REFUSING to backfill: these scores were not measured under the "
            f"conditions this manifest's T_b was.\n"
            f"  manifest T_b : F_LOCK {m_lock} MHz on GPU {m_gpu}\n"
            f"{lines}\n"
            f"  S = f(T_k, T_b, T_SOL) divides one timing by another, so pairing "
            f"them rescales every score by the ratio between two different clocks "
            f"-- silently, and the result looks plausible (STATE.md D26).\n"
            f"  Re-run scoring for this run against the current node, or backfill "
            f"from a manifest measured under the same conditions.")


def _same_card(anchor: dict, actual: dict) -> bool:
    """Is this anchor off the same physical card as this T_k?

    uuid first because it is the only identifier that survives a PCI
    renumbering; BDF plus hostname as the fallback for artifacts written before
    the uuid was recorded. A missing identifier is never a match -- an unknown
    card is not the same card.
    """
    au, bu = anchor.get("uuid"), (actual or {}).get("uuid")
    if au and bu:
        return au == bu
    ab, bb = anchor.get("bdf"), (actual or {}).get("bdf")
    ah, bh = anchor.get("hostname"), (actual or {}).get("hostname")
    return bool(ab and bb and ab == bb and ah and bh and ah == bh)


def anchor_trees(explicit, manifest_doc: dict, tree) -> list[Path]:
    """Which T_b tree(s) the card check interrogates.

    **The default is the tree the manifest was built from.** The check exists to
    prove that the T_b being pulled in came off the same card as the T_k already
    on disk, so it has to interrogate the T_b that is actually IN the manifest,
    not a same-named tree beside it. A problem is commonly anchored on several
    cards across the three nodes, and the merge publishes one of them; checking
    `artifacts/06-MI355X/authoritative` against a manifest built from
    `authoritative-merged` therefore compares one measurement's card against a
    *different measurement of the same problem*. On this run that is the
    difference between 126 and 594 records keeping `sol_score_v1`, and neither
    number is about the bounds.

    Task 06's gate had the mirror image of this -- it audited a tree nobody
    published and reported coverage against a manifest built from another -- which
    is why `build_manifest.py` records `sources` at all. This is the first
    consumer to use it.

    Falls back to `artifacts/06[-<part>]/authoritative` for a manifest with no
    `sources` block: both frozen MI350X manifests, whose behaviour is therefore
    unchanged.
    """
    if explicit:
        return list(explicit)
    stated = (manifest_doc.get("sources") or {}).get("t_b")
    if stated:
        p = Path(stated)
        return [p if p.is_absolute() else ROOT / p]
    return [tree.dir("06") / "authoritative"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--part", choices=list(KNOWN_PARTS), default=None,
                    help="which part's artifact tree the anchors live in; "
                         "default is the part the manifest declares")
    ap.add_argument("--tb-artifacts", default=None, type=Path, action="append",
                    help="the authoritative T_b pass output, whose per-problem "
                         "card_identity blocks the raise is checked against. "
                         "REPEATABLE: the authoritative pass runs on several "
                         "nodes and each writes its own tree, so pass every "
                         "tree and the anchor is taken from whichever one holds "
                         "the card this record's T_k was measured on. "
                         "default: the tree the manifest itself was built from "
                         "(`sources.t_b`), else artifacts/06[-<part>]/"
                         "authoritative")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scores_dir = ROOT / "artifacts" / "10" / "scores" / args.run_id
    if not scores_dir.is_dir():
        sys.exit(f"no scores for run {args.run_id!r} at {scores_dir}")

    t_sol, t_b, meta = load_manifest_bounds(args.manifest)
    if not meta:
        sys.exit(f"manifest not readable: {args.manifest}")
    print(f"manifest {meta['manifest_version']}: T_SOL for "
          f"{meta['problems_with_t_sol']} problems, T_b for "
          f"{meta['problems_with_t_b']}")

    manifest_doc = json.loads(args.manifest.read_text())
    m_prov = manifest_doc.get("_provenance") or {}
    part = (args.part or manifest_doc.get("part") or artifact_part(manifest_doc)
            or DEFAULT_PART)
    tree = ArtifactTree(part)
    tb_trees = anchor_trees(args.tb_artifacts, manifest_doc, tree)
    tb_artifacts = tb_trees[0]

    # The pre-4.4 single-card regime, recognised by what the manifest itself
    # declares rather than by the part name: one authoritative GPU at one
    # F_LOCK is the proof that every T_b in it came off one card, and
    # `_assert_comparable` already checks the T_k side against it. Anything
    # else is checked per problem, against the card the anchor records.
    single_card = (m_prov.get("authoritative_gpu") is not None
                   and m_prov.get("f_lock_mhz") is not None)
    print(f"part {part}; anchors {', '.join(str(t) for t in tb_trees)}; "
          + ("single-card manifest: the per-problem card check does not apply"
             if single_card else
             "per-problem card check ON (STATE.md 4.4)"))

    files = sorted(p for p in scores_dir.glob("*/*.json") if p.name != "summary.json")
    _assert_comparable(files, args.manifest)
    prov = backfill_provenance(manifest=args.manifest,
                               manifest_version=meta["manifest_version"],
                               part=part, tb_trees=tb_trees,
                               run_id=args.run_id)
    changed_files = 0
    changed_records = 0
    basis_counts: dict[str, int] = {}
    violations = 0
    card_refusals: dict[str, str] = {}
    card_matched: set[str] = set()

    for path in files:
        doc = json.loads(path.read_text())
        problem_key = doc.get("problem")
        check = None
        if not single_card:
            # The card the T_k on disk was measured on, as `score_solutions`
            # recorded it. Absent on a score file written before that field
            # existed -- which is itself a refusal, not a pass: an unrecorded
            # card is an unknown card.
            actual = ((doc.get("card_check") or {}).get("actual")
                      or doc.get("scoring_card"))
            # Search the trees for the one holding THIS record's card. §4.4
            # requires T_b and T_k to share a card, and the authoritative pass
            # is 8-way card-pinned across three nodes, so a problem is commonly
            # anchored on several cards and only one of them is the right
            # partner for this T_k. Handing the merged tree to this check
            # refused 178 of 220 problems -- not because the anchors were
            # missing, but because the merge had legitimately picked another
            # node's replicate, which is exactly what merge_authoritative_tb's
            # docstring says the merged tree must NOT be used for. Falls back to
            # the first tree so the refusal still names a concrete path.
            anchor, note = anchor_card(tb_artifacts, problem_key)
            if actual:
                for cand in tb_trees:
                    c_anchor, c_note = anchor_card(cand, problem_key)
                    if c_anchor and _same_card(c_anchor, actual):
                        anchor, note = c_anchor, c_note
                        break
            check = card_verdict(anchor, actual, anchor_note=note,
                                 t_b_in_scope=bool(t_b.get(problem_key)))
            if check["ok"] and check["state"] == "matched":
                card_matched.add(problem_key)
            elif not check["ok"]:
                card_refusals[problem_key] = check["reason"]
        touched = False
        for record in doc.get("records", []):
            record, changed = backfill_record(record, problem_key, t_sol, t_b,
                                              card_check=check)
            if changed:
                changed_records += 1
                touched = True
            key = record.get("score_basis") or "none"
            basis_counts[key] = basis_counts.get(key, 0) + 1
            if record.get("bound_violation"):
                violations += 1
        if touched or (check and not check["ok"]):
            changed_files += 1
            doc["backfilled_from_manifest"] = meta["manifest_version"]
            if check is not None:
                doc["backfill_card_check"] = check
            # Beside the measurement's block, never over it. See the module
            # docstring: `{**stamp(...), **doc}` put the fresh stamp FIRST and
            # `doc`'s own `_provenance` second, so the stamp was discarded by
            # the merge on every file this script had ever written.
            doc[BACKFILL_PROVENANCE_KEY] = prov
            if not args.dry_run:
                path.write_text(json.dumps(doc, indent=2, default=str))

    # The run summary carries the basis census, and `verify_artifacts --task 10`
    # reads it. Leaving it stale after a rebase would make the acceptance check
    # describe a distribution that no longer exists on disk -- the same class of
    # drift as a document and a constant disagreeing about F_LOCK (F17).
    summary_path = scores_dir / "summary.json"
    if summary_path.exists() and not args.dry_run:
        summary = json.loads(summary_path.read_text())
        summary["score_bases"] = dict(sorted(basis_counts.items()))
        summary["backfilled_from_manifest"] = meta["manifest_version"]
        summary["bound_violations"] = violations
        summary["backfill_card_enforcement"] = {
            "mode": "single-card-manifest" if single_card else "per-problem",
            "tb_artifacts": str(tb_artifacts),
            "matched": len(card_matched),
            "refused": len(card_refusals),
            "refusals": card_refusals,
        }
        # The summary is rewritten here too -- its basis census, its violation
        # count and its card enforcement block are all recomputed -- and until
        # now this path called `stamp()` NOWHERE, so a summary rebased against a
        # new manifest kept the original scoring run's timestamp and SHA with
        # nothing at all to say it had moved. Same key as the per-problem files
        # so one grep answers "when was this run last rebased".
        summary[BACKFILL_PROVENANCE_KEY] = prov
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"{changed_files} of {len(files)} score files updated, "
          f"{changed_records} records rebased"
          + ("  (dry run, nothing written)" if args.dry_run else ""))
    print("score bases now:")
    for basis, n in sorted(basis_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {basis:<24} {n}")
    if card_refusals:
        print(f"\n{len(card_refusals)} problem(s) REFUSED the T_b raise on the "
              f"card check; they keep their previous basis and carry "
              f"`t_b_refused`. Re-score them on the card holding their anchor.")
        for key, why in sorted(card_refusals.items())[:10]:
            print(f"  {key}: {why[:160]}")
        if len(card_refusals) > 10:
            print(f"  ... and {len(card_refusals) - 10} more")
    if violations:
        print(f"\n{violations} record(s) remain faster than their T_SOL bound. "
              f"Those bounds are wrong, not those kernels; S must not be "
              f"published for them (STATE.md D25).")
    print(f"\nnext: python scripts/build_scoreboard.py --run-id {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
