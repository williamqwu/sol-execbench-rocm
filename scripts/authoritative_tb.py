#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 06, second pass — re-time the selected T_b variants, card-pinned.

The selection sweep shards across all eight GPUs, which is the only way it
finishes in an evening. But the eight GPUs do not share a clock: on MI350X, at
the same determinism setting, they land between 1242 and 1307 MHz, a 5% spread,
which is wider than most of the differences this benchmark exists to measure. A
T_b assembled from eight GPUs would encode that spread into the score scale, and
nothing downstream could see it.

So selection and measurement are separated:

  selection    GPUs 0-7, whole problems (all variants of one problem on ONE
               GPU, so the comparison that picks a winner is internally
               consistent even though GPUs differ)
  measurement  re-timing just the variants that won

    python scripts/authoritative_tb.py --gpu 0                  # serial, GPU 0
    python scripts/authoritative_tb.py --shard 3/8 --gpu 3      # one of eight

Resumable: a problem whose authoritative artifact exists is skipped.


WHY THE ONE-CARD RULE NO LONGER APPLIES, AND WHY THAT IS NOT A RELAXATION
-------------------------------------------------------------------------
Read this before concluding that someone quietly dropped the GPU-0 rule.

MI350X pinned the whole authoritative pass to GPU 0 for the reason above: with
the eight cards spanning 1242-1307 MHz, a T_b measured on one card was not
comparable to a T_b measured on another, and T_b is the anchor of the score
scale. Every T_b therefore had to come from the same card.

`STATE.md` §4.4 changed what a score is made of: **T_b and T_k are re-timed back
to back on the same card in one session**, and both clock brackets are recorded
on every score. That was decided to fix the two-clock problem (unlocked, T_b was
measured at whatever clock its kernel pulled and T_k at whatever the candidate's
kernel pulled, and nothing normalized them). It has a consequence for this
script, and the consequence makes the requirement **weaker and different**, not
merely weaker:

    before   every T_b must come from ONE card, because T_b values are
             compared against each other through a common score scale
    now      a problem's T_b must share a card with ITS OWN T_k
             — nothing requires problem A's T_b to share a card with
             problem B's T_b

The cross-card clock spread cancels inside each problem's own ratio, which is
the only place it was ever going to be compared. So the pass may run 8-way,
*provided*:

  1. each problem is pinned to a card **deterministically**, as a pure function
     of the problem name — `plan[i::N]` over the sorted plan — so that every
     replicate of a problem lands on the same card, with no lock, no shared
     filesystem and no scheduler in the loop; and
  2. the card identity is **recorded on the artifact** and therefore
     enforceable downstream, so the later re-timing of T_k can be put on the
     same card, and so a mismatch is a detectable error rather than an
     invisible one.

Both are implemented below. This is a consequence of decision §4.4, not an
independent methodology change (prime directive 7), and it is what turns an
11.4-hour serial pass into roughly 1.5 hours.

What did NOT change: **one job per card**. The parallelism is across cards,
never within one. Two timing runs on one card inflate each other and the
artifact cannot tell.


FLAGS
-----
`--gpu G` alone behaves exactly as it did on MI350X: the whole plan, serially,
with `HIP_VISIBLE_DEVICES=G` set for each child by this script. The MI350X
authoritative pass is frozen and must stay reproducible, so no shard flag means
no change to which problem runs where, in what order.

`--shard I/N` takes `plan[I::N]` of the sorted plan. Combine with `--gpu I` for
the card-pinned 8-way run. Coverage is then a property of the MERGED output
directory, never of one shard — run `scripts/check_coverage.py --artifacts
<out>` after all N shards finish, not after one.

Runner-up policy: `--top-k 2` re-times the two fastest passing variants per
workload rather than one. Cross-GPU selection noise can only ever mis-order
variants that are close, so re-timing the top two and taking the minimum
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

SHARD_NOTE = (
    "Produced by one shard of a card-pinned partition: plan[index::count] over "
    "the sorted plan, a pure function of the problem name, so every replicate "
    "of a problem lands on the same card. COVERAGE IS A PROPERTY OF THE MERGED "
    "OUTPUT DIRECTORY, NOT OF ONE SHARD -- run scripts/check_coverage.py "
    "--artifacts <out> only after all `count` shards have finished."
)


def parse_shard(spec: str) -> tuple[int, int]:
    """`"I/N"` -> (I, N). Raises ValueError on anything else.

    Malformed or out-of-range shard specs are refused rather than clamped: a
    shard that silently becomes 0/1 runs the whole plan on one card, and a
    shard index >= N runs nothing at all while exiting 0. Both look like
    success and both are how a sweep quietly loses coverage (CLAUDE.md 0).
    """
    parts = str(spec).split("/")
    if len(parts) != 2:
        raise ValueError(f"--shard must be I/N (e.g. 3/8), got {spec!r}")
    try:
        index, count = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"--shard must be I/N with integers, got {spec!r}") from None
    if str(index) != parts[0].strip() or str(count) != parts[1].strip():
        raise ValueError(f"--shard must be plain integers I/N, got {spec!r}")
    if count < 1:
        raise ValueError(f"--shard count must be >= 1, got {spec!r}")
    if not 0 <= index < count:
        raise ValueError(
            f"--shard index must be in [0, {count}), got {spec!r}")
    return index, count


def shard_of(items: list, shard: tuple[int, int] | None) -> list:
    """`items[index::count]`, over an already-sorted list.

    Strided rather than blocked on purpose. It needs no communication between
    shards, it is stable under `--top-k`/`--within` changes that alter how long
    a problem takes, and the union over index 0..count-1 is exactly `items`
    with no element in two shards -- which is what makes the merged directory's
    coverage check meaningful.
    """
    if shard is None:
        return list(items)
    index, count = shard
    return list(items)[index::count]


def card_identity(gpu: str, timeout: int = 300) -> dict:
    """The PHYSICAL card `HIP_VISIBLE_DEVICES=<gpu>` selects.

    The torch index is not an identity: it is a position in an ordering that
    differs from amdsmi's and from DRM's, and it changes with
    HIP_VISIBLE_DEVICES. Downstream has to put a problem's T_k on the same card
    as its T_b, so what gets recorded is the card's own name -- PCI BDF, DRM
    card node, amdsmi UUID -- resolved through `scripts/gpu_map.py`, the
    repo's existing PCI-identity resolver, rather than a second one written
    here.

    Resolved in a child process so this script never takes a HIP context on the
    card it is about to hand to a timing run.

    Never fabricates. If the card cannot identify itself the result is
    ``{"identified": False, "error": ...}`` and the run continues: an artifact
    marked unidentified is honest and can be triaged; an absent field or a
    guessed one cannot (prime directives 1 and 5).
    """
    probe = (
        "import json, platform, sys\n"
        "sys.path.insert(0, %r)\n"
        "import gpu_map\n"
        "import torch, amdsmi\n"
        "props = torch.cuda.get_device_properties(0)\n"
        "h = gpu_map.amdsmi_handle(0)\n"
        "out = {'identified': True,\n"
        "       'hostname': platform.node(),\n"
        "       'torch_index_in_child': 0,\n"
        "       'device_name': getattr(props, 'name', None),\n"
        "       'pci_bus_id': getattr(props, 'pci_bus_id', None),\n"
        "       'bdf': str(amdsmi.amdsmi_get_gpu_device_bdf(h)),\n"
        "       'uuid': str(amdsmi.amdsmi_get_gpu_device_uuid(h)),\n"
        "       'drm_card': gpu_map.torch_to_drm_card().get(0)}\n"
        "print('CARD_IDENTITY ' + json.dumps(out))\n"
    ) % str(ROOT / "scripts")
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, timeout=timeout, env=env)
        for line in r.stdout.splitlines():
            if line.startswith("CARD_IDENTITY "):
                got = json.loads(line[len("CARD_IDENTITY "):])
                got["hip_visible_devices"] = str(gpu)
                return got
        err = (r.stderr.strip() or r.stdout.strip() or
               f"probe exited {r.returncode} with no output")
    except Exception as exc:                                      # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    return {"identified": False, "hip_visible_devices": str(gpu),
            "error": err.strip()[-2000:]}


def annotate(out_file: Path, extra: dict) -> None:
    """Merge provenance keys into an artifact the runner already wrote.

    Additive only, and never invents the artifact: if the child wrote nothing,
    or wrote something unreadable, that is left exactly as it is and reported.
    """
    try:
        doc = json.loads(out_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARNING: could not stamp card identity on {out_file.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return
    if not isinstance(doc, dict):
        print(f"  WARNING: {out_file.name} is not a JSON object; "
              f"card identity not stamped", flush=True)
        return
    doc.update(extra)
    out_file.write_text(json.dumps(doc, indent=1))


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


def build_plan(cand_dir: Path, data: Path, top_k: int, within: float
               ) -> tuple[list[tuple[Path, list[str]]], list[str]]:
    """(plan, no_winner), both in sorted-by-key order.

    Sorted is load-bearing: `shard_of` strides over this list, so the order has
    to be a function of the problem names alone and of nothing else -- not of
    directory iteration order, and not of which shard is asking.
    """
    plan: list[tuple[Path, list[str]]] = []
    no_winner: list[str] = []
    for f in sorted(cand_dir.glob("*.json")):
        doc = json.loads(f.read_text())
        # The filename is authoritative for the key. A timeout or crash
        # artifact is written by the shard runner, not the problem runner, and
        # records `problem` as the bare directory name with no category on it.
        key = f.stem
        category, name = key.split("__", 1)
        names = variants_to_retime(doc, top_k, within)
        if not names:
            # No variant passed every workload -> this problem has no anchor.
            # It goes to triage, not to the manifest with a guessed one.
            no_winner.append(key)
            continue
        plan.append((data / category / name, names))
    return plan, no_winner


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Re-time the selected T_b variants.\n"
            "  --gpu G            the frozen MI350X behaviour: the whole plan,\n"
            "                     serially, on card G.\n"
            "  --shard I/N --gpu I  the card-pinned 8-way pass sanctioned by\n"
            "                     STATE.md 4.4 (T_b and T_k re-timed back to\n"
            "                     back on the same card). The module docstring\n"
            "                     says why that follows from 4.4 rather than\n"
            "                     relaxing the one-card rule; read it before\n"
            "                     concluding otherwise.\n"))
    ap.add_argument("--candidates", default="artifacts/06/candidates")
    ap.add_argument("--out", default="artifacts/06/authoritative")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--gpu", default="0",
                    help="the card this process pins every child to "
                         "(HIP_VISIBLE_DEVICES); with --shard I/N it must be I")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="run plan[I::N] of the sorted plan, so each problem "
                         "is pinned to a card by name. Coverage is a property "
                         "of the merged --out, not of one shard.")
    ap.add_argument("--allow-gpu-shard-mismatch", action="store_true",
                    help="permit --shard I/N with --gpu J where I != J. Only "
                         "for a deliberate remap (e.g. a dead card); the run "
                         "still records the card it actually used.")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--within", type=float, default=0.25,
                    help="also re-time any variant within this fraction of "
                         "the fastest, beyond the top k")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    shard = None
    if a.shard is not None:
        try:
            shard = parse_shard(a.shard)
        except ValueError as exc:
            ap.error(str(exc))
        # A shard whose card does not match its index is the mistake sharding
        # newly makes possible: it is invisible in the plan (each shard still
        # takes a disjoint slice) but two shards can then land on ONE card,
        # halving nothing and inflating both sets of timings, or leave a card
        # idle while the operator believes eight are busy. Refused by default,
        # overridable explicitly, and either way the card that actually ran the
        # work is recorded on every artifact.
        if str(a.gpu) != str(shard[0]) and not a.allow_gpu_shard_mismatch:
            ap.error(
                f"--shard {shard[0]}/{shard[1]} with --gpu {a.gpu}: the shard "
                f"index and the card disagree. The 8-way pass is card-pinned "
                f"BY shard index (--shard i/N --gpu i), so a mismatch usually "
                f"means two shards were launched onto one card -- which "
                f"inflates both timings and leaves another card idle, with "
                f"nothing in the artifacts to show it. Pass "
                f"--allow-gpu-shard-mismatch if the remap is deliberate.")

    cand_dir, out_dir = Path(a.candidates), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    runner = ROOT / "scripts" / "runners" / "time_tb_candidates.py"

    plan, no_winner = build_plan(cand_dir, Path(a.data), a.top_k, a.within)
    # Partition both lists by the same stride, so that the union over
    # I = 0..N-1 is exactly the whole set and no key belongs to two shards.
    plan_all, no_winner_all = plan, no_winner
    plan = shard_of(plan, shard)
    no_winner = shard_of(no_winner, shard)

    pending = [(p, v) for p, v in plan
               if not (out_dir / f"{p.parent.name}__{p.name}.json").exists()]

    print(f"candidates   {len(list(cand_dir.glob('*.json')))} problems")
    print(f"re-time      {len(plan)} problems, {len(pending)} pending, "
          f"top-{a.top_k} variants each")
    print(f"no winner    {len(no_winner)}"
          + (f"  (first: {no_winner[:3]})" if no_winner else ""))
    print(f"gpu          {a.gpu}  (authoritative, exclusive)")
    if shard is not None:
        print(f"shard        {shard[0]}/{shard[1]}  "
              f"(plan[{shard[0]}::{shard[1]}] of {len(plan_all)} planned, "
              f"{len(no_winner_all)} with no winner)")
        print(f"             coverage is a property of the MERGED {out_dir}, "
              f"not of this shard")
    if a.dry_run:
        # Sharded: list the WHOLE slice. An operator verifying a partition
        # before spending GPU hours needs all of it, not a preview.
        for p, v in (pending if shard is not None else pending[:10]):
            print(f"  {p.parent.name}/{p.name}: {v}")
        return 0

    card = card_identity(a.gpu)
    if not card.get("identified"):
        print(f"WARNING: card {a.gpu} could not identify itself; artifacts "
              f"will be stamped unidentified. {card.get('error', '')[:300]}",
              flush=True)
    else:
        print(f"card         {card.get('uuid')}  {card.get('bdf')}  "
              f"{card.get('hostname')}")
    stamp = {"card_identity": card}
    if shard is not None:
        stamp["shard"] = {"index": shard[0], "count": shard[1],
                          "gpu": str(a.gpu), "_note": SHARD_NOTE}

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
        # Which physical card produced this number, stamped whether the run
        # succeeded or not: a failure that only happens on one card is a fact
        # about that card, and the artifact is the only place it can be seen.
        if out_file.exists():
            annotate(out_file, stamp)
        ok, failed = ok + good, failed + (not good)
        elapsed = time.time() - start
        eta = elapsed / n * (len(pending) - n)
        print(f"[{n}/{len(pending)}] {'ok' if good else 'FAIL':<4} {key}  "
              f"({','.join(names)})  eta {eta/60:.0f}m", flush=True)

    print(f"\ndone: {ok} ok, {failed} failed, {(time.time()-start)/60:.1f} min")
    if no_winner:
        # Per-shard filename when sharded: eight processes writing one
        # no-winner.json is a last-writer-wins race, and the loser's problems
        # simply vanish from triage. The shard's own slice, in its own file.
        nw_file = (out_dir / "no-winner.json" if shard is None else
                   out_dir / f"no-winner.shard{shard[0]}of{shard[1]}.json")
        doc = {"_note": "No variant passed every workload, so these have no T_b "
                        "anchor. Triage before the manifest is cut -- a problem "
                        "without an anchor is not scoreable and must be counted "
                        "as such.",
               "problems": no_winner}
        if shard is not None:
            doc["shard"] = {"index": shard[0], "count": shard[1],
                            "gpu": str(a.gpu), "_note": SHARD_NOTE}
        nw_file.write_text(json.dumps(doc, indent=1))
        print(f"{len(no_winner)} problems have no passing variant; "
              f"see {nw_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
