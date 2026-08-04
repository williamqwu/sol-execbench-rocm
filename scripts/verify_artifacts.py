#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Acceptance checks. A task is done when its check passes -- not before.

    python scripts/verify_artifacts.py --task 01
    python scripts/verify_artifacts.py --task 09 --full

These are deliberately mechanical. The point is that "done" is decided by a
program, not by a judgement call made at the end of a long session.

Machine-checkable things are checked. Things that are not (was F_LOCK chosen
sensibly? is a tolerance justified?) are reported as REQUIRES-JUDGEMENT so they
show up rather than being silently skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
STATE = ROOT / "STATE.md"

PASS, FAIL, WARN, JUDGE = "PASS", "FAIL", "WARN", "REQUIRES-JUDGEMENT"


class Checks:
    def __init__(self):
        self.results: list[tuple[str, str, str]] = []

    def add(self, status, name, detail=""):
        self.results.append((status, name, detail))

    def require(self, cond, name, detail_ok="", detail_bad=""):
        self.add(PASS if cond else FAIL, name, detail_ok if cond else detail_bad)
        return cond

    def report(self) -> int:
        width = max((len(n) for _, n, _ in self.results), default=20)
        for status, name, detail in self.results:
            print(f"  [{status:<18}] {name:<{width}}  {detail}")
        failed = sum(1 for s, _, _ in self.results if s == FAIL)
        judge = sum(1 for s, _, _ in self.results if s == JUDGE)
        print(f"\n  {len(self.results)} checks, {failed} failed, "
              f"{judge} require human judgement")
        return 1 if failed else 0


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def has_provenance(doc) -> bool:
    if not isinstance(doc, dict):
        return False
    prov = doc.get("_provenance", {})
    return bool(prov.get("utc") and prov.get("git_sha") is not None)


def state_text() -> str:
    return STATE.read_text() if STATE.exists() else ""


def f_lock_from_state() -> int | None:
    """F_LOCK as STATE.md declares it, from a canonical marker only.

    Deliberately narrow. The original pattern was ``F_LOCK.*?(\\d{3,4})``, which
    matches the *first* number after the *first* mention of F_LOCK anywhere in the
    file — a prose sentence, a table cell, a deviation write-up, whichever comes
    first. That is fine while STATE.md documents one part and silently wrong as
    soon as it documents two: on an MI355X node whose STATE.md still discussed the
    MI350X bound, this resolved to 1300 and the acceptance check reported

        [PASS] F_LOCK recorded in STATE.md                1300 MHz
        [PASS] F_LOCK at or below lowest observed floor   F_LOCK 1300 <= min p5 1724

    Both green, and neither could have failed: 1300 clears a 1724 floor so
    comfortably that no wrong answer would ever trip it. A check that cannot fail
    is worse than no check, because it is read as evidence.

    The marker is ``**F_LOCK = <n> MHz**``, written once, in *Decisions taken*.
    """
    m = re.search(r"^\s*\*\*F_LOCK\s*=\s*(\d{3,4})\s*MHz\*\*", state_text(),
                  re.I | re.M)
    return int(m.group(1)) if m else None


def determinism_setpoints() -> dict[int, int]:
    """{gpu -> GFX MAX_CLK in MHz} as the *hardware* reports it.

    The determinism setpoint read back off the device, not the one the preset
    table says was requested. `amd-smi metric -c` exposes it as MAX_CLK per GFX
    block, and it needs no load and no timed region -- an idle GPU reports the
    ceiling it is pinned to.

    This exists because `provenance.f_lock_mhz()` answers from the preset table
    without consulting a device, so an artifact's stamp records what was *meant*
    to be applied. If a node is left at a different setpoint by an earlier sweep,
    every artifact is stamped with a clock it was not measured at, the manifest's
    clock guard compares that stamp against the same table it came from, and it
    agrees with itself. The table is not the hardware.
    """
    out: dict[int, int] = {}
    for idx in range(16):
        try:
            p = subprocess.run(["amd-smi", "metric", "-g", str(idx), "-c"],
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            break
        if p.returncode != 0:
            break
        m = re.search(r"MAX_CLK:\s*(\d+)\s*MHz", p.stdout)
        if not m:
            break
        out[idx] = int(m.group(1))
    return out


def requested_clock_from_preset() -> int | None:
    """The setpoint ``lock_clocks()`` asks the driver for, not the achieved one."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import torch

        from sol_execbench.core.bench.config import get_clock_preset

        preset = get_clock_preset(torch.cuda.get_device_name(0))
        return preset.gpu_clk_mhz if preset else None
    except Exception:
        return None


def f_lock_from_preset() -> tuple[int | None, str | None]:
    """(F_LOCK, part) as the *code* will use it, from ``CLOCK_LOCK_PRESETS``.

    This is the value every T_SOL and T_b is actually expressed at, because it is
    the one ``provenance.stamp()`` records and the one ``lock_clocks()`` applies.
    If it and STATE.md disagree, one of them is lying about the frequency the whole
    benchmark is calibrated to, and nothing downstream can tell which.
    """
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import torch

        from sol_execbench.core.bench.config import get_clock_preset
        from solexbench_rocm.parts import detect_part

        preset = get_clock_preset(torch.cuda.get_device_name(0))
        return (preset.f_lock_mhz if preset else None), detect_part().name
    except Exception:
        return None, None


# --------------------------------------------------------------------------

def check_00(c: Checks):
    rep = load_json(ART / "00" / "node-report.json")
    if not c.require(rep is not None, "node-report.json exists",
                     detail_bad="run scripts/node_acceptance.sh"):
        return
    c.require(has_provenance(rep), "node report has provenance")

    gpus = rep.get("gpus", [])
    c.require(len(gpus) == 8, "8 GPUs present", f"found {len(gpus)}",
              f"found {len(gpus)} — record why before proceeding")
    c.require(all("gfx950" in str(g.get("arch", "")) for g in gpus),
              "all GPUs are gfx950")

    # Step 2 of the task is "confirm all eight GPUs are equivalent", which means
    # cap, clock ceiling AND idle temperature -- not just cap. An unprobed field
    # must fail rather than silently skip the check it was supposed to satisfy.
    for field, label, tol in (("power_cap_w", "power caps", 0.05),
                              ("sclk_max_mhz", "max GFX clocks", 0.05),
                              ("temp_c", "idle temperatures", 0.25)):
        vals = [g.get(field) for g in gpus if g.get(field)]
        if not c.require(len(vals) == len(gpus) and len(gpus) > 0,
                         f"{label} probed on every GPU",
                         f"{len(vals)}/{len(gpus)}",
                         f"only {len(vals)}/{len(gpus)} — equivalence unproven; "
                         f"see smi_error in node-report.json"):
            continue
        med = sorted(vals)[len(vals) // 2]
        outliers = [x for x in vals if med and abs(x - med) / med > tol]
        c.require(not outliers, f"{label} uniform (±{tol:.0%})",
                  f"all {med}",
                  f"outliers: {outliers} vs median {med} — a GPU that differs "
                  f"here will quietly produce different timings")

    roof = load_json(ART / "00" / "roofline-gpu0.json")
    c.require(roof is not None and roof.get("hbm_tbs"), "HBM roofline measured")
    c.require(roof is not None and roof.get("gemm_bf16_tflops"),
              "BF16 GEMM roofline measured")
    c.add(JUDGE, "dataset layout matches audit",
          "confirm categories L1=94 L2=82 Quant=33 FlashInfer=26")


def check_01(c: Checks):
    fl = f_lock_from_state()
    c.require(fl is not None, "F_LOCK recorded in STATE.md",
              f"{fl} MHz",
              "no canonical `**F_LOCK = <n> MHz**` line — blocks tasks 03, 05, 06")

    preset_fl, part = f_lock_from_preset()
    c.require(preset_fl is not None, "F_LOCK present in CLOCK_LOCK_PRESETS",
              f"{preset_fl} MHz for {part}",
              "no preset for this device — lock_clocks() will refuse and every "
              "artifact's f_lock_mhz will be null")
    if fl is not None and preset_fl is not None:
        # The one comparison that catches a stale document or a stale constant.
        # Both are load-bearing: the preset is what gets applied and stamped, the
        # document is what a human reads, and a benchmark whose two records of its
        # own clock disagree cannot be trusted to a percent.
        c.require(fl == preset_fl,
                  "STATE.md and CLOCK_LOCK_PRESETS agree on F_LOCK",
                  f"both {fl} MHz",
                  f"STATE.md says {fl} MHz, code applies {preset_fl} MHz — one of "
                  f"them is wrong and every T_SOL and T_b depends on which")

    # The hardware's own answer, compared against what the code asks for. This
    # is the gap the clock guard in build_manifest cannot close on its own: it
    # checks a stamp against the table the stamp came from, so a node left at
    # someone else's setpoint passes by agreeing with itself. Reading MAX_CLK
    # back off every GPU catches exactly that, with no load and no timed region.
    setpoints = determinism_setpoints()
    requested = requested_clock_from_preset()
    if setpoints and requested:
        wrong = {g: v for g, v in setpoints.items() if v != requested}
        c.require(not wrong,
                  "every GPU is at the preset's determinism setpoint",
                  f"all {len(setpoints)} GPUs at {requested} MHz",
                  f"{len(wrong)} GPU(s) at a different setpoint {wrong} while the "
                  f"preset requests {requested} — artifacts measured now would be "
                  f"stamped {requested} and be wrong by the ratio")
    elif not setpoints:
        c.add(JUDGE, "determinism setpoint read back off the GPUs",
              "amd-smi unavailable — the stamp cannot be checked against hardware")

    floors = list((ART / "01").glob("floor-gpu*.json")) if (ART / "01").exists() else []
    c.require(len(floors) >= 3, "clock floor sampled on >=3 GPUs",
              f"{len(floors)} GPUs",
              f"only {len(floors)} — per-GPU variation would go unseen")

    p5s = []
    for f in floors:
        d = load_json(f) or {}
        ss = d.get("steady_state") or {}
        if ss.get("p5_mhz"):
            p5s.append(ss["p5_mhz"])
    # Fall back to the preset when STATE.md has no marker. Gating this on `fl`
    # alone means that tightening the document check DELETES the physics check:
    # a missing marker is a documentation defect, but "F_LOCK exceeds the clock
    # the GPU can actually hold" is the failure that makes every timing drift,
    # and it must still be evaluated. Losing a real check as a side effect of
    # adding a stricter one is the same trade F17 was written to stop.
    fl_eff = fl if fl is not None else preset_fl
    if p5s and fl_eff:
        src = "STATE.md" if fl is not None else "CLOCK_LOCK_PRESETS"
        c.require(fl_eff <= min(p5s), "F_LOCK at or below lowest observed floor",
                  f"F_LOCK {fl_eff} ({src}) <= min p5 {min(p5s)}",
                  f"F_LOCK {fl_eff} ({src}) EXCEEDS lowest floor {min(p5s)} — the "
                  f"GPU cannot hold this; every timing will drift")
        if len(p5s) > 1 and max(p5s) - min(p5s) > 50:
            c.add(WARN, "per-GPU floor spread >50MHz",
                  f"{min(p5s)}-{max(p5s)} MHz; F_LOCK must suit the worst")

    stab = load_json(ART / "01" / "stability-gpu0.json")
    if c.require(stab is not None, "stability measured"):
        cv = stab.get("cv")
        c.require(cv is not None and cv < 0.02, "timing CV < 2%",
                  f"CV={cv:.4f}",
                  f"CV={cv} — noise will swamp real optimization differences")

    intf = load_json(ART / "01" / "interference.json")
    if c.require(intf is not None, "sibling interference measured",
                 detail_bad="this result shapes the whole schedule"):
        c.require(bool(intf.get("scheduling_consequence")),
                  "interference has a stated scheduling consequence",
                  intf.get("verdict", ""))


EXPECTED_CATEGORIES = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}
EXPECTED_TOTAL = sum(EXPECTED_CATEGORIES.values())   # 235


def check_full_coverage(c: Checks, artifact_dir: Path, pattern: str | None = None):
    """Scope is all 235. Omission is the realistic failure mode, not decision."""
    keys = ({f"{p.parent.parent.name}__{p.parent.name}"
             for p in artifact_dir.rglob(pattern)} if pattern
            else {p.stem for p in artifact_dir.glob("*.json")}) \
        if artifact_dir.exists() else set()

    by_cat = {cat: sum(1 for k in keys if k.startswith(f"{cat}__"))
              for cat in EXPECTED_CATEGORIES}
    deferred = {}
    df = ART / "deferred.json"
    if df.exists():
        deferred = (load_json(df) or {}).get("problems", {})

    for cat, exp in EXPECTED_CATEGORIES.items():
        got = by_cat[cat] + sum(1 for k in deferred if k.startswith(f"{cat}__"))
        c.require(got >= exp, f"coverage {cat} ({exp} problems)",
                  f"{by_cat[cat]}/{exp}",
                  f"{by_cat[cat]}/{exp} — {exp - got} neither run nor deferred; "
                  f"run scripts/check_coverage.py to name them")
    total = sum(by_cat.values()) + len(deferred)
    c.require(total >= EXPECTED_TOTAL, f"coverage total ({EXPECTED_TOTAL})",
              f"{sum(by_cat.values())} covered + {len(deferred)} deferred",
              f"only {total}/{EXPECTED_TOTAL} accounted for")


def check_02(c: Checks):
    # The sweep that counts is the one run against the tolerances the
    # benchmark actually ships -- the AMD-derived ones from task 05. The
    # original sweep against the dataset's B200 tolerances is kept beside it
    # and compared below, because the difference between them is the whole
    # argument for task 05 existing.
    d = ART / "02" / "references-amd"
    b200 = ART / "02" / "references"
    if not c.require(d.exists(), "reference sweep ran (AMD tolerances)",
                     detail_bad="run shard_sweep with "
                                "SOLEXBENCH_WORKLOADS_ROOT=artifacts/05/workloads"):
        return
    check_full_coverage(c, d)
    results = [load_json(p) for p in d.glob("*.json")]
    results = [r for r in results if r]
    total = len(results)
    c.require(total > 0, "reference results present", f"{total} problems")

    # Pass rate is over WORKLOADS. Per problem is the wrong denominator: a
    # problem with one failing workload out of sixteen is not a failed
    # problem, and counting it as one hides the fifteen that work.
    deferred_keys = set((load_json(ART / "deferred.json") or {}).get("problems", {}))
    wl = [w for r in results if r.get("problem") not in deferred_keys
          for w in (r.get("per_workload") or [])]
    passed = sum(1 for w in wl if w.get("status") == "PASSED")
    rate = passed / len(wl) if wl else 0
    c.require(rate >= 0.95, "reference pass rate >= 95% of workloads",
              f"{passed}/{len(wl)} ({rate:.1%})",
              f"{passed}/{len(wl)} ({rate:.1%}) — triage before proceeding")

    # A failure with no explanation is the one outcome that must not happen:
    # it is indistinguishable from a sweep that silently skipped the problem.
    undocumented = [r for r in results
                    if not r.get("ok", True) and not r.get("error")]
    # A numerical failure documents itself with its error statistics; only a
    # failure with neither a log nor an error figure is unexplained.
    undocumented += [w for w in wl
                     if w.get("status") not in ("PASSED", None) and not w.get("log")
                     and w.get("max_absolute_error") is None]
    c.require(not undocumented, "every failure has a recorded error",
              detail_bad=f"{len(undocumented)} failures with no error text")
    c.require(all(w.get("methodology") for w in wl),
              "traces record timing methodology",
              f"{len({w.get('methodology') for w in wl})} distinct value(s)")
    c.require((ART / "02" / "flush-sweep.json").exists(),
              "LLC flush-size bandwidth cliff recorded")

    # The comparison that justifies task 05: how many workloads the SAME
    # references fail when scored against B200's tolerances instead.
    if b200.exists():
        old_wl = [w for p in b200.glob("*.json")
                  for w in ((load_json(p) or {}).get("per_workload") or [])
                  if (load_json(p) or {}).get("problem") not in deferred_keys]
        old_bad = sum(1 for w in old_wl if w.get("status") != "PASSED")
        new_bad = len(wl) - passed
        c.add(PASS, "AMD vs B200 tolerances on the same references",
              f"{old_bad} workloads fail under B200's, {new_bad} under AMD's")


def check_03(c: Checks):
    t = load_json(ART / "03" / "t_sol.json")
    if not c.require(t is not None, "t_sol.json exists"):
        return
    c.require(has_provenance(t), "t_sol has provenance")
    entries = t.get("problems", {})
    c.require(len(entries) > 0, "t_sol covers problems", f"{len(entries)}")

    # Per WORKLOAD, not per problem: the bound is defined per workload
    # instance, and a problem-level check passes while most of its workloads
    # carry nothing.
    bounded = [w for e in entries.values()
               for w in (e.get("workloads") or {}).values()
               if w.get("t_sol_cycles") is not None]
    both = [w for w in bounded if w.get("t_sol_ms") is not None]
    c.require(bounded and len(both) == len(bounded),
              "every bound recorded in cycles AND ms",
              f"{len(both)} workloads",
              "the cycles column is what makes an F_LOCK change a division "
              "rather than a re-run")
    c.require(all(w["t_sol_cycles"] > 0 for w in bounded),
              "no bound is zero cycles",
              detail_bad="a zero bound divides by (T_b - 0) in the score")

    problems_with_bounds = sum(
        1 for e in entries.values()
        if any(w.get("t_sol_cycles") is not None
               for w in (e.get("workloads") or {}).values()))
    c.add(PASS if problems_with_bounds else FAIL,
          "problems with at least one bound",
          f"{problems_with_bounds}/{len(entries)}; the rest are SOLAR "
          f"limitations, recorded per workload with the error that caused them")

    # Upstream ships no per-workload SOL times, so the cross-checks are
    # internal. Read their own report rather than restating their logic here.
    xc = (ART / "03" / "cross-checks.md")
    if c.require(xc.exists(), "cross-checks report exists",
                 detail_bad="run scripts/sol_cross_checks.py"):
        text = xc.read_text()
        m = re.search(r"implied bandwidth above DRAM peak: \*\*(\d+)\*\*", text)
        n = re.search(r"implied FLOPS above the precision's peak: \*\*(\d+)\*\*",
                      text)
        k = re.search(r"MISMATCHes: \*\*(\d+)\*\*", text)
        c.require(m and n and int(m.group(1)) == 0 and int(n.group(1)) == 0,
                  "check B: no bound implies an impossible rate")
        c.require(k and int(k.group(1)) == 0,
                  "check C: hand-derived MAC counts agree")
        c.add(JUDGE, "check A: 13 problems below declared traffic",
              "mechanism stated per problem in cross-checks.md")
        _check_d(c)
    c.add(JUDGE, "V1/V2/V3 resolved (TF32, LLC bandwidth, MXFP4 dense)",
          "see STATE.md decisions")


def check_05(c: Checks):
    d = ART / "05" / "workloads"
    if not c.require(d.exists(), "tolerance sweep ran"):
        return
    files = list(d.rglob("workload.jsonl"))
    c.require(len(files) > 0, "AMD workload files produced", f"{len(files)}")
    check_full_coverage(c, d, "workload.jsonl")

    # Prime directive 2: no NVIDIA constant may survive into an AMD artifact.
    b200 = ROOT / "reference" / "b200-tolerances.json"
    if b200.exists():
        upstream = load_json(b200) or {}
        exact = []
        for f in files:
            for line in f.read_text().splitlines():
                try:
                    w = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = f"{f.parent.name}:{w.get('name', '')}"
                if key in upstream and upstream[key] == [
                    w.get("atol"), w.get("rtol"), w.get("matched_ratio")]:
                    exact.append(key)
        c.require(not exact, "no tolerance copied verbatim from B200",
                  detail_bad=f"{len(exact)} exact matches — e.g. {exact[:3]}")
    else:
        c.add(WARN, "B200 tolerance reference absent",
              "copy-detection skipped; add reference/b200-tolerances.json")

    c.require((ART / "05" / "triage.md").exists(),
              "per-problem triage recorded")
    c.add(JUDGE, "problems needing >2x B200 tolerance individually justified")


def _check_d(c: Checks):
    """check D: no measured time may fall below its own T_SOL.

    Previously this line:

        c.add(JUDGE if "PENDING" in text else PASS,
              "check D: T_SOL <= best measured", "needs task 06")

    `cross-checks.md` contains no "PENDING" and no check-D section at all, so it
    was an unconditional PASS that compared nothing. It is also the single
    invariant that would have caught D18 -- a correct kernel measured 3x faster
    than the roofline on 25 workloads of a paged FlashInfer problem, because the
    bound prices a KV cache at its full allocation rather than at the pages the
    workload names.

    It now compares T_SOL against every measurement on disk. The T_b variants
    alone cannot falsify a bound that is too slow, because the reference
    over-reads the same way T_SOL does; agent submissions can, and do.
    """
    manifest = load_json(ART / "09" / "manifest-v1.json")
    if not c.require(manifest is not None, "check D: manifest available"):
        return

    bounds = {}
    for key, p in (manifest.get("problems") or {}).items():
        for uuid, w in (p.get("workloads") or {}).items():
            if w.get("scoreable") and w.get("t_sol_ms"):
                bounds[(key, uuid)] = w["t_sol_ms"]

    violations, n_measured, sources = [], 0, 0
    for scored in sorted((ART / "10").glob("*/scored.json")) if (ART / "10").exists() else []:
        doc = load_json(scored) or {}
        sources += 1
        for r in doc.get("results", []):
            t_sol = bounds.get((r.get("problem"), r.get("workload_uuid")))
            lat = r.get("latency_ms")
            if r.get("status") != "PASSED" or not t_sol or not lat:
                continue
            n_measured += 1
            if lat < t_sol:
                violations.append((r["problem"], lat / t_sol))

    if not sources:
        c.add(JUDGE, "check D: T_SOL <= best measured",
              "no submissions on disk — the T_b variants cannot falsify a bound "
              "that is too slow, so this is untested, not passing")
        return

    worst = min((v for _, v in violations), default=None)
    bad = sorted({p for p, _ in violations})
    c.require(not violations, "check D: no measurement beats its T_SOL",
              f"{n_measured} measured workloads, none below bound",
              f"{len(violations)} of {n_measured} measured workloads are faster "
              f"than T_SOL (worst {worst:.2f}x the bound) across {len(bad)} "
              f"problem(s): {', '.join(p[:44] for p in bad[:3])} — the bound is "
              f"wrong (STATE.md D18)" if violations else "")


def check_06(c: Checks):
    """T_b coverage, read from the layout task 06 actually writes.

    This check asserted a schema that was never produced: a single
    `artifacts/06/t_b.json` with a `problems` map of `winning_variant`, and an
    `anchor-verification.md`. Task 06 writes one file per problem under
    `authoritative/` keyed by `winner_by_workload`, and the anchor result as
    `.json`. So `t_b.json exists` failed on every run this repo has ever had,
    while STATE.md recorded the task as done -- the mirror image of F17: not a
    check that could not fail, but one that could not pass. Both report
    something other than the state of the work.
    """
    auth = ART / "06" / "authoritative"
    docs = sorted(auth.glob("*.json")) if auth.exists() else []
    if not c.require(bool(docs), "authoritative T_b artifacts exist",
                     f"{len(docs)} problems",
                     "artifacts/06/authoritative/ is empty — no problem has an "
                     "anchor and nothing is scoreable"):
        return

    deferred = (load_json(ART / "deferred.json") or {}).get("problems", {})
    expected = EXPECTED_TOTAL - len(deferred)
    with_tb = [d for d in docs
               if (load_json(d) or {}).get("winner_by_workload")]
    c.require(len(with_tb) >= expected,
              f"T_b covers all {expected} non-deferred problems",
              f"{len(with_tb)} of {expected}",
              f"only {len(with_tb)} — every scoreable problem needs an anchor")

    # A named variant, not "optimized PyTorch": the anchor has to be
    # reproducible by someone who has only the manifest.
    unnamed = []
    for d in with_tb:
        wins = (load_json(d) or {}).get("winner_by_workload") or {}
        if not all((w or {}).get("variant") for w in wins.values()):
            unnamed.append(d.name)
    c.require(not unnamed, "winning PyTorch variant recorded per workload",
              f"all {len(with_tb)} problems",
              f"{len(unnamed)} problem(s) name no variant, e.g. {unnamed[:3]}")

    # Every T_b must have been measured at one clock. This is F18's invariant
    # applied at acceptance time rather than only at manifest-build time.
    clocks = {}
    for d in with_tb:
        mhz = ((load_json(d) or {}).get("_provenance") or {}).get("f_lock_mhz")
        clocks[mhz] = clocks.get(mhz, 0) + 1
    measured = {k: v for k, v in clocks.items() if k is not None}
    c.require(len(measured) <= 1, "all T_b measured at one clock",
              f"F_LOCK {next(iter(measured), None)} across {sum(measured.values())}",
              f"T_b spans {len(measured)} clocks {measured} — mixing them "
              f"rescales those problems' scores (F18)")

    anchor = load_json(ART / "06" / "anchor-verification.json")
    if c.require(anchor is not None, "anchor property verified",
                 detail_bad="T_b must score 0.5+-0.03; reference must not "
                            "score above the anchor"):
        # Read the fields this artifact actually has. Written first against
        # guessed names (`n_failed`), which resolved to None and printed
        # "every checked workload within tolerance" over 13 real failures --
        # the same defect being audited, reintroduced while auditing it. A
        # check keyed on a field that does not exist always passes.
        ap = anchor.get("anchor_property") or {}
        rp = anchor.get("reference_not_above_anchor") or {}
        passing, total = ap.get("passing"), ap.get("total")
        c.require(passing is not None and total,
                  "anchor artifact reports passing/total",
                  f"{passing}/{total}",
                  "schema changed — this check cannot evaluate anything")
        if passing is not None and total:
            # Not 100%: D15 records 12 of the 13 on one problem, understood and
            # in the conservative direction. A WARN keeps it visible without
            # asserting a clean result that is not clean.
            c.add(PASS if passing == total else WARN,
                  "T_b re-times to S = 0.5 +- 0.03",
                  f"{passing}/{total} workloads"
                  + ("" if passing == total else " — see STATE.md D15"))
        c.require(rp.get("passing") == rp.get("total") and rp.get("total"),
                  "reference never scores above the anchor",
                  f"{rp.get('passing')}/{rp.get('total')}",
                  f"{(rp.get('total') or 0) - (rp.get('passing') or 0)} workload(s) "
                  f"where the reference beats T_b — the anchor is not the fastest "
                  f"variant and the scale is wrong")
        viol = anchor.get("t_sol_violations")
        c.require(viol == [], "no sampled workload beat its T_SOL",
                  "0 violations over the sample",
                  f"{len(viol or [])} workload(s) measured faster than T_SOL — "
                  f"the bound is wrong, not the kernel (see STATE.md D18)")
    c.add(JUDGE, "authoritative pass ran under documented node conditions",
          "quiet vs busy, per task 01 interference verdict")


def check_07(c: Checks):
    spike = load_json(ART / "07" / "spike.json")
    c.require(spike is not None, "MXFP4 feasibility spike ran")
    if spike:
        c.require(spike.get("verdict") in ("go", "no-go"),
                  "spike has an explicit verdict", str(spike.get("verdict")))
    # Check the evidence, not a prose file. This required
    # `artifacts/07/fp8-validation.md`, which was never written, so it failed on
    # every run while the validation it stands for had in fact been done: all 18
    # non-NVFP4 Quant problems pass every workload in the task 02 reference
    # sweep. Asserting the existence of a write-up says nothing about whether
    # FP8 works, and it reported a missing document as a missing measurement.
    refs = ART / "02" / "references"
    fp8 = sorted(p for p in (ROOT / "data" / "SOL-ExecBench" / "benchmark" /
                             "Quant").glob("*") if p.is_dir()
                 and "nvfp4" not in p.name)
    if not fp8:
        c.add(JUDGE, "FP8 (18 problems) validation recorded",
              "dataset not materialized — cannot verify from here")
    else:
        results = [(load_json(refs / f"Quant__{p.name}.json") or {}) for p in fp8]
        passing = [r for r in results if r.get("all_passed")]
        c.require(len(passing) == len(fp8),
                  f"FP8 ({len(fp8)} problems) validated on this part",
                  f"{len(passing)}/{len(fp8)} pass every workload in the task 02 "
                  f"reference sweep",
                  f"only {len(passing)}/{len(fp8)} — CDNA4 is OCP FP8 and these "
                  f"were expected to port directly")
        doc = ART / "07" / "fp8-validation.md"
        c.add(PASS if doc.exists() else WARN, "FP8 result written up",
              str(doc) if doc.exists()
              else "evidence is in artifacts/02/references/Quant__*.json; no "
                   "summary document")
    st = state_text()
    if spike and spike.get("verdict") == "no-go":
        c.require("220" in st, "deferral documented with problem count",
                  detail_bad="if shipping 220 not 235, say so everywhere")


def check_08(c: Checks):
    r = load_json(ART / "08" / "replay-results.json")
    if not c.require(r is not None, "exploit corpus replayed"):
        return
    # "passed" means the exploit was detected OR neutralized, and the corpus
    # states which per case. Both are acceptable outcomes; neither is
    # "the submission got away with it".
    total, detected = r.get("n_cases", 0), r.get("n_passed", 0)
    c.require(total > 0 and detected == total
              and r.get("all_detected_or_neutralized") is True,
              "every exploit detected or neutralized",
              f"{detected}/{total}",
              f"{detected}/{total} — a miss is a release blocker")
    c.require((ART / "08" / "amd-specific.md").exists(),
              "AMD-specific probes recorded (streams, smi, XCD, LDS)")
    # A detector that fires on the problem's own reference would fail every
    # honest submission, so this is checked rather than eyeballed: the
    # reference sweep is the largest corpus of known-good submissions there is.
    refs = ART / "02" / "references-amd"
    if refs.exists():
        flagged = []
        for f in refs.glob("*.json"):
            doc = load_json(f) or {}
            for w in doc.get("per_workload") or []:
                if "REWARD_HACK" in str(w.get("status", "")):
                    flagged.append(f"{doc.get('problem')}:{w.get('workload_uuid')}")
        c.require(not flagged,
                  "no false positives on the reference sweep",
                  f"0 of 235 problems flagged",
                  f"{len(flagged)} flagged: {flagged[:3]}")
    else:
        c.add(JUDGE, "no false positives on the task-02 reference sweep")


def check_09(c: Checks, full=False):
    m = load_json(ART / "09" / "manifest-v1.json")
    if not c.require(m is not None, "scoring manifest exists"):
        return
    c.require(has_provenance(m), "manifest has provenance")
    probs = m.get("problems", {})
    c.require(len(probs) + len((load_json(ART / "deferred.json") or {}).get(
        "problems", {})) >= EXPECTED_TOTAL,
        f"manifest accounts for all {EXPECTED_TOTAL} problems",
        f"{len(probs)} in manifest",
        f"{len(probs)} in manifest — the rest must be in deferred.json")
    # Completeness is a property of WORKLOADS -- the manifest keys t_sol, t_b
    # and tolerance per workload instance, because that is where they differ.
    deferred_keys = set((load_json(ART / "deferred.json") or {}).get("problems", {}))
    incomplete = []
    for k, v in probs.items():
        if k in deferred_keys:
            continue
        for u, e in (v.get("workloads") or {}).items():
            if not all(e.get(x) is not None
                       for x in ("t_sol_ms", "t_sol_cycles", "t_b_ms", "tolerance")):
                incomplete.append(f"{k}:{u[:8]}")
    c.require(not incomplete,
              "every non-deferred workload has t_sol, t_b and a tolerance",
              f"{m['stats']['scoreable_workloads']} workloads",
              f"{len(incomplete)} incomplete: {incomplete[:3]}")

    # Every bound says which derivation produced it. Without this a consumer
    # cannot tell a roofline over the arithmetic from a pure traffic floor.
    sources = {e.get("t_sol_source")
               for v in probs.values() for e in (v.get("workloads") or {}).values()
               if e.get("t_sol_ms") is not None}
    c.require(None not in sources and sources,
              "every bound records its derivation", ", ".join(sorted(map(str, sources))))

    prov = m.get("_provenance") or {}
    c.require(m.get("methodology"), "manifest records the timing methodology",
              str(m.get("methodology")))
    c.require(prov.get("f_lock_mhz"), "manifest records f_lock_mhz",
              f"{prov.get('f_lock_mhz')} MHz")
    c.require((prov.get("rocm") or {}).get("version"),
              "manifest records rocm_version",
              str((prov.get("rocm") or {}).get("version")))
    c.require((prov.get("torch") or {}).get("version"),
              "manifest records torch_version",
              str((prov.get("torch") or {}).get("version")))
    if full:
        base = load_json(ART / "09" / "agent-baseline.json")
        if base is None:
            c.add(FAIL, "agent baseline accounted for",
                  "no artifacts/09/agent-baseline.json — run it, or record "
                  "explicitly that it was not run and why")
        elif base.get("ran"):
            c.require(base.get("median") is not None,
                      "agent baseline sweep ran", f"median {base.get('median')}")
        else:
            c.add(JUDGE, "agent baseline NOT run",
                  str(base.get("reason", ""))[:90])
        dist = load_json(ART / "09" / "score-distribution.json")
        if c.require(dist is not None, "score distribution computed"):
            ac = dist.get("anchor_check") or {}
            dev = ac.get("max_abs_deviation_from_half")
            c.require(dev is not None and dev <= 1e-6,
                      "S = 0.5 at T_b, by construction",
                      f"max |S-0.5| = {dev:.1e}" if dev is not None else "",
                      "T_b in the manifest is not the time that "
                      "implementation actually takes")
        anchor = load_json(ART / "06" / "anchor-verification.json")
        if c.require(anchor is not None, "anchor re-verified on hardware"):
            ap = anchor["anchor_property"]
            c.require(not anchor["t_sol_violations"],
                      "no measured time below its own T_SOL",
                      f"{anchor['workloads_checked']} workloads checked")
            c.require(ap["passing"] / max(ap["total"], 1) >= 0.95,
                      "re-timed anchor scores 0.5 +- tol",
                      f"{ap['passing']}/{ap['total']}")
        readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
        c.require("within-platform" in readme.lower(),
                  "cross-vendor caveat present in README",
                  detail_bad="this will be the most misread number in the "
                             "project — state it explicitly")


CHECKS = {"00": check_00, "01": check_01, "02": check_02, "03": check_03,
          "05": check_05, "06": check_06, "07": check_07, "08": check_08}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    c = Checks()
    print(f"\nAcceptance check — task {a.task}\n")
    if a.task == "09":
        check_09(c, a.full)
    elif a.task in CHECKS:
        CHECKS[a.task](c)
    elif a.task == "04":
        cmp_dir = ART / "04" / "compare"
        c.require(cmp_dir.exists(), "methodology comparison ran")
        c.require((ART / "04" / "clock-domain-verification.log").exists(),
                  "clock domain verified on real captures",
                  detail_bad="ROCM CONTRACT #1 — wrong domain fails silently")
        rep = ART / "04" / "methodology-comparison.md"
        if c.require(rep.exists(), "methodology report written",
                     detail_bad="run scripts/methodology_report.py"):
            text = rep.read_text()
            m = re.search(r"kernels >= 100 us: ([-+][\d.]+)%", text)
            c.require(m and abs(float(m.group(1))) <= 2.0,
                      "median hip_events vs rocprof divergence <= 2%",
                      f"{m.group(1)}%" if m else "",
                      "the two methodologies are not interchangeable at this "
                      "spread; a trace's methodology field is not enough")
            c.add(JUDGE, "divergence tails",
                  "330/1430 pairs differ by >20%; mechanism in the report")
    else:
        print(f"no automated check for task {a.task}", file=sys.stderr)
        sys.exit(2)

    sys.exit(c.report())


if __name__ == "__main__":
    main()
