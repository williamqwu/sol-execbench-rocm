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
    m = re.search(r"F_LOCK.*?(\d{3,4})\s*(?:MHz)?", state_text(), re.I)
    return int(m.group(1)) if m else None


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
              f"{fl} MHz", "blocks tasks 03, 05, 06")

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
    if p5s and fl:
        c.require(fl <= min(p5s), "F_LOCK at or below lowest observed floor",
                  f"F_LOCK {fl} <= min p5 {min(p5s)}",
                  f"F_LOCK {fl} EXCEEDS lowest floor {min(p5s)} — the GPU "
                  f"cannot hold this; every timing will drift")
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
        c.add(JUDGE if "PENDING" in text else PASS,
              "check D: T_SOL <= best measured", "needs task 06")
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


def check_06(c: Checks):
    tb = load_json(ART / "06" / "t_b.json")
    if not c.require(tb is not None, "t_b.json exists"):
        return
    entries = tb.get("problems", {})
    c.require(len(entries) >= EXPECTED_TOTAL - len(
        (load_json(ART / "deferred.json") or {}).get("problems", {})),
        f"t_b covers all {EXPECTED_TOTAL} problems", f"{len(entries)}",
        f"only {len(entries)} — every problem needs an anchor")
    c.require(all(v.get("winning_variant") for v in entries.values()),
              "winning PyTorch variant recorded per problem",
              detail_bad="'optimized PyTorch' is not reproducible; "
                         "a named variant is")
    anchor = ART / "06" / "anchor-verification.md"
    c.require(anchor.exists(), "anchor property verified",
              detail_bad="T_b must score 0.5+-0.03; reference must score <0.5")
    c.add(JUDGE, "authoritative pass ran under documented node conditions",
          "quiet vs busy, per task 01 interference verdict")


def check_07(c: Checks):
    spike = load_json(ART / "07" / "spike.json")
    c.require(spike is not None, "MXFP4 feasibility spike ran")
    if spike:
        c.require(spike.get("verdict") in ("go", "no-go"),
                  "spike has an explicit verdict", str(spike.get("verdict")))
    c.require((ART / "07" / "fp8-validation.md").exists(),
              "FP8 (18 problems) validation recorded")
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
    incomplete = [k for k, v in probs.items()
                  if not all(x in v for x in ("t_sol", "t_b", "tolerances"))]
    c.require(not incomplete, "every problem has t_sol, t_b, tolerances",
              f"{len(probs)} problems",
              f"{len(incomplete)} incomplete: {incomplete[:3]}")
    for field in ("f_lock_mhz", "methodology", "rocm_version", "torch_version"):
        c.require(field in m, f"manifest records {field}")
    if full:
        c.require((ART / "09" / "agent-baseline.json").exists(),
                  "agent baseline sweep ran")
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
