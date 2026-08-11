#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Corpus-wide census of the TOLERANCE side of the torch.compile failures.

Read-only. No GPU. Reads:

  artifacts/05/workloads/<Cat>/<prob>/workload.jsonl   AMD-derived tolerances
  artifacts/05/<Cat>__<prob>.json                      run-to-run + vs_golden
  artifacts/06/{candidates,authoritative}/*.json       per-variant pass/fail
  data/SOL-ExecBench/benchmark/<Cat>/<prob>/           upstream tolerances, dtypes
  leaderboard/solbench.db                              scoreable set (workload table)

Run:  python artifacts/11/compile-diag/tolerance_census.py

Ground truth for pass/fail is artifacts/06, NOT leaderboard/solbench.db -- see
the `board_check` section, which shows the shipped db has submission 2's
per-workload statuses inverted.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FP32EPS = 1.1920928955078125e-07
EPS = {"float32": FP32EPS, "bfloat16": 0.0078125,
       "float16": 0.0009765625, "float64": 2.220446049250313e-16}
# src/sol_execbench/core/data/workload.py ToleranceSpec defaults, used where
# upstream shipped no tolerance block (all of FlashInfer-Bench) or no ratio.
DEF_ATOL, DEF_RTOL, DEF_MR = 1e-2, 1e-2, 0.99
CATS = ["L1", "L2", "Quant", "FlashInfer-Bench"]


def load_amd():
    out = {}
    for cat in CATS:
        cdir = ROOT / "artifacts/05/workloads" / cat
        for prob in sorted(os.listdir(cdir)):
            p = cdir / prob / "workload.jsonl"
            if not p.exists():
                continue
            key = f"{cat}__{prob}"
            for i, line in enumerate(x for x in p.read_text().splitlines() if x.strip()):
                w = json.loads(line)
                t = w["tolerance"]
                out[(key, w["uuid"])] = dict(
                    cat=cat, prob=key, order=i, atol=t["max_atol"],
                    rtol=t["max_rtol"], mr=t.get("required_matched_ratio"),
                    deriv=t.get("_provenance", ""))
    return out


def load_upstream():
    out = {}
    for cat in CATS:
        cdir = ROOT / "data/SOL-ExecBench/benchmark" / cat
        for prob in sorted(os.listdir(cdir)):
            p = cdir / prob / "workload.jsonl"
            if not p.exists():
                continue
            key = f"{cat}__{prob}"
            for line in (x for x in p.read_text().splitlines() if x.strip()):
                w = json.loads(line)
                t = w.get("tolerance") or {}
                out[(key, w["uuid"])] = (
                    t.get("max_atol", DEF_ATOL) if t.get("max_atol") is not None else DEF_ATOL,
                    t.get("max_rtol", DEF_RTOL) if t.get("max_rtol") is not None else DEF_RTOL,
                    (t.get("required_match_ratio")
                     or t.get("required_matched_ratio") or DEF_MR))
    return out


def load_dtypes():
    out = {}
    for cat in CATS:
        cdir = ROOT / "data/SOL-ExecBench/benchmark" / cat
        for prob in sorted(os.listdir(cdir)):
            p = cdir / prob / "definition.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            out[f"{cat}__{prob}"] = tuple(
                s.get("dtype") for s in (d.get("outputs") or {}).values()
                if isinstance(s, dict))
    return out


def load_r2r():
    out = {}
    for f in sorted(glob.glob(str(ROOT / "artifacts/05/*.json"))):
        d = json.loads(Path(f).read_text())
        key = d.get("problem")
        for w in d.get("per_workload", []):
            if w.get("ok"):
                out[(key, w["workload_uuid"])] = dict(
                    max_abs=w["run_to_run"]["max_abs"],
                    max_rel=w["run_to_run"]["max_rel"],
                    golden=w.get("vs_golden"))
    return out


def load_scoreable():
    db = sqlite3.connect(ROOT / "leaderboard/solbench.db")
    return {(a, b) for a, b in db.execute(
        "select problem_key, uuid from workload where scoreable=1")}, db


def variant_truth(vname, scoreable):
    """Pass/fail per workload, from the task 06 artifacts.

    `latency_ms_by_workload` holds exactly the workloads that PASSED; the ones
    that did not are in `failures`. authoritative/ overrides candidates/.
    """
    F, P = set(), set()
    for src in ("artifacts/06/candidates", "artifacts/06/authoritative"):
        for f in sorted(glob.glob(str(ROOT / src / "*.json"))):
            d = json.loads(Path(f).read_text())
            p = d.get("problem") or Path(f).stem
            v = (d.get("variants") or {}).get(vname)
            if not v:
                continue
            for fl in v.get("failures") or []:
                u = fl["workload_uuid"]
                F.add((p, u)); P.discard((p, u))
            for u in (v.get("latency_ms_by_workload") or {}):
                P.add((p, u)); F.discard((p, u))
    return F & scoreable, P & scoreable


def qs(v):
    v = sorted(v)
    g = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return [v[0], g(.10), g(.25), g(.50), g(.75), g(.90), g(.99), v[-1]]


def two_by_two(pred, label, keys, verdict):
    a = b = c = d = 0
    for k in keys:
        s = verdict.get(k)
        if s is None:
            continue
        if pred(k):
            a += s == "FAILED"; b += s == "PASSED"
        else:
            c += s == "FAILED"; d += s == "PASSED"
    n = a + b + c + d
    orr = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    chi = (n * (a * d - b * c) ** 2 /
           ((a + b) * (c + d) * (a + c) * (b + d))) if min(a+b, c+d, a+c, b+d) else float("nan")
    print(f"\n### {label}   [n={n}]")
    print("| | FAILED | PASSED | total | fail rate |")
    print("|---|---|---|---|---|")
    print(f"| yes | {a} | {b} | {a+b} | {100*a/max(1,a+b):.1f}% |")
    print(f"| no  | {c} | {d} | {c+d} | {100*c/max(1,c+d):.1f}% |")
    print(f"odds ratio (Haldane) {orr:.2f}   chi2(1) {chi:.1f}")
    return a, b, c, d


def main() -> int:
    amd = load_amd(); ups = load_upstream(); dtypes = load_dtypes()
    r2r = load_r2r(); scoreable, db = load_scoreable()
    S = {k: v for k, v in amd.items() if k in scoreable}
    dt0 = lambda p: (dtypes.get(p) or ("none",))[0]

    print("# 0. corpus")
    print(f"AMD tolerance rows: {len(amd)} over {len({k[0] for k in amd})} problems")
    print(f"scoreable (manifest v1.2 / workload table): {len(S)} over "
          f"{len({k[0] for k in S})} problems")

    print("\n# board_check: leaderboard/solbench.db vs artifacts/06")
    F2, P2 = variant_truth("v2_compile", scoreable)
    dbf = {(a, b) for a, b in db.execute(
        "select problem_key, workload_uuid from result "
        "where submission_id=2 and status='FAILED'")}
    print(f"db submission 2 FAILED rows: {len(dbf)}")
    print(f"artifacts/06 v2_compile FAILED: {len(F2)}  PASSED: {len(P2)}")
    print(f"db FAILED that artifacts call FAILED: {len(dbf & F2)}")
    print(f"db FAILED that artifacts call PASSED: {len(dbf & P2)}")
    print(f"artifact failures with no db row at all: {len(F2 - dbf - {(a,b) for a,b in db.execute(chr(115)+'elect problem_key, workload_uuid from result where submission_id=2')})}")

    V = {k: "FAILED" for k in F2}
    V.update({k: "PASSED" for k in P2})

    print("\n\n# 2. tolerance distribution over all scoreable workloads")
    rows = list(S.values())
    n = len(rows)
    print(f"rtol == fp32 eps exactly: {sum(1 for r in rows if r['rtol']==FP32EPS)}"
          f" / {n} = {100*sum(1 for r in rows if r['rtol']==FP32EPS)/n:.1f}%")
    print("floor actually applied (from _provenance):",
          dict(collections.Counter(r["deriv"].split("floored at ")[-1] for r in rows)))
    print("\n| category | workloads | rtol==fp32eps | pct |")
    print("|---|---|---|---|")
    for cat in CATS:
        sub = [r for r in rows if r["cat"] == cat]
        f = sum(1 for r in sub if r["rtol"] == FP32EPS)
        print(f"| {cat} | {len(sub)} | {f} | {100*f/len(sub):.1f}% |")
    print("\n| output dtype | workloads | rtol==fp32eps | rtol==eps(dtype) | median atol |")
    print("|---|---|---|---|---|")
    bd = collections.defaultdict(list)
    for k, r in S.items():
        bd[dt0(k[0])].append(r)
    for dt, sub in sorted(bd.items(), key=lambda kv: -len(kv[1])):
        a = sorted(r["atol"] for r in sub)
        print(f"| {dt} | {len(sub)} | {sum(1 for r in sub if r['rtol']==FP32EPS)} | "
              f"{sum(1 for r in sub if r['rtol']==EPS.get(dt,-1))} | {a[len(a)//2]:.3e} |")
    print("\natol min/p10/p25/med/p75/p90/p99/max:",
          " ".join(f"{x:.3e}" for x in qs([r["atol"] for r in rows])))
    print("rtol min/p10/p25/med/p75/p90/p99/max:",
          " ".join(f"{x:.3e}" for x in qs([r["rtol"] for r in rows])))
    print("atol == 0.0 (integer/bool outputs):", sum(1 for r in rows if r["atol"] == 0.0))
    print("required_matched_ratio values:",
          dict(collections.Counter(r["mr"] for r in rows)))

    print("\n\n# 3. correlation with the torch.compile (v2) failures")
    order = {k: S[k]["order"] for k in S}
    print("failing workloads by file index within the problem:",
          dict(collections.Counter("idx<8" if order[k] < 8 else "idx>=8" for k in F2)))
    lo = [k for k in S if V.get(k) and order[k] < 8]
    hi = [k for k in S if V.get(k) and order[k] >= 8]
    print(f"fail rate, first 8 workloads of each problem: "
          f"{sum(1 for k in lo if V[k]=='FAILED')}/{len(lo)}")
    print(f"fail rate, workloads 9+:                      "
          f"{sum(1 for k in hi if V[k]=='FAILED')}/{len(hi)}")

    allk = [k for k in S if V.get(k)]
    for keys, tag in ((allk, "ALL scoreable"), (lo, "first-8 only (actually compiled)")):
        print(f"\n-- population: {tag} --")
        two_by_two(lambda k: S[k]["rtol"] == FP32EPS, "rtol == fp32 eps", keys, V)
        two_by_two(lambda k: dt0(k[0]) == "float32", "output dtype float32", keys, V)
        two_by_two(lambda k: r2r[k]["max_abs"] == 0.0,
                   "run-to-run variance measured exactly 0", keys, V)

    byp = collections.defaultdict(list)
    for k in S:
        byp[k[0]].append(k)
    probs = sorted(byp)
    pf = {p: sum(1 for k in byp[p] if V.get(k) == "FAILED") for p in probs}
    a = sum(1 for p in probs if dt0(p) == "float32" and pf[p] > 0)
    b = sum(1 for p in probs if dt0(p) == "float32" and pf[p] == 0)
    c = sum(1 for p in probs if dt0(p) != "float32" and pf[p] > 0)
    d = sum(1 for p in probs if dt0(p) != "float32" and pf[p] == 0)
    print(f"\nproblem-level 2x2 (fp32 output x fails anywhere): "
          f"{a} {b} / {c} {d}  OR={((a+.5)*(d+.5))/((b+.5)*(c+.5)):.2f}")

    print("\n### CELL C: NOT fp32-floored yet v2 fails")
    for p in sorted((p for p in probs if pf[p] and dt0(p) != "float32"),
                    key=lambda p: -pf[p]):
        ats = sorted(S[k]["atol"] for k in byp[p])
        print(f"- {p} | {dt0(p)} | {pf[p]}/{len(byp[p])} | atol {ats[0]:.2e}..{ats[-1]:.2e}"
              f" | rtol {S[byp[p][0]]['rtol']:.3e}")
    print("\n### CELL B: fp32-floored yet v2 fully passes")
    for p in sorted(p for p in probs if dt0(p) == "float32" and pf[p] == 0):
        ats = sorted(S[k]["atol"] for k in byp[p])
        print(f"- {p} | {len(byp[p])} wl | atol {ats[0]:.2e}..{ats[-1]:.2e}")

    print("\n\n# 4. is the eager reference bit-identical run to run?")
    det = sum(1 for k in S if r2r[k]["max_abs"] == 0.0)
    print(f"run-to-run max_abs == 0.0 exactly: {det}/{len(S)} = {100*det/len(S):.1f}%")
    for cat in CATS:
        ks = [k for k in S if S[k]["cat"] == cat]
        print(f"  {cat}: {sum(1 for k in ks if r2r[k]['max_abs']==0.0)}/{len(ks)}")
    nz = [r2r[k]["max_abs"] for k in S if r2r[k]["max_abs"] != 0.0]
    print("nonzero run-to-run max_abs: n=%d min=%.3e median=%.3e max=%.3e"
          % (len(nz), min(nz), sorted(nz)[len(nz)//2], max(nz)))
    F1, P1 = variant_truth("v1_eager", scoreable)
    print(f"v1_eager (the SAME reference, re-run in a fresh process) fails "
          f"{len(F1)} workloads: {sorted({p for p,_ in F1})}")
    for k in sorted(F1):
        print(f"   {k[1][:8]} run_to_run max_abs={r2r[k]['max_abs']:.3e} "
              f"derived atol={S[k]['atol']:.4e}")
    g = [k for k in S if r2r[k]["golden"]]
    print(f"\nfloat64/native CPU golden recorded for {len(g)}/{len(S)} workloads "
          f"({len({k[0] for k in g})} problems); modes "
          f"{dict(collections.Counter(r2r[k]['golden'].get('mode') for k in g))}")
    print("golden max_abs min/p10/p25/med/p75/p90/p99/max:",
          " ".join(f"{x:.3g}" for x in qs([r2r[k]["golden"]["max_abs"] for k in g])))
    print("golden max_abs exceeding the derived atol:",
          sum(1 for k in g if r2r[k]["golden"]["max_abs"] > S[k]["atol"]), "of", len(g))

    print("\n\n# 5. upstream (B200) tolerances, as a comparison only")
    print("upstream tolerance block shape, by category:")
    for cat in CATS:
        cdir = ROOT / "data/SOL-ExecBench/benchmark" / cat
        kk = collections.Counter()
        for prob in sorted(os.listdir(cdir)):
            p = cdir / prob / "workload.jsonl"
            if not p.exists():
                continue
            for line in (x for x in p.read_text().splitlines() if x.strip()):
                t = json.loads(line).get("tolerance")
                kk[tuple(sorted(t)) if isinstance(t, dict) else "ABSENT"] += 1
        print(f"  {cat}: {dict(kk)}")
    print("\nexact numeric matches AMD == upstream (prime directive 2):",
          sum(1 for k in S if k in ups
              and S[k]["atol"] == ups[k][0] and S[k]["rtol"] == ups[k][1]))
    print("\n| |y| | rows | min | p10 | p25 | med | p75 | p90 | p99 | max | upstream looser |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for Y in (0.0, 1.0, 100.0):
        for keys, tag in ((list(S), "all"), (sorted(F2), "v2-FAILED only")):
            rs = [(ups[k][0] + ups[k][1]*Y) / (S[k]["atol"] + S[k]["rtol"]*Y)
                  for k in keys if k in ups and S[k]["atol"] + S[k]["rtol"]*Y > 0]
            print(f"| {Y:g} ({tag}) | {len(rs)} | " + " | ".join(f"{x:.3g}" for x in qs(rs))
                  + f" | {100*sum(1 for x in rs if x>1)/len(rs):.1f}% |")
    print("\n| output dtype | n | median atol ratio up/AMD | median rtol ratio | median AMD atol | median upstream atol |")
    print("|---|---|---|---|---|---|")
    for dt in ["float32", "bfloat16", "float16", "int64", "int32", "bool"]:
        ks = [k for k in S if dt0(k[0]) == dt and k in ups and S[k]["atol"] > 0]
        if not ks:
            ks0 = [k for k in S if dt0(k[0]) == dt and k in ups]
            if ks0:
                mu = sorted(ups[k][0] for k in ks0)
                print(f"| {dt} | {len(ks0)} | n/a (AMD atol=0) | n/a (AMD rtol=0) | 0.0 | {mu[len(mu)//2]:.3e} |")
            continue
        ra = sorted(ups[k][0]/S[k]["atol"] for k in ks)
        rr = sorted(ups[k][1]/S[k]["rtol"] for k in ks if S[k]["rtol"] > 0)
        ma = sorted(S[k]["atol"] for k in ks); mu = sorted(ups[k][0] for k in ks)
        print(f"| {dt} | {len(ks)} | {ra[len(ra)//2]:.4g} | {rr[len(rr)//2]:.4g} | "
              f"{ma[len(ma)//2]:.3e} | {mu[len(mu)//2]:.3e} |")
    print("\nrequired matched ratio, upstream -> AMD:",
          dict(collections.Counter((ups[k][2], S[k]["mr"]) for k in S if k in ups)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
