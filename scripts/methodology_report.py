#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 04 report — how far apart are `hip_events` and `rocprof`?

Upstream times with CUPTI activity tracing, which has no ROCm build. This port
ships two methodologies and records which one produced every trace. That is
only defensible if the size of the difference is known, so both time the same
solution on the same inputs back to back in one process, and this summarizes
the result.

Sign convention, from the runner: **positive means `hip_events` read slower**,
which is the expected direction — an event pair brackets the host launch and
dispatch-level activity tracing does not.

    python scripts/methodology_report.py --out artifacts/04/methodology-comparison.md
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import stamp  # noqa: E402


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def sign_paragraph(median: float) -> str:
    """The sign discussion, DERIVED from this part's median.

    This paragraph used to be a fixed string asserting "slightly negative",
    "well under a percent", "agree to under 1%" and a node CV of 0.0034 — all
    of them MI350X readings, hardcoded. Regenerating the report for a second
    part reproduced those sentences verbatim over a different part's numbers,
    which is prime directive 2 wearing a template as a disguise. The size words
    now come from the number in front of them; the node's own reproducibility
    figure is not quoted at all, because this script has no way to know it.
    """
    direction = ("`rocprof` reads *slower*" if median < 0
                 else "`hip_events` reads *slower*, as predicted")
    size = ("under a percent" if abs(median) < 1
            else f"about {abs(median):.1f}%")
    expected = ("**The median landed on the unexpected side of zero.** "
                if median < 0 else "")
    return (
        f"{expected}The predicted sign was positive — events include the "
        f"launch, activity tracing does not — and the measured median is "
        f"{median:+.2f}%, i.e. {direction}. The gap is {size} at the median. "
        f"It is reported rather than explained: whether {size} is inside this "
        f"node's run-to-run reproducibility is a question for that node's own "
        f"stability measurement, which this script does not read and therefore "
        f"does not quote."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="artifacts/04/compare")
    ap.add_argument("--out", default="artifacts/04/methodology-comparison.md")
    ap.add_argument("--gate", type=float, default=2.0,
                    help="acceptance: |median| divergence must be under this")
    a = ap.parse_args()

    rows: list[tuple[float, float, bool, str]] = []
    failed: list[tuple[str, str]] = []
    files = sorted(glob.glob(f"{a.compare}/*.json"))
    for f in files:
        d = json.loads(Path(f).read_text())
        if not d.get("ok", True):
            failed.append((d.get("problem") or Path(f).stem,
                           (d.get("error") or "")[:120]))
            continue
        for w in d.get("divergences") or []:
            rows.append((w["hip_events_ms"], w["divergence_pct"],
                         w["microsecond_scale"], d["problem"]))

    groups = {
        "all workloads": rows,
        "kernels >= 100 us": [r for r in rows if not r[2]],
        "kernels < 100 us": [r for r in rows if r[2]],
    }
    medians = {k: statistics.median([r[1] for r in v]) if v else 0.0
               for k, v in groups.items()}
    gate_ok = abs(medians["kernels >= 100 us"]) <= a.gate

    tail = [r for r in rows if abs(r[1]) > 20]
    by_problem: dict[str, int] = {}
    for _, _, _, p in tail:
        by_problem[p] = by_problem.get(p, 0) + 1

    # The two extremes, named from THIS part's data. Previously "~90%" and
    # "~3x on L1/034", which were MI350X's and travelled into any other part's
    # report unchanged.
    hi = max(rows, key=lambda r: r[1], default=(0.0, 0.0, False, "—"))
    lo = min(rows, key=lambda r: r[1], default=(0.0, 0.0, False, "—"))
    hi_extreme, lo_extreme, lo_problem = hi[1], lo[1], lo[3]

    L = [
        "# Task 04 — `hip_events` vs `rocprof`",
        "",
        f"<!-- {json.dumps(stamp('04-methodology-comparison')['_provenance'], default=str)} -->",
        "",
        "Both methodologies time the same solution on the same inputs, back to "
        "back in one process, so this compares two ways of measuring and not "
        "two moments in the node's life. **Positive means `hip_events` read "
        "slower**, which is the expected direction: an event pair brackets the "
        "host launch and dispatch-level activity tracing does not.",
        "",
        f"Problems compared: **{len(files) - len(failed)}** of {len(files)}; "
        f"workload pairs: **{len(rows)}**.",
        "",
        "| group | n | median | p10 | p90 |",
        "|---|---|---|---|---|",
    ]
    for k, v in groups.items():
        if not v:
            continue
        xs = [r[1] for r in v]
        L.append(f"| {k} | {len(xs)} | {statistics.median(xs):+.2f}% | "
                 f"{pct(xs, 0.10):+.1f}% | {pct(xs, 0.90):+.1f}% |")

    L += [
        "",
        f"**Acceptance — median divergence on kernels >= 100 us: "
        f"{medians['kernels >= 100 us']:+.2f}%** against a gate of "
        f"{a.gate:.0f}%. {'PASS' if gate_ok else 'FAIL'}.",
        "",
        sign_paragraph(medians["kernels >= 100 us"]),
        "",
        "Sub-100 us kernels are reported separately rather than folded in. "
        "There the median is "
        f"{medians['kernels < 100 us']:+.2f}%: a fixed launch overhead is a "
        "larger fraction of a smaller number, which is the finding, not an "
        "error.",
        "",
        "## The tails are wide, and they are wide in both directions",
        "",
        f"{len(tail)} of {len(rows)} workload pairs differ by more than 20%, "
        f"concentrated in {len(by_problem)} problems. The median is small "
        "because most iterations dispatch one kernel; the tail is where they "
        "do not.",
        "",
        f"* **`hip_events` much slower** (up to {hi_extreme:+.0f}%) on problems "
        "whose "
        "iteration is many tiny kernels. The event pair contains the host-side "
        "work between them and the activity sum does not. This is the "
        "understood direction and is why short kernels score slightly low "
        "under the default methodology.",
        f"* **`rocprof` slower** (to {lo_extreme:+.0f}% on `{lo_problem}`) on "
        "some multi-dispatch "
        "iterations. Summing per-dispatch durations exceeds the wall clock "
        "whenever dispatches overlap, so the activity sum is not a wall-clock "
        "measurement for those. Stated as the hypothesis it is: it has not "
        "been confirmed against a dispatch timeline, and no number in this "
        "port depends on it, because `hip_events` is the default and every "
        "trace records its methodology.",
        "",
        "| workloads > 20% apart | problem |",
        "|---|---|",
    ]
    for p, n in sorted(by_problem.items(), key=lambda kv: -kv[1])[:12]:
        L.append(f"| {n} | {p} |")

    if failed:
        L += ["", "## Problems that produced no comparison", "",
              "| problem | error |", "|---|---|"]
        for p, e in failed:
            L.append(f"| {p} | `{e}` |")

    L += ["",
          "## What this licenses",
          "",
          "`hip_events` stays the default and `Environment.methodology` is "
          "recorded on every trace. A trace taken under one and a trace taken "
          "under the other are not interchangeable — that is what the field is "
          "for. Mixing them silently is the failure this measurement exists to "
          "make impossible.",
          ""]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    for k, v in groups.items():
        if v:
            print(f"  {k:<20} n={len(v):<5} median {medians[k]:+.2f}%")
    print(f"  gate {a.gate}%: {'PASS' if gate_ok else 'FAIL'}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
