#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 04, ROCM CONTRACT #1 — prove the shim's two clocks are one clock.

rocprofiler-sdk stamps its records with the HSA clock. The host bracket that
decides which records belong to which iteration must come from rocprofiler's
own timestamp entry point, not `CLOCK_MONOTONIC`. Mixing the two does not
raise: it silently bisects the wrong activities into each window, and every
number downstream is wrong by an amount nobody can see.

So it is checked against a real capture, and the evidence is written to an
artifact rather than to a terminal that scrolls away.

    python scripts/verify_clock_domain.py --out artifacts/04/clock-domain-verification.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "solexbench_rocm" / "activity"))
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/04/clock-domain-verification.log")
    ap.add_argument("--iterations", type=int, default=8)
    a = ap.parse_args()

    # The shim must be configured BEFORE any ROCm runtime initializes:
    # rocprofiler locks its configuration once one does, and a session
    # configured afterwards produces zero records rather than an error. That
    # is why torch is imported below this line and not at the top.
    sys.path.insert(0, str(ROOT / "src" / "solexbench_rocm" / "shim"))
    import _rocprof_shim                                     # noqa: E402

    from activity_sources import verify_clock_domain         # noqa: E402
    from activity_sources import RocprofActivitySource       # noqa: E402

    if hasattr(_rocprof_shim, "configure"):
        _rocprof_shim.configure()

    import torch                                             # noqa: E402

    source = RocprofActivitySource()
    lines: list[str] = []

    def say(s: str) -> None:
        print(s)
        lines.append(s)

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.matmul(x, y)
    torch.cuda.synchronize()

    windows: list[tuple[int, int]] = []
    source.__enter__()
    for _ in range(a.iterations):
        t0 = source.timestamp()
        torch.matmul(x, y)
        torch.cuda.synchronize()
        windows.append((t0, source.timestamp()))
    activities = source.drain()
    source.__exit__()

    say(f"iterations       {a.iterations}")
    say(f"host windows     {len(windows)}")
    say(f"activities       {len(activities)}")
    if not activities:
        say("FAIL: no activities recorded — the shim registered too late, or "
            "a ROCm runtime initialized before rocprofiler_force_configure")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")
        return 1

    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    inside = [x for x in activities if x.start >= lo and x.end <= hi]
    say(f"host span        [{lo}, {hi}]  ({(hi - lo) / 1e6:.3f} ms)")
    say(f"inside the span  {len(inside)}/{len(activities)}")

    try:
        verify_clock_domain(activities, windows)
        say("verify_clock_domain: PASS — record stamps and host bracket are "
            "in the same domain (ROCM CONTRACT #1)")
        rc = 0
    except RuntimeError as e:
        say(f"verify_clock_domain: FAIL — {e}")
        rc = 1

    # How far apart the two clocks actually are on this driver. This is the
    # measurement, not an assumption: if it is ~0 then CLOCK_MONOTONIC and the
    # HSA clock happen to coincide here and the contract's hazard does not
    # bite on this node -- which is worth knowing, and is NOT a reason to
    # remove the guard, since it is a property of the driver and not of the
    # code.
    import dataclasses
    import time

    offset = time.monotonic_ns() - source.timestamp()
    say(f"monotonic - HSA  {offset} ns"
        + ("   (the two domains coincide on this driver)"
           if abs(offset) < 1_000_000 else ""))

    # The negative control. A guard that cannot fail is proving nothing, and
    # that is the failure mode a check like this is most prone to. Shift by a
    # full span so the control is meaningful even when the real offset is 0.
    shift = max(1_000_000, (hi - lo) * 10)
    skewed = [dataclasses.replace(x, start=x.start + shift, end=x.end + shift)
              for x in activities]
    try:
        verify_clock_domain(skewed, windows)
        say(f"negative control: FAIL — records shifted by {shift} ns still "
            f"passed, so the check does not discriminate")
        rc = 1
    except RuntimeError:
        say(f"negative control: PASS — the same records shifted by "
            f"{shift} ns are rejected")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
