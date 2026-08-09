#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record which model actually answered a run's calls, not which one it asked for.

    python scripts/stamp_upstream_model.py --run artifacts/10/gpt56-40 \
        --collected <sbt-state>/runs/gpt56-40/collected.json

A run's `model` is the name the job asked the front door for. That name is an
alias: the fleet's registry resolves `GLM-5.2` to an upstream called
`GLM-5.2-FP8`, so a board row labelled `GLM-5.2` is naming the request and not
the weights. Both are true and they are different facts, and the one a reader
of a benchmark wants is the second.

The upstream name is on every call record the front door wrote. This reads it
from there rather than from a table in this repo, for the same reason
`sbt.collect.observed_model` reads the model from the daemons' records: a
mapping hardcoded here would be a fact about somebody else's deployment,
correct until they change it and silently wrong afterwards.

**Never guesses.** A run whose calls disagree about the upstream is stamped
with the whole list and no single name, and a run with no call records is left
alone rather than given its alias back under a field that promises evidence.

Vendor prefixes are stripped for the comparison only: `zai-org/GLM-5.2-FP8` and
`GLM-5.2-FP8` are two providers of one model, and treating that as a
disagreement would refuse a run for having a second backend, which is a fact
about routing rather than about what ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def upstream_names(collected: dict) -> Counter:
    """{upstream model name -> how many calls it answered}, over one sbt run."""
    seen: Counter = Counter()
    for problem in collected.get("problems") or []:
        for call in problem.get("trajectory") or []:
            name = (call.get("upstream_model") or "").strip()
            if name:
                seen[name] += 1
    return seen


def resolve(seen: Counter) -> tuple[str | None, str]:
    """The single upstream model, or None and why there isn't one."""
    if not seen:
        return None, "no call record carried an upstream_model"
    families = {name.rsplit("/", 1)[-1] for name in seen}
    if len(families) > 1:
        return None, ("the calls were answered by more than one model: "
                      + ", ".join(f"{k} x{v}" for k, v in sorted(seen.items())))
    # One model, possibly via several providers. Name it without the provider,
    # since the provider is not a property of the model that ran.
    return families.pop(), ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path,
                    help="an artifacts/10/<run-id> directory holding run.json")
    ap.add_argument("--collected", required=True, nargs="+", type=Path,
                    help="the sbt collected.json files backing it. More than "
                         "one when the run was merged from several sweeps -- "
                         "all of them, or the evidence is partial.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    run_path = a.run / "run.json"
    run = json.loads(run_path.read_text())

    seen: Counter = Counter()
    for path in a.collected:
        seen += upstream_names(json.loads(path.read_text()))

    name, why = resolve(seen)
    if name is None:
        print(f"{a.run.name}: not stamped -- {why}")
        if seen:
            run["upstream_model"] = None
            run["upstream_model_calls"] = dict(seen)
        else:
            return 0
    else:
        asked = run.get("model")
        run["upstream_model"] = name
        run["upstream_model_calls"] = dict(seen)
        note = "same name" if name == asked else f"asked for {asked!r}"
        print(f"{a.run.name}: upstream {name!r} over {sum(seen.values())} "
              f"calls ({note})")

    if a.dry_run:
        return 0
    run_path.write_text(json.dumps(run, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
