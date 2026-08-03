#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Materialize the HuggingFace parquet distribution into the per-problem layout.

`reference/upstream-audit.md` expects each problem as a directory containing
`definition.json`, `workload.jsonl` and `reference.py`. The Hub actually ships
`data/<Category>.parquet` — one row per problem — because that is what the
dataset viewer needs. The dataset repo's own `scripts/convert_to_parquet.py`
does the forward direction; this is its exact inverse, field for field.

    python scripts/materialize_dataset.py \
        --parquet-dir /var/tmp/solbench/data-hf/data \
        --out data/SOL-ExecBench/benchmark

Idempotent: re-running overwrites in place. Verifies the round-trip by
re-deriving each parquet row from the files just written and comparing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CATEGORIES = ("L1", "L2", "Quant", "FlashInfer-Bench")
EXPECTED = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}

# Exactly the fields convert_to_parquet.py exports, minus the two that become
# their own files (reference -> reference.py, workloads -> workload.jsonl).
DEFINITION_SCALAR = ("name", "description", "hf_id", "custom_inputs_entrypoint")
DEFINITION_JSON = ("axes", "inputs", "outputs")


def _scalar(v):
    """Parquet renders an absent optional string as NaN or ""; JSON wants null.

    Two ways parquet spells "no value", and both must become null:

    * NaN — writing it through emits a bare `nan` token, which is invalid JSON
      that only Python's lenient parser accepts. 72 of the 94 L1 problems have
      no custom entrypoint, so this is the common case, not an edge one.
    * "" — the empty string is a *value*, and the schema types these fields as
      NonEmptyString. All 26 FlashInfer-Bench problems carry `hf_id: ""`, so
      writing it through made every one of them fail to load with
      `String should have at least 1 character` — 26 problems, 11% of the
      benchmark, lost to a distinction between "empty" and "absent".
    """
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v


def materialize(parquet_dir: Path, out: Path) -> dict:
    import pandas as pd

    census: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        src = parquet_dir / f"{cat}.parquet"
        if not src.exists():
            sys.exit(f"missing {src}")
        df = pd.read_parquet(src)
        names = []
        for _, row in df.iterrows():
            name = row["name"]
            d = out / cat / name
            d.mkdir(parents=True, exist_ok=True)

            definition = {k: _scalar(row[k]) for k in DEFINITION_SCALAR}
            # axes/inputs/outputs are JSON-encoded strings in parquet; restore
            # them to real objects so definition.json matches upstream.
            for k in DEFINITION_JSON:
                definition[k] = json.loads(row[k])
            # The reference code goes in BOTH places, on purpose:
            #   definition.json["reference"] -- because `Definition` declares it
            #       as a required field, so a definition.json without it cannot
            #       be loaded by the harness, upstream's `run_dataset.py`, or
            #       the CLI. The audit's file layout described the directory
            #       contents, not a different schema.
            #   reference.py -- because a 200-line kernel embedded in a JSON
            #       string is unreadable and ungreppable, and every sweep
            #       triage starts by reading it.
            # verify_roundtrip checks the two agree, so they cannot drift.
            definition["reference"] = row["reference"]
            (d / "definition.json").write_text(
                json.dumps(definition, indent=2) + "\n")

            (d / "reference.py").write_text(row["reference"])

            workloads = json.loads(row["workloads"])
            with (d / "workload.jsonl").open("w") as f:
                for w in workloads:
                    f.write(json.dumps(w) + "\n")
            names.append(name)
        census[cat] = sorted(names)
    return census


def verify_roundtrip(parquet_dir: Path, out: Path) -> list[str]:
    """Re-derive each parquet row from the written files; report mismatches."""
    import pandas as pd

    problems: list[str] = []
    for cat in CATEGORIES:
        df = pd.read_parquet(parquet_dir / f"{cat}.parquet")
        for _, row in df.iterrows():
            d = out / cat / row["name"]
            definition = json.loads((d / "definition.json").read_text())
            for k in DEFINITION_SCALAR:
                if definition[k] != _scalar(row[k]):
                    problems.append(f"{cat}/{row['name']}: field {k} differs")
            for k in DEFINITION_JSON:
                if definition[k] != json.loads(row[k]):
                    problems.append(f"{cat}/{row['name']}: field {k} differs")
            if (d / "reference.py").read_text() != row["reference"]:
                problems.append(f"{cat}/{row['name']}: reference.py differs")
            if definition.get("reference") != row["reference"]:
                problems.append(
                    f"{cat}/{row['name']}: definition.json reference differs")
            got = [json.loads(x) for x in
                   (d / "workload.jsonl").read_text().splitlines() if x.strip()]
            if got != json.loads(row["workloads"]):
                problems.append(f"{cat}/{row['name']}: workloads differ")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", default="data/SOL-ExecBench/benchmark")
    a = ap.parse_args()

    out = Path(a.out)
    census = materialize(Path(a.parquet_dir), out)

    print(f"\nMaterialized -> {out}\n")
    total = 0
    mismatch = False
    for cat in CATEGORIES:
        n = len(census[cat])
        total += n
        flag = "" if n == EXPECTED[cat] else f"  <-- expected {EXPECTED[cat]}"
        if n != EXPECTED[cat]:
            mismatch = True
        print(f"  {cat:<18} {n:>3} problems{flag}")
    print(f"\n  TOTAL              {total}")

    bad = verify_roundtrip(Path(a.parquet_dir), out)
    if bad:
        print(f"\n  ROUND-TRIP FAILED — {len(bad)} mismatches:")
        for b in bad[:10]:
            print(f"    {b}")
        sys.exit(1)
    print("  round-trip verified: files re-derive the parquet rows exactly")
    sys.exit(1 if mismatch else 0)


if __name__ == "__main__":
    main()
