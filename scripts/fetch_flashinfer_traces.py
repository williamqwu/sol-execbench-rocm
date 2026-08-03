#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the external safetensors blobs the FlashInfer-Bench workloads reference.

9 of the 26 FlashInfer-Bench problems declare inputs of ``type: "safetensors"``
pointing at ``data/flashinfer-trace/blob/...`` -- a *separate* HuggingFace
dataset (``flashinfer-ai/flashinfer-trace``) that the SOL-ExecBench dataset does
not carry. Without it those problems fail at run time with
``Failed to load safetensors``, which would silently cost 9 problems of the 235.
L1, L2 and Quant are fully self-contained; only FlashInfer-Bench needs this.

Downloads only the blobs actually referenced (304 of several thousand) rather
than the whole dataset.

    python scripts/fetch_flashinfer_traces.py --data-root data

Idempotent: already-present files are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "flashinfer-ai/flashinfer-trace"
PREFIX = "data/flashinfer-trace/"


def referenced_blobs(bench_dir: Path) -> set[str]:
    """Every safetensors path referenced by any workload, as written."""
    paths: set[str] = set()
    for wl in bench_dir.rglob("workload.jsonl"):
        for line in wl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            for spec in (w.get("inputs") or {}).values():
                if isinstance(spec, dict) and spec.get("type") == "safetensors":
                    paths.add(spec["path"])
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data",
                    help="directory the workload paths are relative to")
    ap.add_argument("--benchmark-dir", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download

    blobs = referenced_blobs(Path(a.benchmark_dir))
    if not blobs:
        sys.exit("no safetensors references found — wrong --benchmark-dir?")

    # Workload paths are "data/flashinfer-trace/blob/..."; inside the HF repo
    # the same file is at "blob/...".
    root = Path(a.data_root).parent if Path(a.data_root).name == "data" else Path(".")
    todo = []
    for p in sorted(blobs):
        if not p.startswith(PREFIX):
            sys.exit(f"unexpected safetensors path layout: {p}")
        todo.append((p[len(PREFIX):], root / p))

    present = sum(1 for _, dest in todo if dest.exists())
    print(f"{len(todo)} referenced blobs, {present} already present")

    failures: list[tuple[str, str]] = []

    def fetch(repo_rel: str, dest: Path):
        if dest.exists():
            return
        got = hf_hub_download(REPO, repo_rel, repo_type="dataset")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(got).read_bytes())

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, r, d): r for r, d in todo}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:
                failures.append((futs[f], f"{type(e).__name__}: {e}"))
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}")

    have = sum(1 for _, dest in todo if dest.exists())
    print(f"\n{have}/{len(todo)} blobs present")
    if failures:
        print(f"{len(failures)} failed:")
        for r, e in failures[:10]:
            print(f"  {r}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
