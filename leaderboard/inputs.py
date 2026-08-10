#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The exact set of files the database is built from, and a cheap signature.

Shared by `ingest.py` (which records the signature) and `app.py` (which checks
it) so the two can never disagree about what "the inputs" means. Deliberately
free of heavy imports: `app.py` runs in a web venv that has no pydantic, so it
cannot import `ingest.py`, which pulls in the scoring package.

Replaces a git-SHA comparison, which was wrong in both directions:

* **False positive on every commit.** The board's data comes from `artifacts/`,
  but the check compared repo HEAD, so a commit touching only a stylesheet
  raised "this view may be behind the artifacts". A staleness warning that
  fires when nothing has gone stale teaches you to ignore it, which costs more
  than having no warning at all.
* **False negative on untracked data.** `artifacts/10/glm-run1/` was untracked
  when it was first ingested. Nothing in git changed, so a SHA check would have
  reported the board fresh while an entire agent run had appeared.

The signature is stat-only -- count, total bytes, newest mtime -- so it costs
microseconds over ~460 files and can run on every request. It is not a content
hash: a file rewritten with identical content still bumps mtime and will ask
for a rebuild. That direction is the safe one.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The board publishes v1.1. v1 is frozen and unchanged, and every score
# published against it stays valid against it -- but v1.1 corrected 1,048
# bounds (STATE.md D35 and D18) and a board mixing the two would be comparing
# submissions scored against different rooflines, which is the one thing this
# file's own part-per-database rule exists to prevent. `meta.manifest_version`
# carries which, and /methodology renders it.
MANIFEST = ROOT / "artifacts" / "09" / "manifest-v1.1.json"
DEFERRED = ROOT / "artifacts" / "deferred.json"
CANDIDATES = ROOT / "artifacts" / "06" / "candidates"
AUTHORITATIVE = ROOT / "artifacts" / "06" / "authoritative"
AGENT_RUNS = ROOT / "artifacts" / "10"
DATASET = ROOT / "data" / "SOL-ExecBench" / "benchmark"


def agent_roots(extra: list[Path] | None = None) -> list[Path]:
    return [AGENT_RUNS, *(Path(p) for p in (extra or []))]


def input_paths(extra_roots: list[Path] | None = None) -> list[Path]:
    """Every file `ingest.py` reads, in a stable order.

    The dataset definitions are excluded on purpose: 235 `definition.json`
    files supply descriptions and axes only, never a number that is scored,
    and walking them on every page request is not worth the stat calls.
    """
    paths: list[Path] = []
    for f in (MANIFEST, DEFERRED):
        if f.exists():
            paths.append(f)
    for d in (CANDIDATES, AUTHORITATIVE):
        if d.exists():
            paths += sorted(d.glob("*.json"))
    for root in agent_roots(extra_roots):
        if not root.exists():
            continue
        paths += sorted(root.glob("*/scored.json"))
        if (root / "scored.json").exists():
            paths.append(root / "scored.json")
    return paths


def signature(extra_roots: list[Path] | None = None) -> dict:
    paths = input_paths(extra_roots)
    total = 0
    newest = 0.0
    newest_path = None
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        total += st.st_size
        if st.st_mtime > newest:
            newest, newest_path = st.st_mtime, p
    return {
        "n_files": len(paths),
        "total_bytes": total,
        "max_mtime": round(newest, 3),
        "newest_file": (str(newest_path.relative_to(ROOT))
                        if newest_path and _under(newest_path, ROOT)
                        else (str(newest_path) if newest_path else None)),
    }


def _under(p: Path, base: Path) -> bool:
    try:
        p.relative_to(base)
        return True
    except ValueError:
        return False


def compare(recorded: dict, current: dict) -> list[str]:
    """Human-readable reasons the database no longer matches its inputs."""
    if not recorded:
        return []
    reasons = []
    if current["n_files"] != recorded.get("n_files"):
        d = current["n_files"] - recorded.get("n_files", 0)
        reasons.append(
            f"{abs(d)} input file{'s' if abs(d) != 1 else ''} "
            f"{'added' if d > 0 else 'removed'} since the last build")
    elif current["total_bytes"] != recorded.get("total_bytes"):
        reasons.append("an input file changed size since the last build")
    elif current["max_mtime"] > recorded.get("max_mtime", 0) + 1e-6:
        reasons.append(f"{current['newest_file']} was modified since the last build")
    return reasons
