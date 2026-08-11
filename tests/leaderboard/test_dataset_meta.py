#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The tracked descriptive export, and the rule that it never wins.

`reference/dataset-meta.json` exists so a board built from a clone can describe
a problem: `data/` is gitignored, and without the export a deploy renders every
measured number and no description, reference, input, output, axis or workload
parameter (STATE.md D49).

The hazard a tracked copy of someone else's data always carries is that it goes
stale and nobody notices, because it works. Two things stop that here, and both
are tested: the dataset wins wherever it is present, so a machine that has the
real thing can never be served the copy; and `--check` re-derives the export
and compares it byte for byte, so a stale one is a failing command rather than
a quietly wrong page.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "leaderboard"))

import ingest  # noqa: E402

KEY = "L1__001_alpha"
EXPORTED = {
    KEY: {
        "name": "alpha", "description": "from the export", "hf_id": None,
        "axes": {"n": {"type": "var"}}, "inputs": {}, "outputs": {},
        "reference": "def run(): pass",
        "workloads": [{"uuid": "a1", "axes": {"n": 1}},
                      {"uuid": "a2", "axes": {"n": 2}}],
    }
}


@pytest.fixture()
def fake_dataset(tmp_path, monkeypatch):
    """A dataset directory holding L1/001_alpha, and nothing else."""
    d = tmp_path / "benchmark" / "L1" / "001_alpha"
    d.mkdir(parents=True)
    (d / "definition.json").write_text(json.dumps({
        "name": "alpha", "description": "from the dataset",
        "axes": {"n": {"type": "var"}}, "inputs": {}, "outputs": {},
        "reference": "def run(): pass  # dataset",
    }))
    (d / "workload.jsonl").write_text(
        '{"uuid": "a1", "axes": {"n": 1}}\n{"uuid": "a2", "axes": {"n": 2}}\n')
    monkeypatch.setattr(ingest, "DATASET", tmp_path / "benchmark")
    return tmp_path


def test_the_dataset_wins_when_it_is_there(fake_dataset):
    defn, pairs, src = ingest.problem_source(KEY, EXPORTED)
    assert src == "dataset"
    assert defn["description"] == "from the dataset"
    assert [u for u, _ in pairs] == ["a1", "a2"]


def test_the_export_is_used_only_where_the_dataset_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATASET", tmp_path / "nothing-here")
    defn, pairs, src = ingest.problem_source(KEY, EXPORTED)
    assert src == "export"
    assert defn["description"] == "from the export"
    assert [u for u, _ in pairs] == ["a1", "a2"]


def test_neither_is_an_empty_answer_not_a_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATASET", tmp_path / "nothing-here")
    assert ingest.problem_source("L9__404_nope", EXPORTED) == ({}, [], "")


def test_both_paths_merge_axes_identically(tmp_path, monkeypatch):
    """The const/expr merge runs on whichever source supplied the pairs, so
    the two cannot produce different parameters for the same workload."""
    declared = {"n": {"type": "var"},
                "h": {"type": "const", "value": 8},
                "g": {"type": "expr", "expression": "h * 2"}}
    merged = ingest.merge_axes(declared, [("a1", {"n": 1})])
    assert merged["a1"]["axes"] == {"h": 8, "n": 1, "g": 16}
    assert merged["a1"]["var"] == {"n": 1}
    assert merged["a1"]["i"] == 1


def test_the_position_is_the_datasets_own_ordering():
    """`#4` on our page and `#4` upstream are the same workload only because
    this index is the file's order, not the manifest's uuid sort."""
    merged = ingest.merge_axes({}, [("z", {}), ("a", {}), ("m", {})])
    assert [merged[u]["i"] for u in ("z", "a", "m")] == [1, 2, 3]


@pytest.mark.skipif(not (ROOT / "data" / "SOL-ExecBench" / "benchmark").is_dir(),
                    reason="no dataset on this machine to check against")
def test_the_tracked_export_matches_the_dataset():
    """A tracked copy that has drifted from its source is worse than no copy:
    it works, and it is wrong. This is the guard, and it is one command."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_dataset_meta.py"),
         "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout


def test_the_export_carries_nothing_measured():
    """Descriptive only. A timing, a bound or a tolerance in here would be a
    measured value living outside `artifacts/`, where no provenance stamp
    reaches it."""
    f = ROOT / "reference" / "dataset-meta.json"
    if not f.is_file():
        pytest.skip("export not present")
    d = json.loads(f.read_text())
    allowed = {"name", "description", "hf_id", "axes", "inputs", "outputs",
               "reference", "workloads"}
    for key, p in d["problems"].items():
        assert set(p) <= allowed, (key, set(p) - allowed)
        for w in p["workloads"]:
            assert set(w) <= {"uuid", "axes"}, (key, set(w))
