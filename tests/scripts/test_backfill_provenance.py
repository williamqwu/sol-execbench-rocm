# SPDX-License-Identifier: Apache-2.0
"""A rebased score file must say when it was rebased, without forgetting when it
was measured.

`backfill_scores.py` wrote `json.dumps({**stamp('10-backfill'), **doc})`. `doc`
already carries the measurement's `_provenance` and is spread SECOND, so the
freshly computed stamp lost the merge and was thrown away -- on every file, since
the line was written. 417 score files were rebased against MI355X manifest v4 on
2026-08-15 and every one of them still attested to a run from the previous day
(`utc 2026-08-14T22:03:21`, `git_sha 9b7e8435...-dirty`). `summary.json` was
worse: its write path never called `stamp()` at all.

The naive repair -- let the fresh stamp win -- is the one these tests exist to
forbid. It would erase the host, the cards, the ROCm build and the clock that
`t_k_ms` was measured under, and that is the half of the provenance no rerun can
reconstruct. A rebase is arithmetic over timings it did not take; it is entitled
to a block of its own and to nothing else.

CPU-only. `stamp()` enumerates torch devices, which works on a bare CPU box.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import backfill_scores as bf  # noqa: E402

KEY = "L1__001_attention_softmax_dropout_value_matmul_backward"
UUID = "u0"
BPS = 7999920000000.0

#: What the scorer wrote when the run was actually measured. Deliberately
#: implausible values: if any of them survives into the assertions by being
#: recomputed rather than preserved, the test cannot pass by coincidence.
MEASURED_PROVENANCE = {
    "task": "10-score",
    "utc": "2026-08-14T22:03:21.502862+00:00",
    "git_sha": "9b7e84352dde9e6dc5ddae5097171fa614b2785f-dirty",
    "host": "mia1-p02-g45",
    "part": "MI355X",
    "f_lock_mhz": None,
    "authoritative_gpu": 3,
}


def _manifest(path: Path) -> Path:
    """A manifest in the unlocked MI355X shape: `f_lock_mhz` null, so the
    per-problem card check is ON and refuses (no anchor tree exists in this
    temp tree). That refusal is deliberate -- it is the realistic path, and it
    is the path on which a file gets rewritten *without* any record having
    changed, which is exactly where a stamp is easiest to lose."""
    path.write_text(json.dumps({
        "manifest_version": "vTEST",
        "_provenance": {"f_lock_mhz": None, "authoritative_gpu": 3,
                        "part": "MI355X"},
        "part": "MI355X",
        "problems": {KEY: {"workloads": {UUID: {
            "t_sol_ms": 1.3333333333333333,
            "t_sol_cycles": 2_400_000,
            "compute_cycles": 2_400_000.0,
            "memory_bytes": 4096,
            "dram_byte_per_sec": BPS,
            "mac_per_cycle": 524288.0,
            "t_b_ms": 4.0,
            "t_b_variant": "v4_contiguous",
        }}}},
    }))
    return path


def _run(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Backfill one score file and its summary in a throwaway tree.

    `bf.ROOT` is redirected rather than the repo's own `artifacts/10` being
    touched: these are MI355X release artifacts and a test must not rewrite one.
    """
    monkeypatch.setattr(bf, "ROOT", tmp_path)
    scores = tmp_path / "artifacts" / "10" / "scores" / "run-x" / "claude-code"
    scores.mkdir(parents=True)
    score_file = scores / f"{KEY}.json"
    score_file.write_text(json.dumps({
        "_provenance": dict(MEASURED_PROVENANCE),
        "problem": KEY,
        "records": [{"workload_uuid": UUID, "t_k_ms": 2.0, "t_ref_ms": 5.0,
                     "correct": True, "t_sol_ms": None, "t_b_ms": None,
                     "clock_bracket": {"clock_before_mhz": 2400.0,
                                       "clock_after_mhz": 2400.0}}],
    }, indent=2))
    summary = scores.parent / "summary.json"
    summary.write_text(json.dumps({
        "_provenance": dict(MEASURED_PROVENANCE),
        "run_id": "run-x",
        "score_bases": {"sol_score_v1": 1},
    }, indent=2))

    manifest = _manifest(tmp_path / "manifest-vTEST.json")
    monkeypatch.setattr(sys, "argv", ["backfill_scores.py", "--run-id", "run-x",
                                      "--manifest", str(manifest),
                                      "--part", "MI355X"])
    assert bf.main() == 0
    return score_file, summary


def test_the_measurement_provenance_survives_the_rebase(tmp_path, monkeypatch):
    """The half that cannot be recovered. Which host, which cards, which clock
    produced `t_k_ms` is not a fact the rebase is in a position to restate, and
    letting the fresh stamp win would have deleted it."""
    score_file, _ = _run(tmp_path, monkeypatch)
    doc = json.loads(score_file.read_text())
    assert doc["_provenance"] == MEASURED_PROVENANCE


def test_the_rebase_stamps_itself_beside_it(tmp_path, monkeypatch):
    """And the stamp is FRESH -- the defect was a discarded stamp, not a missing
    call, so asserting the key exists would have passed against the old code
    too if the old code had put the old block there."""
    score_file, _ = _run(tmp_path, monkeypatch)
    doc = json.loads(score_file.read_text())
    prov = doc[bf.BACKFILL_PROVENANCE_KEY]
    assert prov["task"] == "10-backfill"
    assert prov["utc"] != MEASURED_PROVENANCE["utc"]
    assert prov["git_sha"] != MEASURED_PROVENANCE["git_sha"]
    assert prov["manifest_version"] == "vTEST"
    assert prov["rebase_only"] is True


def test_the_summary_is_stamped_too(tmp_path, monkeypatch):
    """Its basis census, violation count and card-enforcement block are all
    recomputed here, and its write path called `stamp()` nowhere at all."""
    score_file, summary = _run(tmp_path, monkeypatch)
    doc = json.loads(summary.read_text())
    assert doc["_provenance"] == MEASURED_PROVENANCE
    prov = doc[bf.BACKFILL_PROVENANCE_KEY]
    assert prov["task"] == "10-backfill"
    assert prov["utc"] != MEASURED_PROVENANCE["utc"]
    # One rebase, one timestamp: the summary and the files it summarises must
    # not disagree about when the run was last recomputed.
    assert prov["utc"] == json.loads(
        score_file.read_text())[bf.BACKFILL_PROVENANCE_KEY]["utc"]


def test_the_declared_part_outranks_the_host(tmp_path, monkeypatch):
    """A backfill is arithmetic and touches no GPU, so rebasing MI355X scores on
    an MI350X node is legitimate. The block must say MI355X and still record
    what the host actually was, rather than refusing or silently relabelling."""
    score_file, _ = _run(tmp_path, monkeypatch)
    prov = json.loads(score_file.read_text())[bf.BACKFILL_PROVENANCE_KEY]
    assert prov["part"] == "MI355X"
    assert prov["part_source"] == "declared"
    assert "part_detected" in prov


def test_the_old_merge_shape_would_have_failed_these(tmp_path, monkeypatch):
    """The defect, reproduced in one line, so this file documents what it caught
    rather than only asserting the repaired behaviour.

    `provenance.stamp` returns `{"_provenance": block}`; spreading it before a
    doc that already has that key discards it entirely.
    """
    from provenance import stamp

    doc = {"_provenance": dict(MEASURED_PROVENANCE)}
    merged = {**stamp("10-backfill"), **doc}
    assert merged["_provenance"] == MEASURED_PROVENANCE
    assert bf.BACKFILL_PROVENANCE_KEY not in merged


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch):
    """The stamp is computed before the loop now, which must not turn a dry run
    into a write."""
    monkeypatch.setattr(bf, "ROOT", tmp_path)
    scores = tmp_path / "artifacts" / "10" / "scores" / "run-y" / "claude-code"
    scores.mkdir(parents=True)
    f = scores / f"{KEY}.json"
    original = json.dumps({"_provenance": dict(MEASURED_PROVENANCE),
                           "problem": KEY,
                           "records": [{"workload_uuid": UUID, "t_k_ms": 2.0,
                                        "correct": True}]}, indent=2)
    f.write_text(original)
    manifest = _manifest(tmp_path / "m.json")
    monkeypatch.setattr(sys, "argv", ["backfill_scores.py", "--run-id", "run-y",
                                      "--manifest", str(manifest),
                                      "--part", "MI355X", "--dry-run"])
    assert bf.main() == 0
    assert f.read_text() == original


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
