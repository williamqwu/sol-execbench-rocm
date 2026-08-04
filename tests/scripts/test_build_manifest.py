# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the T_b collection guard in scripts/build_manifest.py.

CPU-only. The defect these cover produced no error, no warning and no visible
difference in the manifest: 87 T_b artifacts measured at F_LOCK 1300 were merged
into a directory of artifacts measured at 1640, and the manifest built from the
mixture. T_b is a wall-clock time, so those problems' scores would have been
rescaled by the clock ratio, per problem, invisibly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_manifest import collect_t_b  # noqa: E402


def _artifact(directory: Path, problem: str, f_lock: int | None,
              t_b_ms: float = 0.5) -> Path:
    doc = {
        "problem": problem,
        "winner_by_workload": {
            f"uuid-{problem}": {"variant": "v2_compile", "t_b_ms": t_b_ms},
        },
    }
    if f_lock is not None:
        doc["_provenance"] = {"f_lock_mhz": f_lock, "host": "somewhere"}
    else:
        doc["_provenance"] = {"host": "somewhere"}
    path = directory / f"{problem}.json"
    path.write_text(json.dumps(doc))
    return path


def test_artifact_at_the_expected_clock_is_admitted(tmp_path):
    _artifact(tmp_path, "L1__001_x", f_lock=1640)
    assert set(collect_t_b(tmp_path, 1640)) == {"L1__001_x"}


def test_artifact_at_another_clock_is_rejected(tmp_path, capsys):
    _artifact(tmp_path, "L1__001_x", f_lock=1640)
    _artifact(tmp_path, "L2__002_y", f_lock=1300)

    got = collect_t_b(tmp_path, 1640)

    assert set(got) == {"L1__001_x"}, "the 1300 MHz artifact must not be used"
    err = capsys.readouterr().err
    assert "REJECTED 1" in err
    assert "L2__002_y.json" in err
    assert "1300" in err


def test_rejection_is_loud_about_every_file(tmp_path, capsys):
    """Silence is the failure mode being fixed, so the count must be reported."""
    for i in range(7):
        _artifact(tmp_path, f"L2__{i:03d}_y", f_lock=1300)
    assert collect_t_b(tmp_path, 1640) == {}
    err = capsys.readouterr().err
    assert "REJECTED 7" in err
    assert "and 2 more" in err, "should summarise beyond the first five"


def test_artifact_without_a_recorded_clock_is_admitted(tmp_path):
    """Missing provenance is a different defect from the wrong clock.

    An artifact predating F_LOCK stamping is not evidence of being measured
    elsewhere, and check_06 already requires provenance separately. Rejecting
    here would conflate the two.
    """
    _artifact(tmp_path, "L1__001_x", f_lock=None)
    assert set(collect_t_b(tmp_path, 1640)) == {"L1__001_x"}


def test_no_expected_clock_admits_everything(tmp_path):
    """Backwards compatible: the guard is opt-in via the argument."""
    _artifact(tmp_path, "L1__001_x", f_lock=1640)
    _artifact(tmp_path, "L2__002_y", f_lock=1300)
    assert len(collect_t_b(tmp_path)) == 2


def test_absent_directory_is_empty_not_an_error(tmp_path):
    assert collect_t_b(tmp_path / "nope", 1640) == {}


def test_file_without_a_winner_is_skipped(tmp_path):
    """`no-winner.json` and failure records live in the same directory."""
    (tmp_path / "no-winner.json").write_text(json.dumps({"problems": ["a", "b"]}))
    _artifact(tmp_path, "L1__001_x", f_lock=1640)
    assert set(collect_t_b(tmp_path, 1640)) == {"L1__001_x"}


@pytest.mark.parametrize("clock", [1300, 1640, 2200])
def test_only_the_matching_clock_survives(tmp_path, clock):
    for c in (1300, 1640, 2200):
        _artifact(tmp_path, f"L1__{c}_x", f_lock=c)
    assert set(collect_t_b(tmp_path, clock)) == {f"L1__{clock}_x"}
