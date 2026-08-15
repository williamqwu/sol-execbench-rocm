# SPDX-License-Identifier: Apache-2.0
"""`agent_score.bounds` scores against the PUBLISHED bound, not the raw column.

D63 in the one place it computes rather than reports. `agent_score.py` is the
submission path: it re-times a kernel on GPU 0 and turns that time into a score.
It used to read `w["t_sol_ms"]` straight out of the manifest record, which on a
manifest built on the unlocked basis is a cycle count divided by whichever
reference clock the winning tier happened to use -- not the bound the manifest
publishes, not the bound check D gates on, and not the bound the board serves.

Measured over `artifacts/09-MI355X/manifest-v4.json`, the two columns differ by
more than 1% on 1622 of 3717 scoreable workloads. A submission scored against
the wrong one is wrong by that much and nothing downstream can see it, because
the record it lands in says only "manifest v4".

These tests pin three things:

  1. the published column wins when it is there,
  2. the legacy column is still usable for the two frozen MI350X manifests,
     which carry `t_sol_ms_published` on 0 of 3717 workloads -- a hard switch
     would blind that whole board, which is a bigger error,
  3. whichever happened is COUNTED and written into the artifact, because a run
     scored on a legacy column and one scored on the published bound are not
     comparable and nothing else in the file would say which it was.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import agent_score as ags  # noqa: E402


def _manifest(tmp_path: Path, workloads: dict, version: str = "v4") -> Path:
    doc = {
        "manifest_version": version,
        "problems": {"L1__001_x": {"workloads": workloads}},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(doc))
    return p


def test_published_column_wins_over_the_legacy_one(tmp_path):
    """Both columns present and different: the published one is the bound.

    The 0.75x ratio is not arbitrary -- it is the size of the real D63 error on
    this corpus (a 1.8 GHz reference column against a ~2.4 GHz bracket), so a
    regression that reads the wrong column reproduces the shipped defect exactly.
    """
    m = _manifest(tmp_path, {"u1": {
        "scoreable": True, "t_b_ms": 10.0,
        "t_sol_ms": 1.0, "f_ref_mhz": 1800.0,
        "t_sol_ms_published": 0.75,
    }})
    b, version, basis = ags.bounds(m)
    assert b[("L1__001_x", "u1")] == (0.75, 10.0)
    assert version == "v4"
    assert basis == {"published": 1}


def test_legacy_column_still_scores_a_frozen_mi350x_manifest(tmp_path):
    """No `t_sol_ms_published` anywhere: the run is still scoreable, and counted.

    `manifest-v1.json` and `manifest-v1.2.json` predate the field and are frozen,
    so they will never gain it. Refusing them would take the MI350X board from
    "scored against a column with no clock stamp" to "not scored at all".
    """
    m = _manifest(tmp_path, {"u1": {
        "scoreable": True, "t_b_ms": 10.0, "t_sol_ms": 2.0,
    }}, version="v1.2")
    b, version, basis = ags.bounds(m)
    assert b[("L1__001_x", "u1")] == (2.0, 10.0)
    assert version == "v1.2"
    assert basis == {"legacy_unstamped": 1}


def test_stamped_legacy_column_is_reported_as_such(tmp_path):
    """A legacy column that carries its own clock is legible, and says so.

    `t_sol_at.bound_ms` accepts it; the basis is `legacy_at_stated_clock` rather
    than `published`, so a reader can tell this run from one scored against a
    published bound without re-deriving anything.
    """
    m = _manifest(tmp_path, {"u1": {
        "scoreable": True, "t_b_ms": 10.0,
        "t_sol_ms": 2.0, "f_ref_mhz": 2400.0,
        "t_sol_cycles": 4800, "compute_cycles": 4800.0,
    }})
    _b, _version, basis = ags.bounds(m)
    assert basis == {"legacy_at_stated_clock": 1}


def test_a_mixed_manifest_reports_both_bases(tmp_path):
    """The census is per record, not per manifest.

    A manifest can be half re-derived -- which is exactly the state
    `manifest-v4` was in mid-session, `f_ref_mhz` on 1421 of 3957 records -- and
    a single "basis" for the whole file would have hidden that.
    """
    m = _manifest(tmp_path, {
        "u1": {"scoreable": True, "t_b_ms": 10.0, "t_sol_ms": 1.0,
               "t_sol_ms_published": 0.75},
        "u2": {"scoreable": True, "t_b_ms": 10.0, "t_sol_ms": 2.0},
    })
    b, _version, basis = ags.bounds(m)
    assert basis == {"legacy_unstamped": 1, "published": 1}
    assert b[("L1__001_x", "u1")][0] == 0.75
    assert b[("L1__001_x", "u2")][0] == 2.0


def test_unscoreable_and_boundless_records_are_neither_scored_nor_counted(tmp_path):
    """A record with no bound is absent from both the map and the census.

    Counting it would make the census a workload count rather than a statement
    about the bounds that produced scores, and putting it in the map would score
    a kernel against `None`.
    """
    m = _manifest(tmp_path, {
        "skip_unscoreable": {"scoreable": False, "t_b_ms": 10.0,
                             "t_sol_ms_published": 1.0},
        "skip_no_tb": {"scoreable": True, "t_sol_ms_published": 1.0},
        "skip_no_bound": {"scoreable": True, "t_b_ms": 10.0},
        "skip_zero_bound": {"scoreable": True, "t_b_ms": 10.0,
                            "t_sol_ms_published": 0.0, "t_sol_ms": 0.0},
        "keep": {"scoreable": True, "t_b_ms": 10.0, "t_sol_ms_published": 1.0},
    })
    b, _version, basis = ags.bounds(m)
    assert set(k[1] for k in b) == {"keep"}
    assert basis == {"published": 1}


def test_bounds_is_the_single_definition_not_a_copy():
    """`agent_score` imports the choke point rather than reimplementing it.

    Identity, not equality: two functions that agree today and drift tomorrow
    are how D63 happened in the first place.
    """
    from bound_headroom import published_bound_ms
    assert ags.published_bound_ms is published_bound_ms
