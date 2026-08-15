# SPDX-License-Identifier: Apache-2.0
"""The board is a view of the PUBLISHED bound, not of the legacy column (D63).

A manifest carries two millisecond columns. `t_sol_ms` is a cycle count over
whatever reference clock the tier that wrote it used -- 1.8 GHz for one MI355X
tier and 2.4 GHz for the other. `t_sol_ms_published` is the bound the manifest
publishes and the one every score is computed against, re-derived at the minimum
of the T_b measurement's own clock bracket.

Measured by rebuilding both databases from `manifest-v4.json` on 2026-08-15,
HEAD's ingest against this one: `t_sol_ms` moves on 3685 of 3957 workload rows
(0.7481x - 1.3370x), `bound_quality` on 147 (ok->narrow 9, ok->loose 15,
narrow->ok 121, loose->vacuous 2), `median_headroom` on 219 of 235 problems, and
13,484 of 14,782 variant scores. The same rebuild against
`artifacts/09/manifest-v1.2.json` moves **nothing**: 0 workload rows, 0 median
headrooms, 0 scores, because both frozen MI350X manifests carry
`t_sol_ms_published` on 0 of 3717 workloads and take the fallback.

That asymmetry is the whole design and both halves are tested here: the MI355X
half because reading the wrong column publishes a quality word about a bound
nothing is scored against, and the MI350X half because a hard switch would blank
every headroom on that board at once.

WHICH OF THESE RUN. Everything below the `client` divider needs fastapi, which
is not installed on the measurement node and has no venv here, so those tests
SKIP rather than pass. They are written, not verified. The ingest-side tests
above the divider need no web stack and do run.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "leaderboard"))

import ingest  # noqa: E402

#: 1.333x apart, which is the D63 ratio: the same cycle count read at 1.8 GHz
#: and at 2.4 GHz. Far enough apart to cross a `bound_quality` boundary, which is
#: what makes the difference board-visible rather than cosmetic.
LEGACY_MS = 0.004
PUBLISHED_MS = 0.003


def _manifest(**overrides) -> dict:
    """One scoreable workload whose two T_SOL columns disagree.

    `t_b_ms` is 0.30 ms: headroom 75x against the legacy column (`ok`) and 100x
    against the published one (`loose`). A test that read the wrong column would
    otherwise still land in the right band and pass.
    """
    w = {"t_sol_ms": LEGACY_MS, "t_sol_cycles": 9600,
         "t_sol_ms_published": PUBLISHED_MS, "t_sol_cycles_published": 7200,
         "t_sol_source": "solar_fused", "sol_bottleneck": "compute",
         "t_b_ms": 0.30, "t_b_variant": "v4_contiguous", "scoreable": True,
         "f_ref_mhz": None}
    w.update(overrides)
    return {
        "manifest_version": "vTEST",
        "problem_set": {"total_in_dataset": 1, "scoreable_problems": 1,
                        "deferred_problems": [],
                        "expected_by_category": {"L1": 1}},
        "stats": {"scoreable_workloads": 1},
        "_provenance": {"part": "MI355X"},
        "problems": {"L1__001_alpha": {
            "category": "L1", "n_workloads": 1, "n_scoreable": 1,
            "workloads": {"u0": w}}},
    }


def test_headroom_bands_band_the_published_bound():
    """T_b/T_SOL is 75x against the legacy column and 100x against the published
    one, and 100x is a band boundary."""
    out = ingest.headroom_bands(_manifest())
    assert out["bands"]["100x - 1000x"] == 1
    assert out["bands"]["10x - 100x"] == 0
    assert out["bound_basis"] == {"published": 1}


def test_headroom_bands_fall_back_when_there_is_no_published_bound():
    """The frozen MI350X shape. Same manifest with the published column removed:
    the band moves back to where the legacy column puts it, and the census says
    so out loud instead of the board looking identical either way."""
    m = _manifest()
    w = m["problems"]["L1__001_alpha"]["workloads"]["u0"]
    del w["t_sol_ms_published"]
    del w["t_sol_cycles_published"]
    out = ingest.headroom_bands(m)
    assert out["bands"]["10x - 100x"] == 1
    assert out["bound_basis"] == {"legacy_unstamped": 1}


def test_the_scoring_map_hands_out_the_published_bound():
    """`bounds()` feeds `sol_score` for the four PyTorch variants. A score
    against the legacy column is a score against a bound nothing else on the
    board -- not the agent runs, not `bound_quality` -- uses."""
    assert ingest.bounds(_manifest()) == {("L1__001_alpha", "u0"):
                                          (PUBLISHED_MS, 0.30)}


def _ingest_to_db(tmp_path: Path, manifest: dict) -> sqlite3.Row:
    db = tmp_path / "b.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript((ROOT / "leaderboard" / "schema.sql").read_text())
    ingest.ingest_meta(conn, manifest, "MI355X", [])
    ingest.ingest_problems(conn, manifest)
    conn.commit()
    return conn


def test_the_workload_row_is_the_published_pair(tmp_path):
    """Milliseconds AND cycles, together. The memory term is a fixed TIME, so
    its cycle count is a function of the clock; a millisecond taken at the
    bracket minimum printed beside a cycle count at some reference clock is a
    pair that describes no frequency at all, and `problem.html` prints both."""
    conn = _ingest_to_db(tmp_path, _manifest())
    row = conn.execute("SELECT * FROM workload").fetchone()
    assert row["t_sol_ms"] == PUBLISHED_MS
    assert row["t_sol_cycles"] == 7200
    assert row["bound_quality"] == "loose"
    assert row["bound_headroom"] == pytest.approx(0.30 / PUBLISHED_MS)
    assert json.loads(
        dict(conn.execute("SELECT key,value FROM meta"))["bound_basis"]
    ) == {"published": 1}


def test_a_legacy_manifest_row_is_written_exactly_as_before(tmp_path):
    """The MI350X path, unchanged: both frozen manifests take it on all 3717 of
    their scoreable workloads and a rebuild moves no row."""
    m = _manifest()
    w = m["problems"]["L1__001_alpha"]["workloads"]["u0"]
    del w["t_sol_ms_published"]
    del w["t_sol_cycles_published"]
    conn = _ingest_to_db(tmp_path, m)
    row = conn.execute("SELECT * FROM workload").fetchone()
    assert row["t_sol_ms"] == LEGACY_MS
    assert row["t_sol_cycles"] == 9600
    assert row["bound_quality"] == "ok"
    assert json.loads(
        dict(conn.execute("SELECT key,value FROM meta"))["bound_basis"]
    ) == {"legacy_unstamped": 1}


def test_the_median_headroom_on_the_problem_row_uses_it_too(tmp_path):
    """`problems.html` sorts on this column and `/methodology` lists the eight
    loosest bounds by it."""
    conn = _ingest_to_db(tmp_path, _manifest())
    row = conn.execute("SELECT median_headroom FROM problem").fetchone()
    assert row["median_headroom"] == pytest.approx(0.30 / PUBLISHED_MS)


# --------------------------------------------------------------------------
# rendered pages -- these need fastapi and SKIP where it is absent
# --------------------------------------------------------------------------
#
# `app.py` used to recompute `T_b / t_sol_ms` in Python on both the problem page
# and the submission x problem page, giving a SECOND answer to a question the
# database already answers in `workload.bound_headroom`. The two agreed only
# while `t_sol_ms` and the published bound did. Both sites now read the column.
#
# The fixture database fills `bound_headroom` from its own `t_sol_ms`, so the
# two are indistinguishable there by construction. These tests therefore DOCTOR
# the fixture -- they move `t_sol_ms` and leave `bound_headroom` alone -- which
# is the only way to tell a read from a division.

def test_the_problem_page_prints_the_stored_headroom(board, client):
    """Doctored so the two disagree: whichever number appears says which code
    path produced it."""
    board.write("UPDATE workload SET t_sol_ms = 0.001, bound_headroom = 42.0, "
                "bound_quality = 'ok' WHERE problem_key='L1__001_alpha' "
                "AND uuid='a1'")
    page = client.get("/problems/L1__001_alpha").text
    assert 'data-sort="42.0"' in page
    # 0.200 / 0.001 = 200 -- what dividing the two columns would have produced.
    assert 'data-sort="200.0"' not in page


def test_the_submission_problem_page_prints_the_stored_headroom(board, client):
    """The second site, on the submission x problem grid."""
    board.write("UPDATE workload SET t_sol_ms = 0.001, bound_headroom = 42.0 "
                "WHERE problem_key='L1__001_alpha' AND uuid='a1'")
    r = client.get("/api/v1/submissions/eager/problems/L1__001_alpha")
    assert r.status_code == 200
    w = next(x for x in r.json()["workloads"] if x["uuid"] == "a1")
    assert w["headroom"] == 42.0


def test_a_workload_with_no_anchor_still_renders(board, client):
    """The guard the old division carried: a deferred problem has a T_SOL and no
    T_b, `bound_quality` returns (None, None), and a None/float division 500'd
    all 15 deferred problem pages once."""
    page = client.get("/problems/Quant__003_gamma")
    assert page.status_code == 200


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
