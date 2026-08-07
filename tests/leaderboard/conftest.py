#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the leaderboard tests.

Two kinds of board, and the distinction matters:

* **`board` / `client`** — a small database this file builds from
  `leaderboard/schema.sql`, in a temp directory that `app.DB_DIR` is pointed
  at. Twelve workloads, five submissions, and every state the UI has to tell
  apart present on purpose: a hidden run, two trials of one group, a kernel
  whose re-time timed out, a workload whose bound is invalid, an eval with no
  timestamp. Tests that assert an *invariant* use this, because an invariant
  that only holds on today's 17 MB artifact set has not been tested — it has
  been observed.
* **`real_board` / `real_client`** — whatever `ingest.py` last built. Only for
  claims about the real data (D22 across 14k rows, the deferred manifest, the
  rebuild being atomic). Every one of those tests says in its docstring why the
  fixture database cannot answer it.

The real-board fixtures resolve the file through `app.part_databases()` and
then pin it with `SOLBENCH_DB`, so the connection and the client are provably
the same file. They used to differ: `conn` opened `leaderboard/solbench.db`
while `client` resolved per-part, and the two agreed only because both happened
to be built from the same artifacts by hand. Worse, the skip guard named the
legacy path, so on a fresh build -- which writes `db/solbench-MI350X.db` and
nothing else -- all seventeen tests skipped green.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "leaderboard"
SCHEMA = LB / "schema.sql"
if str(LB) not in sys.path:
    sys.path.insert(0, str(LB))


# --------------------------------------------------------------------------
# the fixture database
# --------------------------------------------------------------------------

# Two scoreable problems (4 + 2 workloads) and one deferred, so the benchmark
# denominator is 6 and every aggregate can be checked by hand rather than
# against another query. The deferred problem carries all three of its
# explanation fields because the page that renders them 500'd once.
PROBLEMS = [
    dict(key="L1__001_alpha", category="L1", name="alpha", n_workloads=4,
         n_scoreable=4, deferred=0, median_headroom=2.0,
         reference="def alpha(x):\n    return x + 1\n"),
    dict(key="L2__002_beta", category="L2", name="beta", n_workloads=2,
         n_scoreable=2, deferred=0, median_headroom=1.6,
         reference="def beta(x):\n    return x * 2\n"),
    dict(key="Quant__003_gamma", category="Quant", name="gamma", n_workloads=2,
         n_scoreable=0, deferred=1, median_headroom=None,
         reference="def gamma(x):\n    return x\n",
         deferred_reason="nvfp4-no-rocm-path",
         deferred_mechanism="No ROCm kernel exists for the NVFP4 block layout.",
         deferred_error="RuntimeError: quantize_nvfp4 is not implemented"),
]

# (problem, uuid, scoreable, t_sol_ms, t_b_ms)
WORKLOADS = [
    ("L1__001_alpha", "a1", 1, 0.100, 0.200),
    ("L1__001_alpha", "a2", 1, 0.100, 0.180),
    ("L1__001_alpha", "a3", 1, 0.120, 0.240),
    ("L1__001_alpha", "a4", 1, 0.120, 0.200),
    ("L2__002_beta", "b1", 1, 0.050, 0.080),
    ("L2__002_beta", "b2", 1, 0.050, 0.075),
    ("Quant__003_gamma", "g1", 0, 0.030, None),
    ("Quant__003_gamma", "g2", 0, 0.030, None),
]

GROUP = ("agent-trials", "Fixture-Agent · sandbox harness")

SUBMISSIONS = [
    dict(slug="ref-v1-eager", name="eager PyTorch", kind="reference_variant",
         created_utc="2026-01-01T00:00:00+00:00", board_visible=1,
         part="MI350X", model=None),
    dict(slug="agent-alpha", name="Agent Alpha", kind="agent", model="fixture-1",
         created_utc="2026-02-01T00:00:00+00:00", board_visible=1,
         part="MI350X"),
    # Off the ranking, fully ingested. The case `board_visible` exists for:
    # its best workload (0.98) beats every ranked run, so if the flag is read
    # in the wrong place a headline moves.
    dict(slug="agent-trial-a", name="Agent Alpha (capped)", kind="agent",
         model="fixture-1", created_utc="2026-03-01T00:00:00+00:00",
         board_visible=0,
         exclusion_reason="Every session was stopped by its $8 cap, so this is "
                          "a cost measurement, not a score measurement.",
         group_slug=GROUP[0], group_name=GROUP[1], trial_label="$8 / problem",
         trial_n=1, constraint_json=json.dumps({"budget_usd_per_session": 8.0}),
         part="MI350X"),
    dict(slug="agent-trial-b", name="Agent Alpha (uncapped)", kind="agent",
         model="fixture-1", created_utc="2026-04-01T00:00:00+00:00",
         board_visible=1, group_slug=GROUP[0], group_name=GROUP[1],
         trial_label="$100 / problem", trial_n=2,
         constraint_json=json.dumps({"budget_usd_per_session": 100.0}),
         part="MI350X"),
    dict(slug="agent-timeout", name="Agent Timeout", kind="agent",
         model="fixture-2", created_utc="2026-05-01T00:00:00+00:00",
         board_visible=1, part="MI350X",
         depth_note="This harness recorded no trajectory."),
]

# (slug, problem, uuid, status, latency_ms, score)
#
# `agent-trial-a` on a3 is the bound-invalid row: PASSED, a latency below
# T_SOL, and therefore NULL score. It is here so the trial switcher's mean can
# be checked against the run card's -- `AVG(score)` skips that NULL and the two
# then disagree, which is exactly how the bug shipped.
RESULTS = [
    ("ref-v1-eager", "L1__001_alpha", "a1", "PASSED", 0.200, 0.5),
    ("ref-v1-eager", "L1__001_alpha", "a2", "PASSED", 0.180, 0.5),
    ("ref-v1-eager", "L1__001_alpha", "a3", "PASSED", 0.240, 0.5),
    ("ref-v1-eager", "L1__001_alpha", "a4", "PASSED", 0.200, 0.5),
    ("ref-v1-eager", "L2__002_beta", "b1", "PASSED", 0.080, 0.5),
    ("ref-v1-eager", "L2__002_beta", "b2", "PASSED", 0.075, 0.5),

    ("agent-alpha", "L1__001_alpha", "a1", "PASSED", 0.150, 0.60),
    ("agent-alpha", "L1__001_alpha", "a2", "PASSED", 0.160, 0.55),
    ("agent-alpha", "L1__001_alpha", "a3", "PASSED", 0.230, 0.52),
    ("agent-alpha", "L1__001_alpha", "a4", "FAILED", 0.900, None),

    ("agent-trial-a", "L1__001_alpha", "a1", "PASSED", 0.105, 0.98),
    ("agent-trial-a", "L1__001_alpha", "a2", "PASSED", 0.120, 0.90),
    ("agent-trial-a", "L1__001_alpha", "a3", "PASSED", 0.090, None),
    ("agent-trial-a", "L1__001_alpha", "a4", "FAILED", 0.500, None),

    ("agent-trial-b", "L1__001_alpha", "a1", "PASSED", 0.145, 0.62),
    ("agent-trial-b", "L1__001_alpha", "a2", "PASSED", 0.155, 0.58),

    ("agent-timeout", "L1__001_alpha", "a1", "PASSED", 0.190, 0.51),
]

# (slug, problem, retime_ok, retime_error)
#
# Three states, because there are three: measured, measured-and-failed, and
# "the ingest recorded nothing", which is NULL and is not the middle one.
KERNELS = [
    ("agent-trial-a", "L1__001_alpha", 1, None),
    ("agent-timeout", "L2__002_beta", 0,
     "TimeoutExpired: agent_score.py ... timed out after 1200s"),
    ("agent-alpha", "L2__002_beta", None, None),
]

# (slug, problem, started, finished, source). `agent-trial-b` is deliberately
# absent: no timestamp evidence, therefore no row.
WINDOWS = [
    ("agent-trial-a", "L1__001_alpha", "2026-03-01T10:00:00+00:00",
     "2026-03-01T11:30:00+00:00", "first_last_eval"),
    ("agent-alpha", "L1__001_alpha", None, "2026-02-01T09:15:00+00:00",
     "retime_only"),
]

# (n, utc, minutes_in, ok, passed, workloads, mean_score)
#
# Eval 3 is a harness error (nothing ran) and eval 4 has no timestamp at all --
# the two cases that a chart which places NULL at x=0, or a template that
# writes `minutes_in or 0`, renders as measurements.
TRAJECTORY = [
    (1, "2026-03-01T10:00:00+00:00", 0.0, 1, 2, 4, 0.30),
    (2, "2026-03-01T10:37:00+00:00", 37.0, 1, 4, 4, 0.47),
    (3, "2026-03-01T11:00:00+00:00", 60.0, 0, 0, 0, None),
    (4, None, None, 1, 3, 4, 0.44),
]

META = {
    "part": "MI350X",
    "manifest_version": "fixture-v1",
    "manifest_utc": "2026-01-01T00:00:00+00:00",
    "db_built_utc": "2026-06-01T00:00:00+00:00",
    "device": "AMD Instinct MI350X",
    "f_lock_mhz": "1300",
    "rocm_version": "6.4.1",
    "torch_version": "2.7.0",
    "total_problems": "3",
    "scoreable_problems": "2",
    "scoreable_workloads": "6",
    "repo_git_sha": "0" * 40,
    "methodology": "S = (T_b - T_k) / (T_b - T_SOL) / 2 + 0.5",
    "bound_sources": json.dumps({"solar_fused": 6}),
    "excluded_submissions": json.dumps({}),
    "problems_with_invalid_bound": json.dumps([]),
    # An empty recorded signature makes `inputs.compare` return no reasons, so
    # the fixture board never renders a staleness banner about the real repo's
    # artifacts -- which have nothing to do with it.
    "input_signature": json.dumps({}),
    "input_extra_roots": json.dumps([]),
}


def build_fixture_db(path: Path, part: str = "MI350X") -> Path:
    """Write a small board to *path*, built from the real `schema.sql`.

    The schema is the shipped file rather than a copy, so a column added there
    without being ingested, or a CHECK constraint tightened, shows up here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text())
    meta = {**META, "part": part, "device": f"AMD Instinct {part}"}
    conn.executemany("INSERT INTO meta (key,value) VALUES (?,?)", meta.items())
    for p in PROBLEMS:
        conn.execute(
            """INSERT INTO problem (key, category, name, description, reference,
                                    axes_json, inputs_json, outputs_json,
                                    n_workloads, n_scoreable, deferred,
                                    deferred_reason, deferred_mechanism,
                                    deferred_error, median_headroom)
               VALUES (:key, :category, :name, :description, :reference,
                       '{}', '{}', '{}', :n_workloads, :n_scoreable, :deferred,
                       :deferred_reason, :deferred_mechanism, :deferred_error,
                       :median_headroom)""",
            {"description": f"fixture problem {p['key']}",
             "deferred_reason": None, "deferred_mechanism": None,
             "deferred_error": None, **p})
    for key, uuid, scoreable, t_sol, t_b in WORKLOADS:
        conn.execute(
            """INSERT INTO workload (problem_key, uuid, axes_json, t_sol_ms,
                                     t_sol_cycles, t_sol_source, sol_bottleneck,
                                     t_b_ms, t_b_variant, tol_atol, tol_rtol,
                                     scoreable)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key, uuid, json.dumps({"n": 1024}), t_sol,
             int(t_sol * 1.3e6) if t_sol else None, "solar_fused", "memory",
             t_b, "v1_eager" if t_b else None, 1e-3, 1e-3, scoreable))
    ids = {}
    for s in SUBMISSIONS:
        cols = ["slug", "name", "kind", "model", "created_utc", "board_visible",
                "part", "exclusion_reason", "group_slug", "group_name",
                "trial_label", "trial_n", "constraint_json", "depth_note",
                "provenance_json", "cost_usd", "wall_seconds", "gpu"]
        row = {c: s.get(c) for c in cols}
        row["provenance_json"] = json.dumps({"part": s.get("part")})
        cur = conn.execute(
            f"INSERT INTO submission ({','.join(cols)}) "
            f"VALUES ({','.join(':' + c for c in cols)})", row)
        ids[s["slug"]] = cur.lastrowid
    for slug, key, uuid, status, latency, score in RESULTS:
        conn.execute(
            """INSERT INTO result (submission_id, problem_key, workload_uuid,
                                   status, latency_ms, score, flagged)
               VALUES (?,?,?,?,?,?,0)""",
            (ids[slug], key, uuid, status, latency, score))
    for slug, key, ok, err in KERNELS:
        src = f"# kernel: {slug} on {key}\ndef run(x):\n    return x\n"
        conn.execute(
            """INSERT INTO run_kernel (submission_id, problem_key, source,
                                       n_lines, sha256, retime_ok, retime_error)
               VALUES (?,?,?,?,?,?,?)""",
            (ids[slug], key, src, src.count("\n"), "0" * 64, ok, err))
    for slug, key, started, finished, source in WINDOWS:
        conn.execute(
            """INSERT INTO run_window (submission_id, problem_key, started_utc,
                                       finished_utc, source) VALUES (?,?,?,?,?)""",
            (ids[slug], key, started, finished, source))
    for n, utc, minutes, ok, passed, workloads, score in TRAJECTORY:
        conn.execute(
            """INSERT INTO trajectory_eval (submission_id, problem_key, n, utc,
                                            minutes_in, ok, all_passed, passed,
                                            workloads, geomean_speedup,
                                            mean_score, kernel_lines)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ids["agent-trial-a"], "L1__001_alpha", n, utc, minutes, ok,
             int(passed == workloads and workloads > 0), passed, workloads,
             None, score, 40))
    conn.execute(
        """INSERT INTO run_effort (submission_id, problem_key, cost_usd,
                                   wall_seconds, turns, harness_evals, capped)
           VALUES (?,?,?,?,?,?,1)""",
        (ids["agent-trial-a"], "L1__001_alpha", 8.0, 5400.0, 42, 4))
    conn.execute(
        """INSERT INTO variant_source (problem_key, variant, source, n_lines)
           VALUES ('L1__001_alpha', 'v1_eager', 'def alpha(x):\n    return x + 1\n', 2)""")
    conn.commit()
    conn.close()
    return path


class Board:
    """A temp directory laid out the way `app.DB_DIR` expects, plus helpers."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, part: str = "MI350X") -> Path:
        return self.dir / f"solbench-{part}.db"

    def add(self, part: str) -> Path:
        return build_fixture_db(self.path(part), part)

    def drop(self, part: str) -> None:
        self.path(part).unlink()

    def write(self, sql: str, args=(), part: str = "MI350X") -> None:
        """Doctor the fixture in place — how a bad state is produced on purpose."""
        conn = sqlite3.connect(self.path(part))
        conn.execute(sql, args)
        conn.commit()
        conn.close()

    def conn(self, part: str = "MI350X") -> sqlite3.Connection:
        c = sqlite3.connect(f"file:{self.path(part)}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        return c


@pytest.fixture(scope="session")
def app_mod():
    pytest.importorskip("fastapi", reason="leaderboard venv only")
    import app
    return app


@pytest.fixture(scope="session")
def _fixture_db_template(tmp_path_factory) -> Path:
    """Built once; every test gets its own copy so it may doctor it freely."""
    return build_fixture_db(tmp_path_factory.mktemp("template") / "board.db")


@pytest.fixture
def board(tmp_path, monkeypatch, app_mod, _fixture_db_template) -> Board:
    """An isolated one-part board, with `app` pointed at it.

    `LEGACY_DB` is redirected at a path that does not exist and both
    environment overrides are cleared, so a test can never accidentally read
    the machine's real 17 MB board and pass for the wrong reason.
    """
    b = Board(tmp_path / "db")
    shutil.copy(_fixture_db_template, b.path("MI350X"))
    monkeypatch.setattr(app_mod, "DB_DIR", b.dir)
    monkeypatch.setattr(app_mod, "LEGACY_DB", tmp_path / "no-legacy.db")
    monkeypatch.delenv("SOLBENCH_DB", raising=False)
    monkeypatch.delenv("SOLBENCH_PART", raising=False)
    return b


@pytest.fixture
def client(board, app_mod):
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


@pytest.fixture
def make_client(app_mod):
    """A fresh client with its own cookie jar — the part cookie is sticky."""
    from fastapi.testclient import TestClient

    def go(**kw):
        return TestClient(app_mod.app, **kw)
    return go


# --------------------------------------------------------------------------
# the real board
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_db(app_mod) -> Path:
    """The database `ingest.py` last built, resolved the way the app resolves it.

    Not a hardcoded path: a fresh build writes `db/solbench-<PART>.db` and no
    longer writes `leaderboard/solbench.db`, so a guard naming the legacy file
    skipped the entire suite green.
    """
    dbs = app_mod.part_databases()
    if not dbs:
        pytest.skip("no board built; run leaderboard/ingest.py")
    return dbs.get(app_mod.resolve_part()) or next(iter(dbs.values()))


@pytest.fixture(scope="session")
def real_conn(real_db):
    c = sqlite3.connect(f"file:{real_db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture
def real_client(real_db, app_mod, monkeypatch):
    """A client pinned to exactly the file `real_conn` reads.

    `SOLBENCH_DB` rather than a shared resolution, so the two fixtures cannot
    drift apart again: the pin collapses `part_databases()` to this one file.
    """
    from fastapi.testclient import TestClient
    monkeypatch.setenv("SOLBENCH_DB", str(real_db))
    monkeypatch.delenv("SOLBENCH_PART", raising=False)
    return TestClient(app_mod.app)
