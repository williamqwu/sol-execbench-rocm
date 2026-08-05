-- SPDX-License-Identifier: Apache-2.0
--
-- SOL-ExecBench-AMD leaderboard.
--
-- The database is a VIEW of the artifacts, never a source of truth. Every row
-- here is derived from artifacts/09/manifest-v1.json and the per-problem
-- sweep artifacts, and `ingest.py` rebuilds it from scratch. If the database
-- and the artifacts disagree, the artifacts are right and the database is
-- stale -- so it carries the manifest's git SHA in `meta` and the UI shows it.

PRAGMA journal_mode = WAL;

DROP TABLE IF EXISTS result;
DROP TABLE IF EXISTS submission;
DROP TABLE IF EXISTS workload;
DROP TABLE IF EXISTS problem;
DROP TABLE IF EXISTS meta;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE problem (
    key             TEXT PRIMARY KEY,   -- 'L1__069_rms_norm'
    category        TEXT NOT NULL,      -- L1 | L2 | Quant | FlashInfer-Bench
    name            TEXT NOT NULL,
    description     TEXT,
    hf_id           TEXT,
    reference       TEXT,               -- the reference implementation source
    axes_json       TEXT,
    inputs_json     TEXT,
    outputs_json    TEXT,
    n_workloads     INTEGER NOT NULL DEFAULT 0,
    n_scoreable     INTEGER NOT NULL DEFAULT 0,
    deferred        INTEGER NOT NULL DEFAULT 0,
    -- All three, because a deferral shown as a bare "0 scoreable" reads as a
    -- gap in the sweep. It is not: `artifacts/deferred.json` carries the slug,
    -- the mechanism and the interpreter's own error text, and the UI shows
    -- them next to the zero so the reader never has to guess which it is.
    deferred_reason    TEXT,   -- slug, e.g. 'nvfp4-no-rocm-path'
    deferred_mechanism TEXT,   -- one sentence: why more work does not close it
    deferred_error     TEXT,   -- the exception the reference itself raises
    -- median over workloads of T_b / T_SOL: how much room a kernel has.
    median_headroom REAL
);

CREATE TABLE workload (
    problem_key        TEXT NOT NULL REFERENCES problem(key),
    uuid               TEXT NOT NULL,
    axes_json          TEXT,
    t_sol_cycles       INTEGER,
    t_sol_ms           REAL,
    t_sol_source       TEXT,   -- solar_fused | declared_traffic | max_of_both | ...
    t_sol_cycles_solar INTEGER,
    t_sol_cycles_traffic INTEGER,
    sol_bottleneck     TEXT,   -- memory | compute
    t_b_ms             REAL,
    t_b_variant        TEXT,
    tol_atol           REAL,
    tol_rtol           REAL,
    tol_ratio          REAL,
    tol_derivation     TEXT,
    scoreable          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (problem_key, uuid)
);
CREATE INDEX idx_workload_problem ON workload(problem_key);

-- One row per leaderboard entry. `kind` keeps agents and reference variants
-- visibly distinct: the four PyTorch variants are what T_b is DERIVED from, so
-- they sit at S=0.5 by construction and must never be read as agent results.
CREATE TABLE submission (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    slug            TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL,      -- reference_variant | agent | human
    author          TEXT,
    model           TEXT,
    created_utc     TEXT,
    notes           TEXT,
    provenance_json TEXT,
    cost_usd        REAL,
    wall_seconds    REAL,
    gpu             TEXT
);

CREATE TABLE result (
    submission_id  INTEGER NOT NULL REFERENCES submission(id),
    problem_key    TEXT NOT NULL,
    workload_uuid  TEXT NOT NULL,
    status         TEXT,                -- PASSED | FAILED | ERROR | NOT_ATTEMPTED
    latency_ms     REAL,
    score          REAL,
    flagged        INTEGER NOT NULL DEFAULT 0,
    note           TEXT,
    PRIMARY KEY (submission_id, problem_key, workload_uuid)
);
CREATE INDEX idx_result_submission ON result(submission_id);
CREATE INDEX idx_result_problem    ON result(problem_key);
