-- SPDX-License-Identifier: Apache-2.0
--
-- SOL-ExecBench-ROCm leaderboard.
--
-- The database is a VIEW of the artifacts, never a source of truth. Every row
-- here is derived from artifacts/09/manifest-v1.json and the per-problem
-- sweep artifacts, and `ingest.py` rebuilds it from scratch. If the database
-- and the artifacts disagree, the artifacts are right and the database is
-- stale -- so it carries the manifest's git SHA in `meta` and the UI shows it.

PRAGMA journal_mode = WAL;

DROP TABLE IF EXISTS transcript;
DROP TABLE IF EXISTS trajectory_eval;
DROP TABLE IF EXISTS run_effort;
DROP TABLE IF EXISTS run_kernel;
DROP TABLE IF EXISTS variant_source;
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
    gpu             TEXT,
    -- What this harness did NOT record. glm-run1 has 24 problems and no
    -- trajectory, no transcripts and no per-problem cost; the Opus run has 4
    -- problems and all three. Without this the deepest sections are empty on
    -- the broadest submission, and an empty section reads as a missing feature
    -- rather than as a harness that never wrote the file.
    depth_note      TEXT,
    depth_json      TEXT
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

-- ---------------------------------------------------------------------------
-- Depth: what a submission actually did on one problem.
--
-- Everything below is per (submission, problem) -- the cell the run page
-- shows. It is all derived from files the agent harness already wrote and that
-- nothing was reading. Availability is UNEVEN by harness, and that is a fact
-- about the run, not a bug to paper over: `submission.depth_note` records what
-- a given harness did not record, so an empty section can say why instead of
-- looking like a gap.

-- The kernel the submission proposed. Small enough to inline (36 files, ~150
-- lines each); keeping it in the database means the API serves it without a
-- filesystem path escaping into a response.
CREATE TABLE run_kernel (
    submission_id INTEGER NOT NULL REFERENCES submission(id),
    problem_key   TEXT NOT NULL,
    source        TEXT NOT NULL,
    n_lines       INTEGER,
    sha256        TEXT,
    -- A submitted kernel whose authoritative re-time never completed produces
    -- NO result rows, so the board renders it as "not attempted" -- identical
    -- to a problem the agent never opened. glm-run1 has exactly one:
    -- FlashInfer-Bench__014, whose re-time hit `TimeoutExpired` after 1200s.
    -- The kernel exists and the attempt was real; only the measurement is
    -- missing, and that distinction is the whole difference between "did not
    -- try" and "tried, and the harness could not measure it".
    retime_ok     INTEGER,
    retime_error  TEXT,
    PRIMARY KEY (submission_id, problem_key)
);

-- The other side of the diff: the T_b formulation this kernel had to beat.
-- Not stored per run -- it is a property of the problem. Regenerated by
-- applying `reference/tb-candidates/variants.py` to the problem's own
-- reference source, which is exactly how task 06 produced the thing it timed,
-- so the diff shown is against the code that was actually measured.
CREATE TABLE variant_source (
    problem_key TEXT NOT NULL,
    variant     TEXT NOT NULL,
    source      TEXT NOT NULL,
    n_lines     INTEGER,
    PRIMARY KEY (problem_key, variant)
);

-- Cost and effort, per problem. The board only ever showed a run-level total,
-- which hides that one problem took $69 and 188 turns while another took $25.
CREATE TABLE run_effort (
    submission_id      INTEGER NOT NULL REFERENCES submission(id),
    problem_key        TEXT NOT NULL,
    cost_usd           REAL,
    wall_seconds       REAL,
    api_seconds        REAL,
    turns              INTEGER,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    cache_write_tokens INTEGER,
    cache_read_tokens  INTEGER,
    harness_evals      INTEGER,
    kernel_changed     INTEGER,
    capped             INTEGER,
    timed_out          INTEGER,
    gpu                TEXT,
    PRIMARY KEY (submission_id, problem_key)
);

-- One row per harness evaluation the agent ran, in order. This is the record
-- of HOW the endpoint was reached: the regressions are in here as much as the
-- gains, and both matter. `mean_score` is on the SOL scale, so a trajectory is
-- directly comparable to the score the submission ended up with.
CREATE TABLE trajectory_eval (
    submission_id   INTEGER NOT NULL REFERENCES submission(id),
    problem_key     TEXT NOT NULL,
    n               INTEGER NOT NULL,      -- 1-based, ordered by timestamp
    utc             TEXT,
    minutes_in      REAL,                  -- since the first eval of this problem
    ok              INTEGER,               -- did the harness itself run
    all_passed      INTEGER,
    passed          INTEGER,
    workloads       INTEGER,
    geomean_speedup REAL,
    mean_score      REAL,
    kernel_sha      TEXT,                  -- snapshot identity; NULL if not kept
    kernel_source   TEXT,
    kernel_lines    INTEGER,
    PRIMARY KEY (submission_id, problem_key, n)
);

-- Transcripts stay on DISK: 2 MB each, and they carry gateway keys and
-- internal hostnames in their provenance. Only the path and a summary are
-- indexed; `/api/v1/.../transcript` streams the file after checking the path
-- against this table, so no request can name a file the ingest did not.
CREATE TABLE transcript (
    submission_id INTEGER NOT NULL REFERENCES submission(id),
    problem_key   TEXT NOT NULL,
    path          TEXT NOT NULL,
    bytes         INTEGER,
    n_lines       INTEGER,
    n_turns       INTEGER,
    tools_json    TEXT,
    PRIMARY KEY (submission_id, problem_key)
);
