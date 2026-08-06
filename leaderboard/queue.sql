-- SPDX-License-Identifier: Apache-2.0
--
-- The submission queue. A SEPARATE database from solbench.db, on purpose.
--
-- `solbench.db` is a derived view: `ingest.py` drops and rebuilds it from the
-- artifacts, and it is swapped in atomically. Anything written into it would
-- be destroyed by the next rebuild. The queue is the opposite kind of thing --
-- it is the only place a submission exists before it has been scored, so it is
-- durable state and it lives in its own file.
--
-- The scoring pipeline still produces artifacts, and those artifacts are still
-- the source of truth for the board. A job that finishes writes a run
-- directory exactly like a hand-placed one, and `ingest.py` picks it up with
-- no knowledge that a queue exists.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS job (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    token_name    TEXT NOT NULL,        -- who submitted, resolved from the token
    slug          TEXT NOT NULL,        -- the submission this job belongs to
    display_name  TEXT,
    model         TEXT,
    problem_key   TEXT NOT NULL,
    kernel_sha256 TEXT NOT NULL,
    kernel_bytes  INTEGER NOT NULL,
    notes         TEXT,

    -- queued -> running -> scored | failed
    -- `failed` means the pipeline could not produce a measurement. It is NOT
    -- "the kernel was slow" and it is not "the kernel was wrong": both of
    -- those are `scored`, with the answer in the artifacts.
    state         TEXT NOT NULL DEFAULT 'queued',
    error         TEXT,

    submitted_utc TEXT NOT NULL,
    started_utc   TEXT,
    finished_utc  TEXT,
    worker        TEXT,                 -- host:pid that claimed it
    gpu           INTEGER,

    -- Filled in from the run the worker produced. Denormalised on purpose: the
    -- queue must still answer "what happened to job 41" after the artifacts
    -- have been re-ingested, moved, or excluded from the board.
    n_workloads   INTEGER,
    n_passed      INTEGER,
    mean_score    REAL,
    run_dir       TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_state ON job(state, id);
CREATE INDEX IF NOT EXISTS idx_job_slug  ON job(slug);
