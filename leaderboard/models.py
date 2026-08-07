#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Response models for `/api/v1`.

Why these exist at all: before this, every route returned a bare `dict`, so
`/openapi.json` documented thirteen endpoints with **zero response schemas**.
That is the specific thing that blocks running a separate frontend against this
service -- there was no contract to generate a client from and no way for a
field rename to fail anything. A consumer could only discover the shape by
calling the endpoint and reading the JSON.

Two rules held throughout:

* **Optional means "may be absent", not "defaults to zero".** Almost every
  numeric field here is `float | None`, because a missing measurement is not a
  small one. Coercing `t_b_ms` to 0.0 would make a workload that was never
  timed look infinitely fast, and no downstream consumer could tell.
* **The model is descriptive, not aspirational.** Fields exist here because a
  route returns them today. Adding a field the ingest does not populate would
  put a schema in front of consumers promising data that is not there.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Freshness(BaseModel):
    stale: bool
    reasons: list[str] = []
    db_built_utc: str | None = None
    built_from_git_sha: str | None = None
    rebuild_command: str | None = None
    inputs: dict | None = None
    error: str | None = None


class Meta(BaseModel):
    manifest_version: str | None = None
    methodology: str | None = None
    part: str | None = None
    device: str | None = None
    f_lock_mhz: str | None = None
    rocm_version: str | None = None
    torch_version: str | None = None
    total_problems: str | None = None
    scoreable_problems: str | None = None
    scoreable_workloads: str | None = None
    git_sha: str | None = None
    freshness: Freshness | None = None

    model_config = {"extra": "allow"}   # meta is a key/value table; do not truncate it


class CategoryCount(BaseModel):
    category: str
    n: int
    deferred: int


class Stats(BaseModel):
    meta: dict
    by_category: list[CategoryCount]


class LeaderboardRow(BaseModel):
    rank: int
    slug: str
    name: str
    kind: str
    model: str | None = None
    author: str | None = None
    created_utc: str | None = None
    notes: str | None = None
    cost_usd: float | None = None
    wall_seconds: float | None = None
    gpu: str | None = None
    depth_note: str | None = None

    benchmark_score: float = Field(
        description="Sum of per-workload scores over EVERY scoreable workload, "
                    "divided by all of them. Failed and never-attempted both "
                    "contribute zero.")
    mean_score_attempted: float = Field(
        description="Scores summed over the workloads this submission "
                    "attempted. A failed attempt counts as zero; it does not "
                    "leave the denominator.")
    mean_score_passed: float = Field(
        description="Mean over passes only. Provided for comparison, NOT for "
                    "ranking: it can be raised by attempting less.")
    coverage: float
    coverage_attempted: float = Field(
        default=0.0,
        description="Passes over ATTEMPTS rather than over the whole "
                    "benchmark. The pass rate of what was actually run.")
    partial: bool = Field(
        description="True when the submission was not run on every scoreable "
                    "workload, so benchmark_score is a floor.")
    workloads_total: int
    workloads_passed: int
    workloads_attempted: int
    workloads_failed: int
    workloads_untested: int
    problems_total: int
    problems_complete: int
    problems_attempted: int = Field(
        default=0,
        description="Problems this submission has at least one result on. "
                    "`problems_complete` is the subset it swept clean.")
    # The four states the board's coverage bar draws. Disjoint, and they sum to
    # problems_total -- which is the property that makes the bar readable and
    # the one to assert if these are ever recomputed.
    problems_clean: int = Field(
        default=0, description="Every scoreable workload passed.")
    problems_partial: int = Field(
        default=0, description="Some workloads passed, some did not.")
    problems_failed: int = Field(
        default=0, description="Attempted, and no workload passed. NOT the "
                               "same as never attempted, which is the next "
                               "field.")
    problems_untouched: int = Field(
        default=0, description="No result of any kind. Never run on it.")
    rank: int = Field(
        default=0,
        description="Rank under the full-benchmark scope (`benchmark_score`).")
    rank_attempted: int = Field(
        default=0,
        description="Rank under the attempted scope (`mean_score_attempted`). "
                    "This is the board's default order, and it is NOT "
                    "comparable across rows with different coverage.")
    n_flagged: int


class ProblemSummary(BaseModel):
    key: str
    category: str
    name: str
    description: str | None = None
    hf_id: str | None = None
    n_workloads: int
    n_scoreable: int
    deferred: int
    deferred_reason: str | None = None
    deferred_mechanism: str | None = None
    deferred_error: str | None = None
    median_headroom: float | None = None
    # Both are headline figures and both count RANKED runs only: a submission
    # with `board_visible = 0` is evidence that may be read, not a number that
    # may sit at the top of a page. It is still listed, flagged, in
    # `ProblemDetail.submissions` and in each workload's `results`.
    best_score: float | None = Field(
        None, description="Highest score any board-visible submission reached "
                          "on this problem. None where none has passed one of "
                          "its workloads. Excludes board_visible=0 runs.")
    n_submissions: int | None = Field(
        None, description="Board-visible submissions with at least one pass "
                          "here. Excludes board_visible=0 runs.")


class WorkloadResult(BaseModel):
    slug: str
    submission_name: str
    kind: str
    status: str | None = None
    latency_ms: float | None = None
    score: float | None = None
    flagged: int = 0
    note: str | None = None
    board_visible: int
    exclusion_reason: str | None = None


class WorkloadBound(BaseModel):
    uuid: str
    axes: dict = {}
    scoreable: int
    t_sol_cycles: int | None = None
    t_sol_ms: float | None = None
    t_sol_source: str | None = None
    sol_bottleneck: str | None = None
    t_b_ms: float | None = None
    t_b_variant: str | None = None
    headroom: float | None = None
    tol_atol: float | None = None
    tol_rtol: float | None = None
    results: list[WorkloadResult] = []


class SubmissionOnProblem(BaseModel):
    slug: str
    name: str
    kind: str
    model: str | None = None
    passed: int | None = None
    mean_score: float | None = None
    best_latency: float | None = None
    flagged: int | None = None
    # NOT NULL in the schema and reached through an inner join, so it is an
    # int and never absent. It is here because a problem page shows every run
    # that touched the problem, including ones the board does not rank: a
    # score with no way to tell which it is invites the reader to rank it
    # themselves.
    board_visible: int
    exclusion_reason: str | None = None


class ProblemDetail(BaseModel):
    problem: ProblemSummary
    workloads: list[WorkloadBound]
    submissions: list[SubmissionOnProblem]


class SubmissionProblemRow(BaseModel):
    key: str
    category: str
    name: str
    n_scoreable: int
    passed: int | None = None
    attempted: int | None = None
    mean_score: float | None = None
    flagged: int | None = None
    cost_usd: float | None = None
    wall_seconds: float | None = None
    turns: int | None = None
    harness_evals: int | None = None


class Trial(BaseModel):
    """One run of a setup that was run more than once.

    A trial is a whole submission -- its own kernels, trajectory and cost --
    under a different constraint, so the outcome fields below are the outcome
    on **the problem in context**. On a page with no single problem in context
    they are None rather than aggregates over the run: a trial's overall score
    is a different quantity and putting it in the same field would let the two
    be compared as if they were one.
    """

    slug: str
    name: str
    trial_label: str | None = Field(
        None, description="The constraint that distinguishes this trial, read "
                          "from the run's own artifact ('$8 / problem'). None "
                          "where the harness recorded no constraint.")
    trial_n: int | None = None
    constraint: dict = Field(
        {}, description="The constraint as the artifact recorded it. A key is "
                        "absent, not null, where nothing wrote it down.")
    board_visible: int
    exclusion_reason: str | None = None
    is_current: bool
    url: str
    touched: bool | None = Field(
        None, description="Whether this trial worked on the problem in "
                          "context. True with no results means the kernel "
                          "exists and its measurement failed (D23), which is "
                          "not the same as never opening the problem. None "
                          "where there is no problem in context.")
    attempted: int | None = Field(
        None, description="Scoreable workloads of the problem in context this "
                          "trial attempted. None where there is no problem in "
                          "context.")
    passed: int | None = None
    mean_score: float | None = Field(
        None, description="Score summed over the workloads in `attempted`, "
                          "divided by them. A failed attempt and a workload "
                          "whose bound is invalid each contribute zero rather "
                          "than leaving the denominator -- the same quantity "
                          "as RunSummary.mean_attempted, because the switcher "
                          "and the run card render side by side and two "
                          "numbers under one label is a contradiction, not a "
                          "nuance.")


class SubmissionDetail(BaseModel):
    submission: dict
    problems: list[SubmissionProblemRow]
    trials: list[Trial] = Field(
        [], description="Every trial of this submission's setup, this one "
                        "included. Empty when the run is not part of a group.")


# ----------------------------------------------------------------- run detail

class RunWorkload(BaseModel):
    uuid: str
    axes: dict = {}
    scoreable: int
    attempted: bool
    passed: bool
    status: str | None = None
    t_sol_ms: float | None = None
    t_sol_source: str | None = None
    sol_bottleneck: str | None = None
    t_b_ms: float | None = None
    t_b_variant: str | None = None
    latency_ms: float | None = None
    score: float | None = None
    speedup_vs_tb: float | None = None
    headroom: float | None = None
    # None, not 0. These rows come from `workload LEFT JOIN result`, so a
    # workload this submission never attempted has no result row and `flagged`
    # is NULL -- and NULL is the honest value: the reward-hack check did not
    # run, which is a different statement from "it ran and found nothing".
    # Declaring this `int = 0` was the first thing the response model caught,
    # on 661 of 3343 endpoints. The bug was mine and the schema found it before
    # any consumer did, which is the entire argument for having one.
    flagged: int | None = None
    note: str | None = None
    bound_invalid: bool = False
    unmeasured: bool = Field(
        False, description="This workload was not attempted AND the submission "
                           "has a kernel for the problem whose authoritative "
                           "re-time did not complete (run_kernel.retime_ok=0). "
                           "'Tried, could not be measured' is not 'did not "
                           "try', and the grid draws them differently (D23).")


class RunSummary(BaseModel):
    n_scoreable: int
    attempted: int
    passed: int
    failed: int
    untested: int
    mean_attempted: float | None = None
    mean_passed: float | None = None
    problem_score: float | None = None
    best: float | None = None
    n_flagged: int


class RunKernel(BaseModel):
    source: str
    n_lines: int | None = None
    sha256: str | None = None
    retime_ok: int | None = None
    retime_error: str | None = None


class VariantSource(BaseModel):
    variant: str
    source: str
    n_lines: int | None = None
    won_workloads: int = Field(
        0, description="How many of this problem's workloads this variant won "
                       "T_b on. The anchor is per workload, so more than one "
                       "variant can be 'the' baseline within a single problem.")


class TrajectoryEval(BaseModel):
    n: int
    utc: str | None = None
    minutes_in: float | None = None
    ok: int | None = None
    all_passed: int | None = None
    passed: int | None = None
    workloads: int | None = None
    geomean_speedup: float | None = None
    mean_score: float | None = None
    kernel_sha: str | None = None
    kernel_lines: int | None = None
    regression: bool = Field(
        False, description="Correctness only: this eval passed fewer workloads "
                           "than an earlier one. Score movement is reported as "
                           "delta_vs_best instead, with no verdict attached — "
                           "there is no measured noise floor for an agent-side "
                           "eval, so calling a 0.2% dip a regression would be "
                           "an invented threshold.")
    harness_error: bool = Field(
        False, description="The eval harness did not run (workloads == 0). "
                           "Nothing was measured, so this is neither a gain "
                           "nor a regression.")
    delta_vs_best: float | None = Field(
        None, description="mean_score minus the best mean_score seen earlier "
                          "in this problem. Negative is a dip; judge the size "
                          "yourself.")
    # `minutes_in` and `utc` are both nullable, and both render as data if a
    # template touches them raw: `minutes_in or 0` invents "+0 min", and a NULL
    # `utc` interpolates as the string "None". The rendered forms are supplied
    # here so that absence stays absent -- None, never a placeholder, because
    # substituting a dash is a display choice and printing "+0m" is a claim.
    at_label: str | None = Field(
        None, description="`minutes_in` rendered ('+37m'). None where the eval "
                          "carries no elapsed time.")
    utc_label: str | None = Field(
        None, description="`utc` rendered as a labelled wall clock ('14:22 "
                          "UTC'). None where the eval carries no timestamp.")


class RunEffort(BaseModel):
    cost_usd: float | None = None
    wall_seconds: float | None = None
    api_seconds: float | None = None
    turns: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None
    harness_evals: int | None = None
    kernel_changed: int | None = None
    capped: int | None = None
    timed_out: int | None = None
    gpu: str | None = None


class TranscriptInfo(BaseModel):
    bytes: int | None = None
    n_lines: int | None = None
    n_turns: int | None = None
    tools: dict = {}
    url: str


class RunWindow(BaseModel):
    """When a run worked on a problem, and by whose clock.

    `source` is required and must be printed wherever the times are: no
    artifact records a session start, so every window here is a proxy, and
    which proxy changes what the number means. `first_last_eval` is a window
    *inside* the session; `retime_only` is when the kernel was scored on GPU 0,
    which is not when it was worked on.

    A run with no timestamp evidence has no `RunWindow` at all. There is no
    `unknown` source, because absence already says it and a value would invite
    a caller to render a blank window as a real one.
    """

    started_utc: str | None = None
    finished_utc: str | None = None
    elapsed_seconds: float | None = Field(
        None, description="finished - started. None whenever either end is "
                          "missing; a one-ended window has no duration.")
    source: str = Field(
        description="One of first_last_eval, session, retime_only. Enforced by "
                    "a CHECK constraint in the schema, not just documented.")


class RunDetail(BaseModel):
    submission: dict
    problem: dict
    summary: RunSummary
    workloads: list[RunWorkload]
    peers: list[SubmissionOnProblem]
    trials: list[Trial] = Field(
        [], description="Every trial of this submission's setup and its "
                        "outcome on THIS problem. Empty when the run is not "
                        "part of a group.")
    window: RunWindow | None = None
    part: str | None = Field(
        None, description="The part this RUN was measured on, from its own "
                          "re-time provenance. None where its artifacts named "
                          "none -- deliberately not filled in from the "
                          "database's part, which is a fact about the bounds.")
    kernel: RunKernel | None = None
    variants: list[VariantSource] = []
    reference: str | None = None
    trajectory: list[TrajectoryEval] = []
    effort: RunEffort | None = None
    transcript: TranscriptInfo | None = None
    depth_note: str | None = Field(
        None, description="What this harness did not record. Present when a "
                          "section is empty for a reason other than 'no data "
                          "exists yet'.")


class PartInfo(BaseModel):
    """One entry in the part switch.

    The part is not a filter over one dataset, it selects the dataset: MI350X
    and MI355X have different power caps and therefore different F_LOCK, T_SOL
    and T_b, so their scores are not comparable and never share a database.
    """

    name: str
    available: bool
    n_results: int | None = Field(
        None, description="Measured workload results on this part. None -- not "
                          "0 -- where there is no database: 'not measured' and "
                          "'measured nothing' are different statements.")
    active: bool = False
    url: str = Field(
        description="This same page on that part: path and query preserved, "
                    "?part= set.")


class PartMismatch(BaseModel):
    """A submission stored in a database whose part it was not measured on.

    Always empty in a correctly built database — `ingest.py` refuses the write
    — and checked again at query time because the ranking is where the mistake
    would do its damage: an MI355X T_k scored against MI350X bounds is a
    plausible number with nothing detectably wrong. Such a row is dropped from
    `leaderboard_rows()`, and this is what makes the drop visible; a row that
    silently disappears from a ranking is its own failure.
    """

    slug: str
    name: str
    submission_part: str


class Health(BaseModel):
    ok: bool = Field(
        description="False when the active part has no database, or when a "
                    "submission in it was measured on a different part.")
    db: str | None = Field(
        None, description="The database serving the active part. None when "
                          "that part has none.")
    part: str | None = None
    manifest_version: str | None = None
    freshness: Freshness | None = None
    error: str | None = None
    part_mismatch: list[PartMismatch] = Field(
        [], description="Submissions excluded from the ranking because their "
                        "own part disagrees with the database's. Non-empty "
                        "means the database needs rebuilding.")
