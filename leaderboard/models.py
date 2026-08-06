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
    best_score: float | None = None
    n_submissions: int | None = None


class WorkloadResult(BaseModel):
    slug: str
    submission_name: str
    kind: str
    status: str | None = None
    latency_ms: float | None = None
    score: float | None = None
    flagged: int = 0
    note: str | None = None


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


class SubmissionDetail(BaseModel):
    submission: dict
    problems: list[SubmissionProblemRow]


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


class RunDetail(BaseModel):
    submission: dict
    problem: dict
    summary: RunSummary
    workloads: list[RunWorkload]
    peers: list[SubmissionOnProblem]
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


class Health(BaseModel):
    ok: bool
    db: str
    part: str | None = None
    manifest_version: str | None = None
    freshness: Freshness | None = None
    error: str | None = None
