# SOL-ExecBench-AMD leaderboard

A local leaderboard for the AMD port: rankings, a problem index, and a
**per-problem subpage** showing the scoring bounds, tolerances and every
submission's result on each workload.

```bash
leaderboard/run.sh            # http://127.0.0.1:8088
PORT=9000 leaderboard/run.sh
```

First run creates the venv and builds the database; after that it just serves.

## What it is

| layer | choice | why |
|---|---|---|
| database | SQLite, `leaderboard/solbench.db` | Rebuildable view over `artifacts/`. Gitignored. |
| backend | FastAPI + uvicorn, `app.py` | Server-rendered pages plus a JSON API under `/api/`. |
| frontend | Jinja2 + one stylesheet | No build step, no npm. The node has no node. |
| venv | `leaderboard/.venv`, host python 3.11 | **Not** the pinned measurement container. |

That last row is the load-bearing one. Adding fastapi to
`solbench:rocm7.2-torch2.9.1` would change the environment every baseline in
this repo was measured under, which prime directive 6 forbids. The web app
never touches a GPU, so it has no business in that image.

## Rebuilding

```bash
leaderboard/.venv/bin/python leaderboard/ingest.py
```

`ingest.py` drops and recreates every table from:

* `artifacts/09/manifest-v1.json` — problems, workloads, T_SOL, T_b, tolerances
* `data/SOL-ExecBench/benchmark/*/*/definition.json` — descriptions, axes, reference source
* `artifacts/06/candidates/` and `artifacts/06/authoritative/` — the four PyTorch variants
* `artifacts/10/*/scored.json` — agent runs

The database is never a source of truth. If it disagrees with the artifacts,
it is stale — rerun the ingest. Scores are computed by importing the repo's own
`sol_execbench.sol_score`, not by reimplementing the formula, so the board
cannot drift from the harness that produced the numbers.

## Two rankings, both shown

**Benchmark score** sums per-workload scores across *every* scoreable workload
and counts anything not passed as zero. It is the headline because it cannot be
improved by attempting fewer problems. On the current board `torch.compile` has
the best mean score of any variant (0.491) and ranks third, because it only
passes 69.6% of workloads.

**Mean (attempted)** is the average over what a submission actually passed. The
UI never renders it without coverage beside it; alone, it pays a submission for
skipping the hard problems.

## Things the board is careful about

* **Reference variants are labelled as such.** The four PyTorch formulations are
  what T_b is *derived from* — the winner of each workload scores exactly 0.5
  there by construction. They calibrate the scale rather than compete on it.
* **Deferred ≠ missing.** The 15 NVFP4 problems appear, marked, with the reason,
  and are counted in every denominator.
* **Entries that produced nothing are recorded, not dropped.**
  `v5_compile_contiguous` ran on all 235 problems and passed zero workloads;
  it is excluded from the rankings and listed on the methodology page with why.
* **Flagged submissions stay visible.** A workload the anti-reward-hack checks
  rejected scores zero and is still displayed as flagged.

## API

| endpoint | returns |
|---|---|
| `GET /api/stats` | manifest provenance, per-category problem counts |
| `GET /api/leaderboard?category=L1` | ranked submissions |
| `GET /api/problems?category=` | problem index |
| `GET /api/problems/{key}` | one problem: bounds, tolerances, per-workload results |
| `GET /api/submissions/{slug}` | one submission, broken down per problem |
| `GET /healthz` | liveness + database path |

Interactive docs at `/api/docs`.

## Note on SQLite

The system SQLite here is 3.26, which predates `FILTER` and `NULLS LAST`
(both 3.30). Those parse as syntax errors rather than being ignored, so the
queries use `CASE WHEN` aggregates and `ORDER BY (x IS NULL), x` instead.
