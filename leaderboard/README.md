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

* **One part per board.** Every score is `MI350X` at F_LOCK 1300 MHz, and the
  header says so. T_SOL in milliseconds, T_b and F_LOCK all differ by part, so
  a score from another part cannot be ranked against these —
  `scripts/score_solutions.py` refuses that comparison rather than rescaling it.
  The MI355X port on `feat/agent-scoreboard` is deliberately **not** merged in:
  its T_b is not anchor-verified, its agent runs are scored on headroom and
  speedup rather than S, and it has no S to publish. It needs its own board.
* **Reference variants are labelled as such.** The four PyTorch formulations are
  what T_b is *derived from* — the winner of each workload scores exactly 0.5
  there by construction. They calibrate the scale rather than compete on it.
* **Deferred ≠ missing.** The 15 NVFP4 problems appear, marked, and each one
  prints the mechanism and the interpreter's own error next to its `0`. A
  documented decision that renders as a bare zero is indistinguishable from a
  sweep that never ran, so `ingest.py` now *raises* if the manifest defers a
  problem that `artifacts/deferred.json` cannot explain.
* **Entries that produced nothing are recorded, not dropped.**
  `v5_compile_contiguous` ran on all 235 problems and passed zero workloads;
  the `pilot8` agent run was a smoke test whose $8 cap stopped all eight
  sessions mid-work. Both are off the rankings and listed on the methodology
  page with the reason.
* **Flagged submissions stay visible.** A workload the anti-reward-hack checks
  rejected scores zero and is still displayed as flagged.
* **Staleness is reported, not assumed away.** The database is a view over the
  artifacts and goes stale the moment they move — silently, since every page
  still renders. `ingest.py` stamps the repo SHA and build time; `app.py`
  compares them against the working tree and puts a banner in the header when
  they diverge. `/healthz` returns the same check as JSON.

## Themes

Dark by default, light on request, toggled from the header and remembered in
`localStorage`; with nothing stored it follows `prefers-color-scheme`. The
theme is applied by an inline script in `<head>` before first paint, because
reading it later renders the page dark and then flashes to light. Every colour
is a CSS variable declared once per theme, so no component carries a
theme-specific override — `style.css` is the only place either palette lives.

## API

| endpoint | returns |
|---|---|
| `GET /api/stats` | manifest provenance, per-category problem counts |
| `GET /api/leaderboard?category=L1` | ranked submissions |
| `GET /api/problems?category=` | problem index |
| `GET /api/problems/{key}` | one problem: bounds, tolerances, per-workload results |
| `GET /api/submissions/{slug}` | one submission, broken down per problem |
| `GET /healthz` | liveness, part, and whether the database has gone stale |

Interactive docs at `/api/docs`.

## Note on SQLite

The system SQLite here is 3.26, which predates `FILTER` and `NULLS LAST`
(both 3.30). Those parse as syntax errors rather than being ignored, so the
queries use `CASE WHEN` aggregates and `ORDER BY (x IS NULL), x` instead.
