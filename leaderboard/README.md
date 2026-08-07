# SOL-ExecBench-ROCm leaderboard

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
| database | SQLite, `leaderboard/db/solbench-<PART>.db` | Rebuildable view over `artifacts/`. **One file per part** — see below. Gitignored. |
| backend | FastAPI + uvicorn, `app.py` | Server-rendered pages plus a typed JSON API under `/api/v1/`. |
| contract | `models.py`, pydantic | Every `/api/v1` route declares a response model, so `/openapi.json` is a real schema and a client can be generated from it. |
| frontend | Jinja2 + one stylesheet | No build step, no npm. The node has no node. |
| write path | `submit.py` + `queue.db` | Queue only. Accepting a submission never runs anything. |
| worker | `worker.py`, GPU 0, one job at a time | Re-times through `scripts/agent_score.py`, so there is exactly one scorer. |
| venv | `leaderboard/.venv`, host python 3.11 | **Not** the pinned measurement container. |

That last row is the load-bearing one. Adding fastapi to
`solbench:rocm7.2-torch2.9.1` would change the environment every baseline in
this repo was measured under, which prime directive 6 forbids. The web app
never touches a GPU, so it has no business in that image.

## One database per part

A score measured on MI350X and one measured on MI355X are **not comparable**:
different power cap, different sustained clock, so a different F_LOCK and
therefore a different T_SOL and T_b. So the part is not a filter over one
dataset — it selects the dataset, and the split is enforced by the filesystem
so that no query can mix two parts by accident:

    leaderboard/db/solbench-MI350X.db     <- the measured board
    leaderboard/db/solbench-MI355X.db     <- does not exist, and must not be faked

`ingest.py` names the file after the part the **manifest** says it was measured
on. `--part` *asserts* that value and errors if it disagrees; it cannot relabel
a build, because `solbench-MI355X.db` full of MI350X timings is exactly the
failure the split exists to prevent.

Which part a request is about resolves in this order: `?part=` → the `part`
cookie → `SOLBENCH_PART` → the only part with a database → `MI350X`. An unknown
name is a 400. A part the port targets that has no database is a first-class
page saying nothing has been measured on it — not a 404, not an empty table —
and `/api/v1` answers 503 there. `run.sh` prints the same fact at startup:

```
serving MI355X, which has no database -- the board will show its empty state.
```

`SOLBENCH_DB` still pins one file and collapses the board to it; `run.sh` and
`worker.py` build *that* file when it is set, so the pin means one thing
everywhere. `leaderboard/solbench.db`, the pre-split single-file layout, is
still read if a leftover is there, but nothing writes it any more.

## Rebuilding

```bash
leaderboard/.venv/bin/python leaderboard/ingest.py     # -> db/solbench-<PART>.db
```

`ingest.py` builds a **new** database beside the old one and `os.replace()`s it
in. It used to delete the live file and rebuild in place, which gave readers a
window -- measured at 0.30s, and caught serving `rows=0` on the first attempt --
where the API returned 200 with an empty leaderboard. Not an error a client
could detect: indistinguishable from "nobody has submitted".

### Runs kept outside the repo

List them once, in `leaderboard/sources.json` (machine-local and gitignored;
copy `sources.json.example`), which `ingest.py` reads **by default**:

```json
{"agent_run_roots": ["/path/to/an/out-of-repo/run-collection"]}
```

`--agent-runs` still works and *overrides* the config; `SOLEXBENCH_AGENT_RUNS`
adds to whichever won. Every root is printed with where it came from, because
"which roots did this build read" is the question behind every instance of this
going wrong.

STATE.md D24 records three instances of this, always the same way: something
shelled a bare `ingest.py`, the out-of-repo runs were not in it, and the board
came back smaller with nothing saying so. The fix for the second one — a
drop-guard in `worker.py` — then quietly stopped working when the database moved
to `db/solbench-<PART>.db`, because the guard was still reading
`leaderboard/solbench.db`: it compared an empty set against an empty set and
reported success.

Patching each caller does not fix it; the next caller has not been written yet.
So ingest itself now **refuses to publish a board with fewer submissions than
the one it replaces**, leaving the existing board untouched and naming the roots
the two builds disagree about. `--allow-drop` is how you retire a run on
purpose. `run.sh` and `worker.py` ask `app.py` which file is served rather than
spelling a path, so there is one notion of "the database"; and because the
config exists, `worker.py` no longer passes `--agent-runs` at all.

`ingest.py` recreates every table from:

* `artifacts/09/manifest-v1.json` — problems, workloads, T_SOL, T_b, tolerances
* `data/SOL-ExecBench/benchmark/*/*/definition.json` — descriptions, axes, reference source
* `artifacts/06/candidates/` and `artifacts/06/authoritative/` — the four PyTorch variants
* `artifacts/10/*/scored.json` — agent runs
* `artifacts/10/*/kernels/*.py` — the kernel each submission proposed
* `artifacts/10/*/trajectory/<problem>/eval-*.json` — every harness eval, with its kernel snapshot
* `artifacts/10/*/cost-report.json` — per-problem cost, turns, tokens
* `artifacts/10/*/transcripts/*.jsonl` — indexed by path only; see below
* `reference/tb-candidates/variants.py` — applied to each problem's reference to
  regenerate the T_b formulations, so the run page can show what a kernel had to beat

Depth is uneven by harness and that is recorded, not hidden:
`submission.depth_note` says what a given run did not write down, so an empty
panel explains itself instead of looking broken.

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

* **One part per board.** Every score on the built board is `MI350X` at F_LOCK
  1300 MHz, and the header says so. T_SOL in milliseconds, T_b and F_LOCK all
  differ by part, so a score from another part cannot be ranked against these —
  `scripts/score_solutions.py` refuses that comparison rather than rescaling it.
  The part switch selects a whole database (above); it never joins two. MI355X
  is in the switch and has **no** database, because nothing has been measured on
  it — the port needs no work there, every number does. See `TODO-MI355X.md`.
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


## Submitting

```bash
# one token per line, `token:name`. No file => the write API is 503, by design.
printf 'S3CRET:william\n' > leaderboard/.tokens && chmod 600 leaderboard/.tokens

curl -X POST http://127.0.0.1:8088/api/v1/submit \
  -H 'Authorization: Bearer S3CRET' -H 'Content-Type: application/json' \
  -d '{"slug":"my-run","problem_key":"L1__085_geglu_activation","kernel":"import torch\ndef run(...):..."}'
# -> 202 with a job id

curl http://127.0.0.1:8088/api/v1/jobs/1        # poll

leaderboard/.venv/bin/python leaderboard/worker.py --drain   # score the queue
```

A slug groups jobs into **one** leaderboard entry: submitting a second problem
under the same slug extends the entry rather than creating a rival to it.

`202`, not `200`. Nothing has been measured when the POST returns.

### The trust boundary

Scoring means executing code somebody else wrote, on a GPU, in this repo's
container. **`env/solb` is a reproducibility boundary, not a security one** — it
runs as a normal user with the repo bind-mounted read-write and no seccomp
profile or network namespace of its own. A submitted kernel can read and write
the tree.

So: **authenticated internal users, and nobody else.** The token gates who can
queue work; nothing gates what queued code does once it runs. That is fine for
a team-internal service and is not fine for a public one, and the difference is
a sandbox that does not exist yet. Do not expose the port.

### Why one job at a time

Every number on this board is timed on GPU 0 with nothing else on it. That is
what makes T_b, T_SOL and every submitted score comparable. Two workers, or one
worker running two jobs, would produce numbers that are not comparable to the
3717 already published. A lock file enforces it; a second worker exits rather
than share the GPU. When the queue backs up, the answer is to wait.

## Transcripts

Served from disk by `/api/v1/submissions/{slug}/problems/{key}/transcript`,
which looks the path up in the `transcript` table — never takes it from the
request. They are 2 MB each and their provenance records carry gateway
hostnames and API-key prefixes, so they are gitignored and internal-only.

## Tests

```bash
leaderboard/.venv/bin/python -m pytest tests/leaderboard -q
```

They skip in the measurement container (no fastapi there, deliberately), so
`env/solb pytest tests/` stays green. They cover the invariants that have
actually broken: the atomic rebuild, no score on a failed workload, untested vs
failed, the write API failing closed, the GPU-0 lock, and the regression
classifier not inventing a noise threshold.
