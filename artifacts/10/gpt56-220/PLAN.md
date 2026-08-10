# gpt-5.6-sol, full 220 — what it needs before it starts

Not started. This is the prep, written 2026-08-10 so the decision is a decision
and not an improvisation at launch time.

## What exists

`artifacts/10/gpt56-40/` covers **40 of the 220 scoreable problems**, mean
S = 0.6406 against manifest v1.2, 0 flagged, 1 bound violation
(`L2__045`). `remaining-problems.txt` beside this file is the other **180**:
L1 77, L2 67, FlashInfer-Bench 21, Quant 15. Verified against v1.2's scoreable
set — every one of the 40 is in it, so the two lists partition 220 exactly.

## Sizing, from the run that already happened

Measured from `run_window`, not estimated:

| | gpt-5.6-sol (40) | GLM-5.2-FP8 (220) |
|---|---|---|
| median per problem | **851 s** | 2718 s |
| p90 | 1715 s | 3435 s |
| GPU-hours | 11.4 | 142.5 |
| wall span | 1.84 h | 38.5 h |

**Use gpt-5.6-sol's own figure and not GLM's.** GLM hit the harness's 3600 s cap
on 168 of 220 sessions, so its median is the cap rather than the work; that run
is a floor, not a duration. gpt-5.6-sol's 851 s median is well clear of the cap
and is a real measurement.

At 0.285 GPU-hours per problem, **180 problems ≈ 51 GPU-hours**. On 7 cards that
is ~7.3 h wall; at the measured-safe 14 concurrent (2 per card, ±2.5% of an idle
card through the broker) it is ~3.7 h. The 40 that ran achieved an effective
concurrency of 6.2, so 7.3 h is the figure to plan against and 3.7 h is the
upside.

API volume: 4,347 calls for 40 problems, ~109 per problem, so **~19,600 calls**
for 180. `gpt56-40/scored.json` records `total_cost_usd: 0` — cost is not
tracked for this model, so there is no dollar estimate and inventing one would
be worse than saying that.

## The path, and why it is the awkward part

Both prior sweeps ran **through the amdpilot fleet**: J2 backfill jobs placed by
the GPU Scheduler, each in its own container with one MI350X, codex-cli against
the Model API front door.

There is no straightforward path around it. The front door at `127.0.0.1:7204`
answers `unauthorized: a valid per-job token is required`, and those tokens are
minted per job by the fleet. `scripts/agent_baseline.py` drives its own sessions
but is not wired to that front door. So a gpt-5.6-sol sweep means using the
scheduler that is under active development.

**What the scheduler's own state says, read-only, 2026-08-10 03:42 UTC:**

* Its active work has moved to the **MI355X** fleet — `mia1-p02-g05`, `g23`,
  `g46`. Last job anywhere: 02:25 UTC.
* **Nothing has touched `gbt350-odcdh1-a08-1` since 2026-08-09 10:23 UTC**, over
  17 hours.
* No job is running on any node right now.
* The GPU-0 reservation `h-a7197e` is **still active**, 19.7 h old, `owner=human`
  — which is what keeps authoritative timing safe and must not be released.

So a sweep placed on this node does not land where the development is. That is
an observation about right now, not a guarantee: the check to repeat before
launching is the same three queries against
`dash-overlay/gpu-scheduler/.state/gs.sqlite3` (copy the file, do not open it in
place).

## One thing to settle first: which timer measured the 40

`agent-gpt56-40`'s 40 problems were evaluated **before** the side-stream timing
fix (`STATE.md` D38). Merging them with 180 measured after it would put two
harness versions in one run.

Checked rather than assumed: **none of gpt56-40's 40 kernels constructs a
`torch.cuda.Stream`.** They cannot have been affected, so the merge is sound as
it stands.

Even so, re-timing them is cheap and removes the question entirely:

```
python scripts/agent_score.py --run artifacts/10/gpt56-40 --gpu 0 \
    --manifest artifacts/09/manifest-v1.2.json \
    $(sed 's/^/--retime /' artifacts/10/gpt56-220/remaining-problems.txt)   # no: see below
```

— that command is wrong on purpose, as a warning: `--retime` takes problems in
*this* run. To re-time all 40, drop `--reuse-retimed` entirely and let it
re-measure everything. Budget roughly an hour on GPU 0 and do it while the
agents are running on 1–7, which task 01 measured as costing −0.11%.

## After it finishes — three manual steps that look like a pipeline and are not

`STATE.md` D-list has this as an open defect: the fleet writes the leaderboard's
address into every job payload, which reads as though results publish
themselves. They do not.

```
sbt collect                                    # kernels out of ~/.jobd/jobs/<id>/
python scripts/agent_score.py --run artifacts/10/gpt56-220 --gpu 0 \
       --manifest artifacts/09/manifest-v1.2.json
python scripts/check_coverage.py --artifacts artifacts/10/gpt56-220
leaderboard/.venv/bin/python leaderboard/ingest.py
```

`check_coverage.py` is not optional. A sweep that dies partway and gets marked
done looks exactly like a sweep that finished, and this is the only thing that
says otherwise. A gap recorded in `artifacts/deferred.json` with a reason is a
decision; a gap without one is a bug.

## Open questions for a human

1. Use the fleet while it is under development, or wait? Its work is on MI355X
   and this node is idle to it, but that is today's state.
2. Merge into one `gpt56-220` run, or keep `gpt56-40` and add `gpt56-180` as a
   second row? Merging is what `scripts/merge_agent_runs.py` exists for and is
   what `glm-sweep-2` did.
3. Re-time the 40 anyway (~1 h on GPU 0) for a single-harness run, or rely on
   the check above that none of them uses a stream?
