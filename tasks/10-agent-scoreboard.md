# Task 10 — Agent scoreboard

**Goal:** measure how well coding agents solve this benchmark on ROCm, and
report it in a way that cannot be misread.

Task 09 step 2 asks for "an agent baseline sweep" — one agent, one number, the
analogue of upstream's median-0.732. This task generalizes it to a comparison
across several (model, harness) pairs, which is a different measurement and needs
saying out loud:

> **A harness is not a model.** Claude Code and Codex differ in how many tool
> calls they will make, how they recover from a compile error, how much context
> they carry between attempts, and how long they persist before giving up. A
> difference in score is a difference between *agents*. It is not evidence about
> the underlying models compared in isolation, and must never be presented as a
> model benchmark.

## Preconditions

- Task 00 and task 01 **on this part**. Every latency the scoreboard reports is
  taken at F_LOCK, and F_LOCK does not transfer between MI350X and MI355X
  (prime directive 3).
- Task 02: the harness evaluates all 235 problems.
- Task 05: AMD-derived tolerances. Scoring a kernel correct-or-not by a B200
  tolerance is prime directive 2 in its most consequential form, so the sweep
  refuses to fall back to the shipped ones silently.
- Task 03 (T_SOL) and task 06 (T_b) are **not** preconditions. Without them the
  scoreboard reports a weaker basis and says which; see *Score bases* below.

## Steps

### 1. Sweep

```bash
python scripts/run_agents.py --run-id pilot-01 --limit-per-category 6
python scripts/run_agents.py --run-id full-01 --budget-usd 2000
```

One agent, one problem, one leased GPU, up to `--max-attempts` verification runs
against real hardware, under a wallclock cap. Resumable: a unit with a
`session.json` is done, and a *failed* unit still writes one, so a failure is a
result rather than a gap that is retried forever.

Sample the pilot **evenly** across each category rather than taking the first N.
Problem numbering follows the source model, so the first six of L1 are all from
one family and a pilot over them measures that family.

### 2. Score

```bash
python scripts/score_solutions.py --run-id pilot-01
```

Re-evaluates every harvested `solution.json` from the repo tree, serially, on the
authoritative GPU. Nothing an agent produced is trusted to score itself:

- `./verify` ran on a pool GPU with busy neighbours; authoritative timing is
  pinned to one GPU and every timing artifact records which.
- The agent runs as a local process with a writable filesystem, so the harness
  fingerprint taken at sweep start is compared at scoring time. A mismatch is
  *reported*, not judged — an operator edit looks the same as tampering.
- Attempts are capped, so the last thing an agent ran is often not the last
  thing it wrote.

### 3. Strengthen the basis when the bounds land

```bash
python scripts/backfill_scores.py --run-id pilot-01 \
    --manifest artifacts/09/manifest-MI355X-v1.json
```

Recomputes `S`, the headroom and the basis from a newer manifest **without
re-running anything on a GPU**. `T_k`, `T_ref` and the pass/fail verdicts were
measured on the authoritative GPU at F_LOCK and stay valid; only the bounds
changed. Re-scoring instead would spend hours reproducing the same latencies, and
would produce them on a different day under different node conditions — so a
record's basis and its timing would come from different runs.

Each record keeps its previous basis in `score_basis_history`, so a strengthened
score is visibly strengthened, and a **retracted** one is visibly retracted.
That matters: session 3 published 98 `sol_score_v1` records and then rolled them
back when the anchor check failed (blocker B2), and the history is the only reason
that is legible after the fact.

### 4. Publish

```bash
python scripts/build_scoreboard.py --all-runs
```

**Do not run a sweep, a shard sweep, or an agent run while measuring `T_b` or
verifying the anchor.** The GPUs do not interfere — task 01 measured +0.02% — but
Triton autotuning and `torch.compile` are CPU-bound, and seven concurrent agents
will starve a timing run on the authoritative GPU. Session 3 lost a `T_b`
measurement exactly this way: re-running the identical variant came out 5×
slower than the recorded anchor, and the whole score scale had to be voided.

Writes `artifacts/10/scoreboard.json` and a self-contained
`artifacts/10/dashboard.html` — no CDN, because the likely reader opens it over a
forwarded port or copies it off the node.

## Score bases

Every record carries `score_basis`, and **records of different bases are never
averaged together**. Pooling them produces a mean that moves when the *bounds*
land rather than when the kernels improve, and nothing in the output would say
which happened.

| basis | needs | what it means |
|---|---|---|
| `correctness_only` | tolerances | passed or not. No timing claim. |
| `speedup_vs_reference` | + a reference timing | T_ref / T_k. **Not** the SOL score: the reference is whatever the dataset shipped, while T_b is an optimized-PyTorch variant selected by a sweep. |
| `sol_headroom` | + T_SOL for this part | `(T_ref − T_k)/(T_ref − T_SOL)`, the share of the gap closed. Upstream reports S correlating with this at r = 0.981. |
| `sol_score_v1` | + T_b | `S = 1/(1 + (T_k − T_SOL)/(T_b − T_SOL))`. The real thing. |

A kernel timed faster than its own T_SOL is surfaced as a `bound_violation`
rather than clamped. S > 1 is impossible, so it means the bound is too loose —
which is worth fixing and would be hidden by clamping (cf. deviation D12, where
eight workloads had T_SOL truncated to zero cycles).

## What is measured besides the score

Cost and wallclock, per problem, from the harness's own accounting. Not
decoration: "which agent solves more" is unanswerable on its own, because an
agent given twice the budget should solve more. The scoreboard therefore reports
`$/solved` and `min/solved` beside the raw count.

Where a harness reports no cost — Codex reports tokens only — the field is
**null**, not a token count multiplied by a guessed rate. A guessed rate in a
currency column looks exactly like a measurement.

Two more columns exist to stop the headline being misread:

- **Reference copies.** Resubmitting the reference is correct, so it lifts the
  pass rate without demonstrating any kernel work, and lands near the anchor by
  construction. Detected by AST comparison (so comments and formatting do not
  disguise it), labelled, not penalized.
- **Failure stage.** `no_solution` and `invalid_solution` are harness or budget
  outcomes; `compile_error` and `incorrect_numerical` are the model's. Merging
  them would attribute a timeout to the model.

## Anti-gaming

The static source screen from task 08 runs on every submission before it is
scored, and a submission that trips it scores zero — it is not a slow kernel, it
is a kernel measured under conditions the benchmark does not permit. The runtime
guards (monkey-patch detection, stream policy, smi lockout, lazy-output
detection) are the harness's own and apply unchanged.

Containment of the agent is **not** claimed. It runs as a local process and could
write anywhere. The defensible position is that scoring re-evaluates from a
fingerprinted tree, so tampering is detectable even though it is not preventable,
and that claim is stated rather than implied.

## Operational notes that cost time to discover

- **Both CLIs refuse to use tools unattended as root, differently.** Claude
  Code's `--dangerously-skip-permissions` exits non-zero under root before any
  model call; `--permission-mode bypassPermissions` plus `IS_SANDBOX=1` works.
  Codex needs `--dangerously-bypass-approvals-and-sandbox` and
  `--skip-git-repo-check`, the latter because a packet is not a git repo.
- **The agent's `verify` points at a reduced harness tree**, not the repo. A
  smoke run was observed reading `src/sol_execbench/core/bench/timing.py`; the
  same reach covers `artifacts/05` tolerance derivations and `artifacts/03`
  bounds. See deviation D17.
- **`benchmark_reference` defaults to False**, which silently leaves every
  speedup undefined. Both the scorer and `agent_verify.py` set it.
- **`are_clocks_locked()` reads an environment variable, not the hardware.**
  `score_solutions.py` probes sysfs for `perf_determinism` on the authoritative
  GPU and refuses to score if it is absent, then sets the flag. Exporting the
  flag on an unlocked node would make every latency read as authoritative while
  being taken at a boost clock. See D19.
- **Gateway 403s happen mid-session** and are retried rather than recorded as a
  model failure. A session that produced a solution is never retried. See D18.
- **Agents get GPUs 1–7 and never GPU 0.** On this node the achieved clock at one
  determinism setting spans 1318–1644 MHz across the eight GPUs (D16), so an
  agent's own timings come from a ~20% slower clock than the score does. That is
  harmless — it compares its kernel to the reference on its own GPU, so the ratio
  it optimizes against is right — but it is why the score never comes from the
  agent's GPU.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 10
```

Passes when:

- Every problem in scope has a session for every harness, or a recorded reason
  in `artifacts/deferred.json`. Per harness, because a sweep can cover one and
  miss the other:

  ```bash
  for h in claude-code codex; do
    python scripts/check_coverage.py --artifacts artifacts/10/scores/full-01/$h
  done
  ```

  The score files use the flat `<Category>__<problem>.json` layout the script
  already understands. It must exit zero for each — an omitted `--category` is
  the realistic way scope shrinks, and it looks exactly like success.
- Every score record carries a `score_basis`, and no aggregate mixes two.
- The harness fingerprint was taken at sweep start and its comparison is on the
  summary, whether it matched or not.
- Cost and wallclock are present per session, or explicitly null for a harness
  that does not report them.
- `dashboard.html` renders with no network access.

## Guard rails

- **Do not compare a score across score bases**, including across runs. A
  `speedup_vs_reference` run and a `sol_score_v1` run are different measurements.
- **Do not quote an agent score against upstream's median-0.732** unless the
  basis is `sol_score_v1` and the manifest passed `verify_anchor.py`. And even
  then it is a within-platform figure; the cross-vendor caveat in task 09 step 5
  applies in full.
- Do not raise `--max-attempts` for one harness and not another, then compare.
  The budget is part of the measurement.
- Do not drop a problem because an agent kept failing on it. That is the result.

## Outputs

- `artifacts/10/runs/<run-id>/` — packets and session records
- `artifacts/10/scores/<run-id>/` — per-problem score files and a summary
- `artifacts/10/scoreboard.json`, `artifacts/10/dashboard.html`
- `STATE.md`: pilot cost and time per problem, and the extrapolated full-run
  budget that decision was based on
