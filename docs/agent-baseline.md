# Agent baseline

`artifacts/09/agent-baseline.json` records that no kernel-optimizing agent was
ever run on this node, and why the four PyTorch variants cannot stand in for
one: they cluster at S = 0.5 by construction, because T_b is defined as the
fastest of them. Upstream's headline numbers — median SOL 0.732, headroom
correlation r = 0.981, 14.5% flagged — are results *about agents*, and nothing
in this repo was comparable to them.

This is the instrument that closes the gap, and the cost study that says what
closing it fully would take.

## What it does

```bash
# 1. put N agents in sandboxes, one per GPU, and let them optimize
python scripts/agent_baseline.py --sample 8 --gpus 1,2,3,4,5,6,7 --budget-usd 8

# 2. re-time every surviving kernel on an idle GPU 0, then score it
python scripts/agent_score.py --run artifacts/10/<run-id>

# 3. what it cost, and what a full run would cost
python scripts/agent_cost_report.py --run artifacts/10/<run-id>
```

Each problem gets a sandbox containing the reference implementation, a
`TASK.md` describing the hardware and the rules, and an `./evaluate` command
wired to the *real* harness against the *AMD-derived* tolerances. The agent
edits `kernel.py` and runs `./evaluate` as often as it likes. Whatever is in
`kernel.py` when the session ends is the submission.

The agent is a Claude Code session (`claude -p`) driven through the AMD LLM
gateway's Anthropic-native endpoint.

## Three properties that make it a measurement rather than a demo

**GPU 0 never belongs to an agent.** Agent sessions are exploration under
CLAUDE.md section 4, so they get 1–7, seven of them at once, each perturbing
the others. Nothing measured under those conditions can be a score. Every
surviving kernel is re-timed afterwards on an idle GPU 0 at 50 iterations —
the settings T_b itself was measured at — and only that number is scored.

The re-time is also the honesty check. A kernel that looked fast in a contended
sandbox and is not fast on an idle card was measuring its neighbours.

**The agent is never shown T_SOL or T_b.** It sees correctness, its own
latency, and the reference latency. That is what a kernel engineer gets from a
profiler. Handing it the scoring constants would let it optimize against the
score rather than the hardware, and the resulting baseline would measure the
leak.

**Sessions that fail still count.** Cost is captured per session from the CLI's
own accounting, including sessions that produced nothing and sessions stopped
at the spend cap. A cost study that drops its failures prices the easy problems.

## The three numbers, and what they mean

| | measured how |
|---|---|
| **cost** | `total_cost_usd` per session from the Claude Code CLI, plus the raw token counts. The token counts are the measurement; the dollars are a list-price conversion. |
| **wall time** | per session, and for the run as a whole at N-way concurrency. |
| **GPU concurrency** | one agent per GPU, with per-card busy% sampled every 5 s for the duration. |

The GPU number is the surprising one, and it is the reason the study exists:
an agent is GPU-idle for most of a session. It reads, reasons and edits, and
touches the GPU only when it evaluates. Occupancy tells you how many agents a
node can host, which is what determines the wall time of a full run.

### Card indices are scrambled

`rocm-smi` reports in card order and `HIP_VISIBLE_DEVICES` selects in torch
order, and on this node those disagree: torch 0 is card3, torch 1 is card0.
The utilization table resolves the mapping through PCI bus identity
(`scripts/gpu_map.py`) before labelling anything. Without that step, an agent's
work gets attributed to the wrong GPU — including making the reserved GPU 0
appear busy, which is exactly the kind of plausible, undetectable wrongness the
mapping helper was written for.

## Reading the extrapolation

`cost-report.md` extrapolates the sample to the full benchmark. Two things
bound how much weight it carries:

* **Capped sessions are lower bounds.** A session stopped at its spend cap cost
  what the cap allowed, not what the problem needed. They are counted
  separately and flagged in the per-problem table.
* **Cost per problem varies by more than an order of magnitude.** The
  extrapolation is a range from the observed median and p75, with the sample
  size stated beside it. It is not a point estimate and should not be quoted
  as one.

## Eval cost is not uniform across categories

An L1 elementwise problem evaluates in seconds. `FlashInfer-Bench__019` — 38
workloads over a ~1M-page KV cache — takes over ten minutes for a single
`./evaluate`. The feedback loop, not the token spend, is what limits agentic
optimization on those problems: an agent with a one-hour session gets a
handful of measurements, where an L1 agent gets dozens.

One harness choice makes this worse than it needs to be: `agent_eval.py` sets
`benchmark_reference=True` so the agent can see its speedup, which times the
reference on every single evaluation. Measuring the reference once when the
sandbox is built, and reusing it, would roughly halve every evaluation. That is
a change for the next run, not this one — altering it mid-flight would make the
sessions incomparable.

## What the first pilot found

Eight problems, `Claude-Opus-5`, an $8-per-problem cap
(`artifacts/10/pilot8/cost-report.md`):

| | |
|---|---|
| cost | **$65.08**, 16.0M tokens, all eight sessions stopped at the cap |
| wall time | 46 min at 7 concurrent agents; 27 min median per problem |
| GPU occupancy | **mean 7.5%** across the seven cards |
| scored | 99 workloads, mean S = **0.776** |

Four of the eight improved on the reference (+0.265 to +0.525 in score);
four gained nothing. Median gain **+0.133**.

**Three of the eight submitted a kernel that fails correctness.** 0/11, 0/16
and 1/16 on re-time. Their only clean evaluation in the whole session was the
untouched baseline — every optimization they attempted was broken, and they
left the last broken one in place when the budget ran out, despite `TASK.md`
telling them to revert to the last version that passed. That is worse than a
null result: an agent baseline that reported only its successes would be
reporting on 5 problems and calling it 8.

The GPU number is the one that changes planning. Six of seven cards averaged
0.4–3.5% busy; the seventh sat at 50.8% only because its problem reads a
1.14 GB paged cache on every evaluation. **A node does not need one GPU per
agent** — it needs enough GPUs that evaluations do not queue. What it cannot
do is let two agents share a card while either is timing.

### The pilot found a defect in the benchmark, not just in itself

The best kernel in the run came in **faster than T_SOL on 25 of 38 workloads**.
Nothing beats the roofline, so that is a broken bound — the declared-traffic
tier counts a whole paged KV cache as mandatory traffic when the kernel gathers
34 pages out of 989,669. Six FlashInfer problems and 249 scoreable workloads are
exposed. `STATE.md` D18 has the derivation and the v1.1 fix.

The `T_SOL <= T_b` gate could not catch it, because T_b comes from a PyTorch
reference that over-reads in exactly the same way. It took a kernel that
avoided the traffic to separate the two numbers — which is a good argument for
running an agent against a benchmark you intend to publish.

## What this is not

It is a **pilot over a sample**, not a full-benchmark submission. Its coverage
on the leaderboard is a fraction of a percent by design, and the board's
headline benchmark score reflects that honestly rather than hiding it. The
purpose is to price a full run, and to prove the path works end to end.
