# Agent baseline: what it costs

`Claude-Opus-5` via the AMD LLM gateway, driven by the Claude Code CLI, on 4 problems sampled across category and headroom.

Run `opus5-budget100` &middot; 2026-08-04T22:18:27.557503+00:00

## Headline

| | |
|---|---|
| cost, 4 problems | **$249.58** |
| cost per problem | median **$71.59**, mean $62.40, range $25.04–$81.36 |
| wall time per problem | median 136 min, max 149 min |
| wall time, whole run | 149 min at 4 concurrent agents |
| GPUs occupied | 4 (one per agent), mean busy 6.8% |
| result | 59 workloads scored, mean S = **0.701**, 0 flagged |

## Per problem

| problem | GPU | cost | wall | turns | evals | workloads passed | mean S | speedup | capped |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| `Quant__004_fp8_moe_expert_linear` | 4 | $81.36 | 138 min | 216 | 9 | 16/16 | 0.944 | 11.02x |  |
| `L2__050_vae_decoder_mid_block_attention_resnet` | 3 | $74.16 | 134 min | 209 | 13 | 11/11 | 0.539 | 1.04x |  |
| `L1__030_attention_output_projection_with_residual` | 1 | $69.02 | 149 min | 188 | 15 | 16/16 | 0.566 | 1.19x |  |
| `L1__046_attention_softmax_with_softcapping_and_dropout` | 2 | $25.04 | 72 min | 90 | 8 | 16/16 | 0.705 | 4.24x |  |

Burn rate varies from $0.35 to $0.59 per session-minute, averaging $0.51. The spread is not model variance: an agent blocked on a slow evaluation spends wall time without spending tokens, so the problems with the most expensive evaluations have the *lowest* dollar-per-minute.

## Tokens

| kind | tokens | $/Mtok | cost |
|---|--:|--:|--:|
| input | 39,989,376 | 5.00 | $199.95 |
| output | 1,308,781 | 25.00 | $32.72 |
| cache write | 867,857 | 6.25 | $5.42 |
| cache read | 15,244,832 | 0.50 | $7.62 |
| **total** | **57,410,846** | | **$249.58** |

Token counts are the measurement. Dollars are the CLI's list-price conversion; the gateway bills separately.

## GPU concurrency

Agents ran on GPUs [1, 2, 3, 4] — 4 concurrent, one each. GPU 0 was held idle and every score was re-measured on it afterwards.

| card | mean busy % | peak busy % |
|---|--:|--:|
| GPU 1 (card0) | 19.2 | 100 |
| GPU 3 (card1) | 4.3 | 100 |
| GPU 2 (card2) | 0.3 | 71 |
| GPU 0 (card3) | 21.4 | 100 |
| GPU 5 (card4) | 0.2 | 14 |
| GPU 7 (card5) | 0.2 | 13 |
| GPU 6 (card6) | 0.2 | 51 |
| GPU 4 (card7) | 8.4 | 100 |

Busy% is occupancy, not throughput. An agent spends most of a session reading, reasoning and editing; the GPU is idle for all of it. This is the number that says how many agents a node can actually host.

## What the marginal dollar buys

Every session in this run ended at `budget_exhausted`, which means **cost per problem is the cap, not the problem**. The agents burn $0.51 per minute of session and will use whatever they are given. So the useful question is not what a problem costs but what more budget buys.

Per problem, the first evaluation is the untouched reference; the last is what the budget bought:

| problem | evals | first S | best S | gain | submitted kernel correct |
|---|--:|--:|--:|--:|:--:|
| `L1__030_attention_output_projection_with_res` | 15 | 0.493 | 0.566 | +0.073 | yes |
| `L1__046_attention_softmax_with_softcapping_a` | 8 | 0.355 | 0.704 | +0.349 | yes |
| `L2__050_vae_decoder_mid_block_attention_resn` | 13 | 0.498 | 0.541 | +0.043 | yes |
| `Quant__004_fp8_moe_expert_linear` | 9 | 0.500 | 0.944 | +0.445 | yes |

Median gain over the reference: **+0.211**.


### Where the score stopped moving

| problem | evals | reached 99% of best at | evals after that |
|---|--:|--:|--:|
| `L1__030_attention_output_projection_with_res` | 15 | 5 | 10 |
| `L1__046_attention_softmax_with_softcapping_a` | 8 | 5 | 3 |
| `L2__050_vae_decoder_mid_block_attention_resn` | 13 | 11 | 2 |
| `Quant__004_fp8_moe_expert_linear` | 9 | 5 | 4 |

**19 of 45 evaluations came after the score had stopped improving.** A session that runs to natural completion keeps polishing well past its last measurable gain, so the budget worth paying for is the one that reaches the plateau, not the one the agent stops at. Eval index is a proxy for spend here, not a linear one.

A cross-problem average at eval N would be survivorship: the problems that reach a high N are the ones the agent was struggling with, so the mean falls as N rises for reasons that have nothing to do with budget.

Measured on GPUs 1-7 under seven-way agent load, so these are trajectory values, not scores. The authoritative numbers are the idle-GPU-0 re-times in scored.json.

## Extrapolating to the full benchmark

| | |
|---|---|
| problems | 220 scoreable |
| cost | **$15,750 – $16,710** (median–p75 per problem × 220) |
| wall time at 4-way concurrency | **113 h** |
| wall time serial | 452 h |

Caveats, none of them optional:

* No session hit the $100.0 cap — all 4 ran to natural completion, so these are the costs the problems actually incur rather than the budget they were allowed. They are an UPPER bound on what is worth spending: the score plateaus well before the agent stops.
* n = 4. Cost per problem spans $25.04 to $81.36 in this sample; the range above reflects that spread, not confidence in a point estimate.
* Dollars are list-price equivalents computed by the Claude Code CLI. Token counts are the measurement; the AMD gateway bills separately and its contract rates may differ.
* Wall time assumes one agent per GPU. Agents are GPU-idle most of the time, so oversubscribing is possible -- but it would perturb the timings each agent optimizes against.
