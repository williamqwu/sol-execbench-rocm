# Agent baseline: what it costs

`Claude-Opus-5` via the AMD LLM gateway, driven by the Claude Code CLI, on 8 problems sampled across category and headroom.

Run `pilot8` &middot; 2026-08-04T20:36:21.357004+00:00

## Headline

| | |
|---|---|
| cost, 8 problems | **$65.08** |
| cost per problem | median **$8.16**, mean $8.13, range $8.01–$8.23 |
| wall time per problem | median 27 min, max 36 min |
| wall time, whole run | 46 min at 7 concurrent agents |
| GPUs occupied | 7 (one per agent), mean busy 7.5% |
| result | 99 workloads scored, mean S = **0.776**, 0 flagged |

> **8 of 8 sessions hit the $8.0 spend cap** and were stopped mid-work. Their cost is what the cap allowed, not what the problem needed, so every figure derived from them is a lower bound.

## Per problem

| problem | GPU | cost | wall | turns | evals | workloads passed | mean S | speedup | capped |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| `L2__050_vae_decoder_mid_block_attention_resnet` | 6 | $8.23 | 24 min | 51 | 1 | 0/11 | — | — | yes |
| `FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` | 1 | $8.23 | 36 min | 46 | 2 | 38/38 | 0.990 ⚠25 | 182.49x | yes |
| `L1__046_attention_softmax_with_softcapping_and_dropout` | 3 | $8.18 | 30 min | 47 | 8 | 16/16 | 0.620 | 3.04x | yes |
| `Quant__004_fp8_moe_expert_linear` | 7 | $8.16 | 33 min | 38 | 3 | 16/16 | 0.940 | 10.48x | yes |
| `L2__069_joint_transformer_block_residual_path` | 7 | $8.15 | 14 min | 44 | 2 | 0/16 | — | — | yes |
| `L1__053_gaussian_topk_sparse_activation` | 4 | $8.10 | 20 min | 46 | 2 | 12/12 | 0.937 | 17.85x | yes |
| `L2__015_audio_sinusoidal_position_embedding_with_conv_projection` | 5 | $8.02 | 29 min | 46 | 2 | 1/16 | 0.482 | 0.97x | yes |
| `L1__030_attention_output_projection_with_residual` | 2 | $8.01 | 17 min | 51 | 5 | 16/16 | 0.491 | 1.00x | yes |

⚠ = workloads measured **faster than T_SOL**, excluded from the mean. Nothing beats the roofline, so those are defective bounds rather than results. See STATE.md D18.

Burn rate varies from $0.23 to $0.59 per session-minute, averaging $0.32. The spread is not model variance: an agent blocked on a slow evaluation spends wall time without spending tokens, so the problems with the most expensive evaluations have the *lowest* dollar-per-minute.

## Tokens

| kind | tokens | $/Mtok | cost |
|---|--:|--:|--:|
| input | 9,067,849 | 5.00 | $45.34 |
| output | 477,817 | 25.00 | $11.95 |
| cache write | 283,589 | 6.25 | $1.77 |
| cache read | 6,162,213 | 0.50 | $3.08 |
| **total** | **15,991,468** | | **$65.08** |

Token counts are the measurement. Dollars are the CLI's list-price conversion; the gateway bills separately.

## GPU concurrency

Agents ran on GPUs [1, 2, 3, 4, 5, 6, 7] — 7 concurrent, one each. GPU 0 was held idle and every score was re-measured on it afterwards.

| card | mean busy % | peak busy % |
|---|--:|--:|
| GPU 1 (card0) | 50.8 | 100 |
| GPU 3 (card1) | 0.4 | 48 |
| GPU 2 (card2) | 0.5 | 94 |
| GPU 0 (card3) | 0.1 | 8 |
| GPU 5 (card4) | 2.2 | 100 |
| GPU 7 (card5) | 2.2 | 100 |
| GPU 6 (card6) | 3.5 | 100 |
| GPU 4 (card7) | 0.5 | 99 |

Busy% is occupancy, not throughput. An agent spends most of a session reading, reasoning and editing; the GPU is idle for all of it. This is the number that says how many agents a node can actually host.

## What the marginal dollar buys

Every session in this run ended at `budget_exhausted`, which means **cost per problem is the cap, not the problem**. The agents burn $0.32 per minute of session and will use whatever they are given. So the useful question is not what a problem costs but what more budget buys.

Per problem, the first evaluation is the untouched reference; the last is what the budget bought:

| problem | evals | first S | best S | gain | submitted kernel correct |
|---|--:|--:|--:|--:|:--:|
| `FlashInfer-Bench__019_mla_paged_prefill_caus` | 2 | 0.498 | 1.017 | +0.519 | yes |
| `L1__030_attention_output_projection_with_res` | 5 | 0.493 | 0.493 | +0.000 | yes |
| `L1__046_attention_softmax_with_softcapping_a` | 8 | 0.356 | 0.621 | +0.265 | yes |
| `L1__053_gaussian_topk_sparse_activation` | 2 | 0.412 | 0.937 | +0.525 | yes |
| `L2__015_audio_sinusoidal_position_embedding_` | 2 | 0.498 | 0.498 | +0.000 | **no** (1/16) |
| `L2__050_vae_decoder_mid_block_attention_resn` | 1 | 0.487 | 0.487 | +0.000 | **no** (0/11) |
| `L2__069_joint_transformer_block_residual_pat` | 2 | 0.499 | 0.499 | +0.000 | **no** (0/16) |
| `Quant__004_fp8_moe_expert_linear` | 3 | 0.500 | 0.940 | +0.440 | yes |

Median gain over the reference: **+0.133**.

A cross-problem average at eval N would be survivorship: the problems that reach a high N are the ones the agent was struggling with, so the mean falls as N rises for reasons that have nothing to do with budget.

Measured on GPUs 1-7 under seven-way agent load, so these are trajectory values, not scores. The authoritative numbers are the idle-GPU-0 re-times in scored.json.

## Extrapolating to the full benchmark

| | |
|---|---|
| problems | 220 scoreable |
| cost | **$1,795 – $1,802** (median–p75 per problem × 220) |
| wall time at 7-way concurrency | **13 h** |
| wall time serial | 93 h |

Caveats, none of them optional:

* 8 of 8 sessions were stopped at the $8.0 spend cap, so their cost is a LOWER bound and so is any figure derived from them.
* n = 8. Cost per problem spans $8.01 to $8.23 in this sample; the range above reflects that spread, not confidence in a point estimate.
* Dollars are list-price equivalents computed by the Claude Code CLI. Token counts are the measurement; the AMD gateway bills separately and its contract rates may differ.
* Wall time assumes one agent per GPU. Agents are GPU-idle most of the time, so oversubscribing is possible -- but it would perturb the timings each agent optimizes against.
