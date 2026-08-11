### census:taxonomy
findings

### census:tolerance
## Tolerance census — how the AMD tolerances were derived, and what they select for

Everything below is reproducible from **one command** (read-only, no GPU):

```
python artifacts/11/compile-diag/tolerance_census.py
```

New file written: `/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/tolerance_census.py`. No repo file was edited.

---

## 0. Two things in the brief that I have to falsify first

### 0a. The leaderboard DB has submission 2's per-workload statuses INVERTED

The brief says "585 FAILED of 3171 scoreable workload rows". The count 585 is in the DB, but the *rows it names* are the passing ones.

```
# board_check section of tolerance_census.py
db submission 2 FAILED rows: 585
artifacts/06 v2_compile FAILED: 523  PASSED: 3171
db FAILED that artifacts call FAILED: 0
db FAILED that artifacts call PASSED: 585
artifact failures with no db row at all: 523
```

Zero overlap. Every one of the 585 DB "FAILED" rows carries a measured `latency_ms`, and `latency_ms_by_workload` in `artifacts/06/candidates/*.json` is *by construction the set that passed* (`leaderboard/ingest.py:779-780` comment says so explicitly). Direct check:

```
sqlite3 -header -column leaderboard/solbench.db \
  "select workload_uuid,status,latency_ms from result where submission_id=2 and problem_key like 'L2__009%';"
```
returns 8 rows, all `FAILED`, with latencies `82.2137641906738, 164.386260986328, …` — those are exactly the 8 UUIDs and 8 latencies in `variants.v2_compile.latency_ms_by_workload` of `artifacts/06/candidates/L2__009_decoder_layer_with_residual_connections.json`, i.e. the 8 that *passed*. The 8 that actually failed (`f0da3300…`, the one the lead reproduced on GPU at matched_ratio 0.44) have **no row at all**.

The DB predates the D28 fix described in the working-tree `ingest.py`; `ingest.py` is currently modified and un-re-ingested. **Do not use board statuses for this analysis.** I used `artifacts/06/{candidates,authoritative}` throughout. Ground truth:

| variant | FAILED | PASSED | covered / 3717 | failing problems |
|---|---|---|---|---|
| v1_eager | 10 | 3707 | 3717 | 1 |
| v2_compile | **523** | 3171 | 3694 | **71** (L2 41, L1 24, Quant 6) |
| v3_compile_max_autotune | 581 | 3041 | 3622 | 81 |

### 0b. The dominant structure is not tolerance at all — it is `torch._dynamo.config.recompile_limit == 8`

Before any tolerance correlation is meaningful, this: **522 of the 523 v2 failures are at workload index < 8 within their problem's `workload.jsonl`.**

```
failing workloads by file index within the problem: {'idx<8': 522, 'idx>=8': 1}
fail rate, first 8 workloads of each problem: 522/1745  (29.9%)
fail rate, workloads 9+:                      1/1949    (0.05%)
```

Of 71 failing problems, 57 fail an exact prefix `[0..k-1]`; the other 14 fail only indices `< 8` too (sole exception: `L2__018_cu_seqlens_variable_length_vision_attention` index 12). v3 behaves the same way: 567/1712 at idx<8, 14/1910 at idx>=8.

Mechanism, confirmed:

- `reference/tb-candidates/variants.py:63` compiles **one module-level callable, reused for every workload of the problem**, with `dynamic=False`:
  ```python
  _solb_compiled = _solb_torch.compile(_solb_ref_run, mode={mode_arg}, dynamic=False)
  ```
- `docker exec -w /work solbench python -c "import torch._dynamo.config as c; print(c.recompile_limit, c.fail_on_recompile_limit_hit)"` →
  ```
  torch 2.9.1+rocm7.2.0.git7e1940d4
  cache_size_limit 8
  recompile_limit 8
  fail_on_recompile_limit_hit False
  ```

So the 9th distinct shape and beyond **silently fall back to eager** and therefore pass bit-exactly against the eager reference. `torch.compile` was only actually exercised on 1745 of 3694 workloads. The honest denominator is 1745, not 3171 — and on it the real failure rate is **29.9%**, not 16.5%.

---

## 1. How `max_atol` / `max_rtol` are derived

`scripts/runners/calibrate_tolerance.py`. Two executions of the *same* reference on the *same* inputs, over 10 seeds, on GPU:

```python
for seed in range(a.seeds):
    torch.manual_seed(seed)
    inputs = prepare_inputs(definition, wl, ns)
    with torch.no_grad():
        out_a = [t.detach().clone() for t in _as_list(run(*inputs))]
    torch.cuda.empty_cache()
    with torch.no_grad():
        out_b = [t.detach().clone() for t in _as_list(run(*inputs))]
    for x, y in zip(out_a, out_b):
        max_abs = max(max_abs, _max_abs(x, y))
```

then (lines 190-256):

```python
eps  = _dtype_floor(base)
atol = max(max_abs * a.margin, eps["atol"])
...
entry["tolerance"] = {
    "max_atol": atol,
    "max_rtol": max(max_rel * a.margin, eps["rtol"]),
    "required_matched_ratio": 0.99,
    "_derivation": (f"max run-to-run error over {a.seeds} seeds x "
                    f"{a.margin} margin, floored at {base[0].dtype} epsilon"),
}
```

The floor (`_dtype_floor`, lines 288-357):

```python
eps = float(torch.finfo(dtype).eps)          # TypeError -> {"atol":0.0,"rtol":0.0} for ints
...
scale = math.sqrt(total_sq / total_n) if total_n else 0.0
return {"atol": eps * scale, "rtol": eps}
```

So: **`atol_floor = eps(dtype) x RMS|output|`, `rtol_floor = eps(dtype)`.** Note the floor is *per output dtype*, not fp32-specific — the brief's "floored at torch.float32 epsilon" is what appears in the `_provenance` string only for fp32-output problems. Corpus-wide, taken from the strings themselves:

```
floored at: {'bfloat16': 2030, 'float32': 1291, 'float16': 320, 'int64': 48, 'int32': 16, 'bool': 12}
```

**The float64 CPU golden is computed, recorded, and then not used.** Lines 226-241 write `entry["vs_golden"]`; the tolerance at line 248 reads only `max_abs`/`max_rel` from the run-to-run loop. The golden never widens or narrows a tolerance. `tasks/05-tolerances.md` §"Use the float64 CPU golden references" asks for it as a *bug detector*, and the acceptance check only requires it be "recorded as run or explicitly not-applicable per problem".

Because `atol_floor = eps x RMS`, the floored bound at a typical element is `eps·RMS + eps·|y| ≈ 2·eps·RMS`. **The floor demands agreement to ~1 ulp of the output dtype, whatever that dtype is** — which is the whole story of §3.

---

## 2. Distribution of the derived tolerances (3717 scoreable workloads)

```
atol  min/p10/p25/med/p75/p90/p99/max: 0.000e+00 8.429e-08 2.668e-07 4.649e-03 9.766e-03 9.393e-02 2.056e+02 7.782e+05
rtol  min/p10/p25/med/p75/p90/p99/max: 0.000e+00 1.192e-07 1.192e-07 7.812e-03 7.812e-03 7.812e-03 2.955e-01 6.667e-01
atol == 0.0 exactly (integer/bool outputs): 76
required_matched_ratio: {0.99: 3717}   # uniform
```

**`rtol == 1.1920928955078125e-07` exactly: 1224 / 3717 = 32.9%.**

| category | workloads | rtol==fp32eps | pct |
|---|---|---|---|
| L1 | 1480 | 673 | 45.5% |
| L2 | 1299 | 551 | 42.4% |
| Quant | 278 | 0 | 0.0% |
| FlashInfer-Bench | 660 | 0 | 0.0% |

| first output dtype | workloads | rtol==fp32eps | rtol==eps(dtype) | median atol |
|---|---|---|---|---|
| bfloat16 | 2030 | 0 | 1968 (97.0%) | 7.805e-03 |
| float32 | 1291 | **1224 (94.8%)** | 1224 | 1.680e-07 |
| float16 | 320 | 0 | 320 (100%) | 3.563e-02 |
| int64 | 48 | 0 | 0 (atol=rtol=0) | 0.000e+00 |
| int32 | 16 | 0 | 0 (atol=rtol=0) | 0.000e+00 |
| bool | 12 | 0 | 0 (atol=rtol=0) | 0.000e+00 |

3502 of 3717 (94.2%) sit at their own dtype's eps floor. The fp32-eps subset is not "the floored ones" — it is "the floored ones whose dtype floor happens to be 65 536x tighter than bf16's".

---

## 3. THE CORRELATION

### 3a. Over all scoreable workloads with a v2 verdict (n = 3694)

**`rtol == fp32 eps`:**

| | v2 FAILED | v2 PASSED | total | fail rate |
|---|---|---|---|---|
| floored at fp32 eps | 339 | 875 | 1214 | 27.9% |
| not | 184 | 2296 | 2480 | 7.4% |

odds ratio 4.83, chi2(1) = 282.0

**"run-to-run variance measured exactly 0" (i.e. *some* floor binds) — the weak version:**

| | v2 FAILED | v2 PASSED | total | fail rate |
|---|---|---|---|---|
| variance == 0 | 517 | 3041 | 3558 | 14.5% |
| variance > 0 | 6 | 130 | 136 | 4.4% |

odds ratio 3.42, chi2(1) = 11.0 — much weaker, because 96.3% of the corpus is floored.

### 3b. Restricted to the population torch.compile actually compiled (index < 8, n = 1745)

This is the correct denominator. The signal roughly doubles:

| discriminator | FAILED | PASSED | fail rate (yes) | fail rate (no) | OR | chi2 |
|---|---|---|---|---|---|---|
| `rtol == fp32 eps` | 339 / 273 | 183 / 950 | **55.4%** | 16.2% | **6.43** | 291.8 |
| output dtype `float32` | 339 / 307 | 183 / 916 | 52.5% | 16.7% | 5.51 | 249.1 |
| `atol == 0` (int/bool) | 11 / 24 | 511 / 1199 | 31.4% | 29.9% | 1.10 | 0.0 |

Fail rate by output dtype, index<8 only — this is the cleanest statement of the mechanism:

| dtype | eps floor | n | FAILED | rate |
|---|---|---|---|---|
| float32 | 1.192e-07 | 646 | 339 | **52.5%** |
| bfloat16 | 7.8125e-03 | 972 | 172 | 17.7% |
| float16 | 9.766e-04 | 92 | 0 | **0.0%** |
| int64 | 0 | 19 | 11 | 57.9% |
| int32 | 0 | 8 | 0 | 0.0% |
| bool | 0 | 8 | 0 | 0.0% |

Problem level (220 problems, any-fail): fp32-output 44 fail / 37 pass; non-fp32 27 fail / 112 pass. OR 4.85, chi2 28.5.

**Verdict: "tolerance is at the fp32-eps floor" is a strong but not sufficient discriminator.** It explains 339 of 523 failures (64.8%) and roughly a 6x odds shift, but both off-diagonal cells are populated and neither is small.

### 3c. Off-diagonal, named

**Cell C — not fp32-floored, yet fails (27 problems, 184 workloads).** Almost all bf16 with `rtol = 7.812e-03`; the floor still binds, it is just the *bf16* floor:

| problem | dtype | failed/total | atol range |
|---|---|---|---|
| L2__018_cu_seqlens_variable_length_vision_attention | bfloat16 | 9/16 | 3.79e-04..7.21e-04 |
| L1__005_conv_gated_projection_with_causal_conv | bfloat16 | 8/16 | 2.55e-02..2.64e-02 |
| L1__015_grouped_query_attention_with_rope_and_qk_norm | bfloat16 | 8/16 | 4.07e-04..2.24e-03 |
| L1__047_attention_with_qk_norm_and_rope | bfloat16 | 8/16 | 3.48e-03..5.46e-03 |
| L1__093_grouped_topk_moe_routing_backward | bfloat16 | 8/16 | 2.64e-03..2.83e-03 |
| L2__004_fused_residual_rms_mlp | bfloat16 | 8/16 | 4.66e-03..4.67e-03 |
| L2__027_grouped_query_attention_with_yarn_rope_and_qk_norm | bfloat16 | 8/16 | 2.11e-03..5.67e-03 |
| L2__028_gqa_rotary_attention_core_backward | bfloat16 | 8/16 | 2.90e-02..1.96e-01 |
| L2__034_vision_language_cross_attention_fusion | bfloat16 | 8/16 | 5.61e-04..2.83e-03 |
| L2__041_kv_shared_attention_with_dual_rope | bfloat16 | 8/16 | 4.11e-03..6.39e-03 |
| **L2__049_group_limited_topk_routing** | **int64** | 8/16 | **0.00e+00** (atol=rtol=0) |
| L2__056_language_model_decoder_prenorm_attention_ffn_residual_backward | bfloat16 | 8/16 | 1.84e-01..1.39e+00 |
| L2__058_mamba2_selective_scan | bfloat16 | 8/16 | 4.65e-03..4.67e-03 |
| Quant__004_fp8_moe_expert_linear | bfloat16 | 8/16 | 4.61e-03..4.85e-03 |
| Quant__012_fp8_shared_expert_mlp | bfloat16 | 8/16 | 4.63e-03..4.65e-03 |
| Quant__013_fp8_mla_kv_compression_projection | bfloat16 | 8/16 | 1.79e-01..1.79e-01 |
| Quant__016_fp8_multi_latent_attention_qkv_projection | bfloat16 | 8/16 | 7.80e-03..7.81e-03 |
| Quant__017_fp8_shared_expert_mlp | bfloat16 | 8/16 | 4.62e-03..4.65e-03 |
| L2__040_altup_predict_correction_cycle_backward | bfloat16 | 7/16 | 1.03e-02 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | bfloat16 | 7/16 | 3.88e-01..1.25e+00 |
| L2__054_vision_encoder_layer_with_gated_residuals | bfloat16 | 6/16 | 7.84e-03..1.18e-02 |
| L1__061_tanh_gated_residual_add_backward | bfloat16 | 5/16 | 6.36e-03..1.20e-02 |
| L2__007_multimodal_rotary_embedding_attention | bfloat16 | 5/16 | 7.69e-03..8.73e-03 |
| L2__062_decoder_complete_layer | bfloat16 | 3/16 | 9.12e-03..9.39e-03 |
| **Quant__011_fp8_moe_gate_routing** | **int64** | 3/16 | **0.00e+00** |
| L2__015_audio_sinusoidal_position_embedding_with_conv_projection | bfloat16 | 2/16 | 1.84e+02..3.20e+02 |
| L1__062_kv_cache_update_with_rope_backward | bfloat16 | 1/16 | 7.81e-03..8.26e-03 |

Two sub-classes worth separating:
- **Integer outputs get `atol = rtol = 0`** — `torch.finfo` raises `TypeError` for int/bool, and `_dtype_floor` returns `{"atol": 0.0, "rtol": 0.0}`. That is a demand for *literal bit-identity of index tensors*. `L2__049` (top-k routing indices) and `Quant__011` fail exactly there: any tie-break reordering by Inductor is fatal. 11 of 523 failures. Upstream gave `L2__049` atol 0.56–0.84 / rtol 0.01 / ratio 0.98 — the tie-break was free there.
- The bf16 residue: `L2__015` (atol 184–320, huge outputs) and `L2__056` (atol 0.18–1.39) fail *despite* loose absolute tolerances, so these are genuine numerical divergences, not floor artefacts. Those are the cases worth a GPU repro.

**Cell B — fp32-floored, yet the compiled variant passes every workload (37 problems, 592 workloads).** These are the honest refutation of a pure "fp32-eps floor ⇒ fail" claim. Selected:

| problem | atol range | why it survives (readable from the atol) |
|---|---|---|
| L2__033_multi_scale_feature_pyramid | 1.48e+04 .. 7.78e+05 | fp32 output but *nondeterministic*, so the floor never bound |
| L1__024_vision_rotary_position_embedding_generation_backward | 5.72e-05 .. 1.71e-02 | ditto (in `artifacts/05/triage.md` "structurally nondeterministic") |
| L2__057_residual_coupling_flow_block | 8.90e-05 .. 1.83e-03 | ditto |
| L1__040_conv2d_residual_block | 3.50e-06 .. 3.81e-06 | atol ~30x the bare eps: large RMS |
| L1__066_masked_softmax_with_attention_dropout_backward | 2.66e-11 .. 1.00e-09 | **floor is 100x TIGHTER than fp32 eps and it still passes** |
| L1__072_cross_attention_qkv_projection_with_gqa_repeat_backward | 1.55e-09 .. 5.55e-09 | same |
| L2__065_sparse_expert_dispatch_and_combine | 3.30e-09 .. 3.73e-08 | same |
| L1__088_rotary_position_embedding_application | 1.19e-07 (flat) | at the floor, elementwise op, Inductor reproduces it bit-exactly |
| L1__090_batched_2d_rope_position_encoding_backward | 1.19e-07 (flat) | same |
| L2__051_seqlen-…-hyena_complete_forward_block | 1.20e-07 (flat) | v2 passes 16/16 — but see §4, **eager fails 10/16 here** |

Full list also includes L1__006, 007, 016, 021, 023, 025, 034, 035, 038, 041, 042, 054, 059, 065, 078, 080, 081, 084, 085; L2__017, 025, 030, 031, 052, 067, 071, 076.

The pattern in cell B: floored-fp32 problems that pass are **elementwise / short-reduction** kernels where Inductor's codegen preserves the accumulation order. The ones that fail are **long reductions** (layernorm, matmul chains, softmax, MoE accumulation) where any reassociation moves the low bits — and moving the low bits of >1% of elements is enough, because the bound is 1 ulp.

---

## 4. Is the eager reference actually bit-identical run to run?

Measured by task 05 itself:

```
run-to-run max_abs == 0.0 exactly: 3581/3717 = 96.3%
  L1 1442/1480   L2 1201/1299   Quant 278/278   FlashInfer-Bench 660/660
nonzero run-to-run max_abs: n=136  min=1.490e-08  median=3.906e-03  max=6.226e+05
```

So yes — **for 96.3% of workloads the measured variance was literally 0.0, and the shipped tolerance is 100% floor**. That is what makes the floor bind everywhere.

**But the measurement under-samples the true variance, and there is direct evidence in-repo.** `calibrate_tolerance.py` runs both executions *inside one process, on one GPU, back to back*, which holds hipBLASLt/MIOpen algorithm selection and the allocator history roughly constant. Across processes it does not hold:

- `L2__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block`: all 16 workloads measured `run_to_run max_abs = 0.000e+00`, so all 16 got `atol ≈ 1.198e-07`, `rtol = 1.192e-07`.
- `artifacts/06/candidates/L2__051_….json`, `variants.v1_eager`: `passed: 6 / 16`, ten `INCORRECT_NUMERICAL`. **v1_eager is the unmodified reference** (`variants.py: def _eager(src): return src`), run in a fresh process on GPU 6.

The reference fails its own tolerance on 10 of 16 workloads. The floor is below the reference's own cross-process reproducibility. This is a task-05 methodology defect, independent of torch.compile, and it is the single cleanest piece of evidence that the failure population is tolerance-side.

### The float64 golden cross-check is recorded but not usable as shipped

```
golden recorded for 2331/3717 workloads (165 problems); modes {'float64': 2136, 'native_cpu': 195}
golden max_abs  min/p10/p25/med/p75/p90/p99/max: 0  1.16  6.03  12.1  207  439  1.72e+05  1.06e+12
golden max_abs exceeding the derived atol: 2302 of 2331 (98.8%)
```

The GPU reference is a median of 12.1 absolute away from its "golden", against a median derived atol of 4.6e-03. Taken at face value that would mean the whole benchmark's references are wrong. **I do not believe it, and I would not report it as a measurement of error**, because of a probable confound I found by reading the code, not by running it:

- `scripts/gen_golden.py:98-99` → `torch.manual_seed(0); prepare_inputs(..., device="cpu")`
- `scripts/runners/calibrate_tolerance.py:162-163` → `torch.manual_seed(seed); prepare_inputs(definition, wl, ns)`, and `_common.py:206` defaults `device="cuda:0"`; `io.py:439` passes that device straight into `custom_inputs_fn(axes_and_scalars, dev)`.

Same seed, different generator (CPU vs CUDA) ⇒ **different random input data**. The comparison is very likely of two answers to two different questions, which also explains the `max_rel` tail reaching 5.7e16. Only 5 workloads (all `Quant__014_fp8_yarn_rope_embedding`) came out exactly 0. This needs a 2-line GPU check to confirm; I have not run it. Either way, the *tolerance* was never affected — the golden is recorded and discarded.

---

## 5. Upstream (B200) vs the AMD re-derivation

No B200 number is copied anywhere here; both are read and only their *ratio* is reported. The checker's own count agrees: **exact numeric matches AMD == upstream: 0**.

What upstream ships, by category (`tolerance` block shape, counted over the dataset):

| category | upstream tolerance block |
|---|---|
| L1 | `max_atol`,`max_rtol` on all 1480 (no matched-ratio ⇒ harness default 0.99) |
| L2 | `max_atol`,`max_rtol`,`required_match_ratio` on all 1299 |
| Quant | `max_atol`,`max_rtol`,`required_matched_ratio` on all 518 |
| FlashInfer-Bench | **no tolerance at all** (375 absent, 285 only `allow_negative_inf`) ⇒ harness defaults `1e-2 / 1e-2 / 0.99` |

Effective-bound ratio, `bound = atol + rtol·|y|`, upstream ÷ AMD:

| |y| | population | rows | min | p10 | p25 | **median** | p75 | p90 | p99 | max | upstream looser |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | all scoreable | 3641 | 1.15e-07 | 0.005 | 0.278 | **1.57** | 2.25e+03 | 2.71e+04 | 7.15e+05 | 3.07e+07 | 60.4% |
| 0 | v2-FAILED only | 512 | 8.3e-04 | 0.661 | 4.17 | **1.30e+04** | 2.90e+04 | 1.31e+05 | 1.50e+06 | 3.07e+07 | 88.3% |
| 1 | all scoreable | 3641 | 1.15e-07 | 0.452 | 1.38 | **4.11** | 137 | 1.70e+04 | 4.44e+05 | 1.29e+06 | 81.0% |
| 1 | v2-FAILED only | 512 | 5.6e-03 | 2.01 | 5.78 | **6.68e+03** | 2.00e+04 | 6.39e+04 | 7.72e+05 | 1.29e+06 | 95.5% |
| 100 | all scoreable | 3641 | 1.16e-07 | 1.28 | 5.19 | **6.39** | 84.7 | 1.34e+03 | 2.25e+04 | 3.32e+05 | 96.0% |
| 100 | v2-FAILED only | 512 | 0.030 | 2.55 | 6.39 | **241** | 572 | 9.16e+03 | 5.48e+04 | 1.86e+05 | 98.6% |

By dtype (all scoreable):

| output dtype | n | median atol ratio up/AMD | median rtol ratio up/AMD | median AMD atol | median upstream atol |
|---|---|---|---|---|---|
| **float32** | 1291 | **1.59e+04** | **83.9** | 1.680e-07 | 2.300e-03 |
| bfloat16 | 2030 | 0.698 (AMD *looser*) | 6.4 | 7.805e-03 | 5.900e-03 |
| float16 | 320 | 0.280 (AMD *looser*) | 10.24 | 3.563e-02 | 1.000e-02 |
| int64 | 48 | AMD atol = 0 | AMD rtol = 0 | 0.0 | 5.000e-03 |
| int32 | 16 | AMD atol = 0 | AMD rtol = 0 | 0.0 | 1.000e-05 |
| bool | 12 | AMD atol = 0 | AMD rtol = 0 | 0.0 | 1.000e-05 |

And the matched-ratio, which also moved:

```
required matched ratio, upstream -> AMD: {(0.99, 0.99): 2418, (0.98 -> 0.99): 1299}
```
All 1299 L2 workloads were tightened from 0.98 to 0.99 — a 2x reduction in the permitted fraction of out-of-bound elements, on the category with the most failures.

Direction, over all 3717:

| | rtol AMD tighter | rtol AMD looser |
|---|---|---|
| **atol AMD tighter** | 2218 | 58 |
| **atol AMD looser** | 1404 | 37 |

### Would torch.compile have passed under upstream's numbers?

I cannot claim a re-run I did not do, so: **one workload is settled by measurement, the rest is a bound.**

Settled — the lead's GPU repro on `L2__009_decoder_layer_with_residual_connections` workload `f0da3300`:
- AMD: `atol 1.30e-07, rtol 1.19e-07, ratio 0.99` → matched_ratio 0.44 → FAIL
- Upstream on the same UUID (`data/SOL-ExecBench/benchmark/L2/009_…/workload.jsonl` line 1): `{"max_atol": 0.19, "max_rtol": 1e-05, "required_match_ratio": 0.98}`
- Measured compiled-vs-eager max abs error: **4.8e-06**, which is 2.5e-05 of upstream's atol *alone*. Every element is inside `0.19 + 1e-5·|y|`, so matched_ratio = 1.000 → **PASS**, with ~40 000x of margin.

Bounded for the rest: on 88.3% of the 523 failures upstream's bound is looser at `|y|=0` and on 95.5% at `|y|=1`; the median failing workload's bound is **1.3e+04x** looser in the absolute term. 309 of 512 are ≥100x looser and 287 are ≥1e4x looser. For those, the compiled output would have to be off by four-plus orders of magnitude more than the fp32 last bit for the verdict to survive the tolerance change — implausible for `torch.compile` reassociation, but I have not measured it and will not assert it. The ~40 workloads where upstream is *tighter* (ratio < 1 at `|y|=1`: 4.5%) would still fail, and the 11 integer-output failures would flip to PASS outright (upstream atol 0.005–0.84 vs AMD 0.0).

**The methodology finding, stated plainly:** the AMD re-derivation created this failure population, and it did so through one structural choice, not through per-problem judgement. Upstream calibrated tolerances *against an implementation's expected numerical spread*; task 05 calibrates against *the reference's own back-to-back self-agreement in one process*, floored at one ulp. Where the reference is bit-exact — 96.3% of the corpus — that reduces to "the submission must reproduce the reference's exact accumulation order". For bf16 outputs 1 ulp is 0.78% and reassociation almost always fits; for fp32 outputs 1 ulp is 1.2e-05 % and it almost never does; for integer outputs the tolerance is literally zero. That is the entire shape of the 2x2 in §3.

`tasks/05-tolerances.md` guard rail — "Never loosen a tolerance to make a kernel pass" — is right and should hold. But the converse test was never run: nothing checks that a derived tolerance is *achievable by a correct implementation*. `L2__051`'s eager reference failing its own tolerance 10/16 is that check failing, already, in the repo.

---

## Suggested follow-ups (in dependency order)

1. **Re-ingest the leaderboard.** `leaderboard/solbench.db` publishes inverted per-workload verdicts for submission 2 (and drops 523 real failures). Working-tree `ingest.py` looks like it already fixes this; it has not been run.
2. **The recompile-limit artefact invalidates the v2/v3 anchors as stated.** Workloads 9–16 of every problem were timed as `torch.compile` but ran eager. That is a task-06 T_b correctness issue, not just a numerics one — the T_b for those workloads is an eager time labelled `v2_compile`. Fix is `torch._dynamo.config.recompile_limit = 64` (or one compiled callable per workload) and a re-run.
3. **Cross-process determinism must enter the calibration.** Two executions in one process is not the variance the harness is exposed to. `L2__051` is the reproducer.
4. **The golden cross-check compares different inputs** (CPU vs CUDA RNG under the same seed). Worth 10 minutes to confirm and fix — as shipped, `vs_golden` records nothing usable for 2331 workloads and task 05's acceptance check accepts it.
5. **Integer/bool outputs should not get `atol = rtol = 0`.** `_dtype_floor`'s `except TypeError` path makes index-tensor bit-identity mandatory; 76 workloads and at least 11 failures ride on it.
