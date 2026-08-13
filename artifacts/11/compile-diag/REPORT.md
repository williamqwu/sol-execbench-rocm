# Why `torch.compile` fails ~70 problems on SOL-ExecBench-ROCm (MI350X)

*Analysis of `artifacts/06`, `artifacts/05`, `artifacts/09/manifest-v1.2.json` and five GPU deep-dives with adversarial verification. Every number below was measured; where a verifier corrected the deep-dive that produced it, the corrected value is the one quoted and the disagreement is named.*

---

## 1. The answer in five sentences

Task 05 derives each workload's tolerance from the *run-to-run spread of the eager reference run twice in one process*, which is **exactly 0.0 on 3581 of 3717 scoreable workloads (96.3%)**, so the shipped tolerance is almost always pure dtype-epsilon floor — `atol = eps(dtype)·RMS|y|`, `rtol = eps(dtype)` — i.e. a demand that a submission reproduce eager's own rounding to about one ULP on 99% of elements. `torch.compile` cannot meet that, not because it is wrong but because Inductor legally re-associates reductions, contracts multiply-adds into FMAs, and elides intermediate dtype roundings, each of which moves the last bit or two of a large fraction of elements: on the failing problems the measured compiled-vs-eager divergence is 1–40 ULP where the bound is ~1 ULP, and `matched_ratio` collapses to 0.25–0.99 against a required 0.99. The failure population is **523 `INCORRECT_NUMERICAL` workloads over 71 problems for `v2_compile`** (L1 24, L2 41, Quant 6) and 571 over 80 for `v3_compile_max_autotune`; **70 problems lost both compile anchors**, covering **1115 of 3717 scoreable workloads (30.0%)**. Against a float64 golden the compiled result is **not** the less accurate one on 5 of the 6 problems where anyone actually adjudicated it — on `L1__062` the compiled output is *bit-identical to the correctly-rounded fp64 answer* while eager is not, and on `L1__067` the eager reference itself scores `matched_ratio = 0.089854` against truth under its own tolerance — so on the measured cases the gate is grading *agreement with eager*, not correctness. Separately and independently, **`torch._dynamo.config.recompile_limit` is 8**, so on the 225 of 235 problems with ≥9 distinct shapes the 9th-and-later workloads (2061 of 3957) silently ran eager under a `torch.compile` label: 522 of the 523 failures sit in the compiled prefix, and **626 of the 3717 published anchors are stamped `v2/v3` but were never compiled**.

---

## 2. The mechanism, end to end

### 2.1 How the tolerance is derived

`scripts/runners/calibrate_tolerance.py` runs the *same* reference twice, on the *same* inputs, in the *same* process, for 10 seeds, and takes the largest disagreement:

```python
for seed in range(a.seeds):
    torch.manual_seed(seed); inputs = prepare_inputs(...)
    out_a = run(*inputs); torch.cuda.empty_cache(); out_b = run(*inputs)
    max_abs = max(max_abs, _max_abs(out_a, out_b))
...
atol = max(max_abs * a.margin, eps["atol"])          # line 191
"max_rtol": max(max_rel * a.margin, eps["rtol"])     # line 249
```

and the floor (`_dtype_floor`, lines 288–357) is

```python
eps   = float(torch.finfo(dtype).eps)     # TypeError -> {"atol":0.0,"rtol":0.0} for int/bool
scale = sqrt(sum(y**2)/n)                 # RMS|y|
return {"atol": eps * scale, "rtol": eps}
```

`required_matched_ratio = 0.99` on all 3717 workloads. The comparison rule (`src/sol_execbench/core/bench/correctness.py:127-141`) is elementwise `|x−y| ≤ atol + rtol·|y|` with `matched_ratio ≥ 0.99`; it is **not** a max-error test, and `max_error_cap` is null on the problems examined.

### 2.2 Why that becomes a bit-identity requirement

**Measured, corpus-wide:** run-to-run `max_abs == 0.0` **exactly** on **3581 / 3717 = 96.34%** of workloads (nonzero: n=136, min 1.490e-08, median 3.906e-03, max 6.226e+05). For that 96.3% the calibration measured nothing, so the shipped tolerance is *100% floor*: 3502 of 3717 sit at their own dtype's epsilon.

At an element of magnitude ≈ RMS the floored bound is `eps·RMS + eps·|y| ≈ 2·eps·RMS` — about one ULP of the output dtype. The floor is per-*output-dtype*, so what "one ULP" costs varies by 65 536×:

| floor dtype | workloads | rtol | median atol |
|---|---|---|---|
| bfloat16 | 2030 | 7.8125e-03 | 7.805e-03 |
| float32 | 1291 | **1.1921e-07** | 1.680e-07 |
| float16 | 320 | 9.766e-04 | 3.563e-02 |
| int64 / int32 / bool | 48 / 16 / 12 | **0.0** | **0.0** |

`rtol` is *exactly* `torch.finfo(float32).eps` on **1224 of 3717 (32.9%)** workloads. Integer/bool outputs get `atol = rtol = 0` — literal bit-identity of index tensors — because `torch.finfo` raises `TypeError` and the `except` path returns zeros. That zero is then applied to the problem's *float* outputs too.

### 2.3 What torch.compile actually does to the bits

Four distinct Inductor behaviours were isolated, each proved causally rather than argued:

1. **Reduction re-association.** `L2__009`: Inductor emits `triton_red_fused_add_mean_mul_pow_rsqrt_0`, a looped `[XBLOCK, R0_BLOCK]` fp32 accumulator over 2048 elements + a tree reduce; ATen's `mean.dim` uses a different block factor. Both accumulate in fp32. Measured: `x.pow(2)` **bit-identical (0 / 2,097,152 differ)**; `x.pow(2).mean(-1)` differs on **351 / 1024** rows by 2.384e-07.
2. **FMA contraction.** `L1__067` / `L2__009`'s rope: `(x*cos) + (rot*sin)` is emitted with the same association, but the AMDGPU backend contracts the first multiply-add into one `v_fma_f32`. Bit-exact attribution: of the 141,089 differing elements, **100.00%** bit-equal `fma(x, cos, fp32(rot*sin))`; the other association matches only 22.93%. Eager bit-equals the two-roundings model on 1,048,576 / 1,048,576.
3. **Elided intermediate dtype rounding** (`codegen_upcast_to_fp32`). `L1__062`: eager materialises `grad_key_states_rotated * cos_expanded` as bf16; the fused kernel keeps it fp32 and rounds once. Causal proof: `torch._inductor.config.emulate_precision_casts = True` restores **bit-identity with eager on all six outputs (0.000e+00 everywhere)**. Same for `L2__058` (`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` → `mr=1.000000 PASS`) and `Quant__004` (`codegen_upcast_to_fp32=False` → `n_diff 0/3,670,016`).
4. **GEMM template swap — max-autotune only.** Under `max-autotune-no-cudagraphs`, `L1__067`'s attention bmms stop being `extern_kernels.bmm` (hipBLASLt, bit-identical to eager) and become `triton_tem_fused_bmm_0`: qk matmul max_abs **1.5259e-05 on 74.93% of elements**, versus **0.000e+00** under default. `ALLOW_TF32=False`, `ACC_TYPE='tl.float32'` — a tiling/accumulation-order change, not a precision change. This is why v3 fails 571 workloads to v2's 523.

### 2.4 Tolerance vs divergence, in orders of magnitude

| problem | floor | atol | measured compiled-vs-eager | ratio | matched_ratio |
|---|---|---|---|---|---|
| `L2__009` | fp32 | 1.223e-07 … 1.375e-07 | 2.0e-06 … 5.8e-06 (8–40 ULP) | ~15–45× | 0.415–0.806 |
| `L1__067` | fp32 | 4.706e-09 … 3.446e-08 | 1.19e-06 … 2.03e-06 | 35–405× | 0.249–0.481 |
| `Quant__004` | bf16 | 4.61e-03 … 4.85e-03 | 1.31e-01 … 2.03e-01 | 28–43× | 0.734–0.743 |
| `L2__058` | bf16 | 4.645e-03 … 4.666e-03 | 1.56e-02 … 2.34e-02 | 3–5× | 0.985–0.989 |
| `L1__062` | bf16 | 7.81e-03 … 8.26e-03 | 3.13e-02 … 1.25e-01 | 4–16× | 0.977–0.996 |

The tolerance multiple needed to pass is small and problem-specific: **k=8** for `L2__009` and `Quant__004`, **k=2** for `L2__058` and `L2__078`, **k=∞ (never)** for `L2__049` where the tolerance is literally 0.

### 2.5 The self-consistency check that never fired

`calibrate_tolerance.py` runs both executions **in one process on one GPU back to back**, which holds hipBLASLt/MIOpen algorithm selection roughly constant. Across processes it does not:

> `L2__051_..._hyena_complete_forward_block`: all 16 workloads measured `run_to_run max_abs = 0.000e+00`, so all 16 got `atol ≈ 1.198e-07`. **`v1_eager` — the unmodified reference (`variants.py: def _eager(src): return src`) — then fails 10 of those 16 workloads with `INCORRECT_NUMERICAL` in `artifacts/06/candidates`.**

The reference fails its own tolerance. Nothing in task 05 checks that a derived tolerance is achievable by a correct implementation; the guard rail runs one way only ("never loosen a tolerance to make a kernel pass").

The float64 golden that *could* have caught this is computed, recorded and discarded: `entry["vs_golden"]` is written at lines 226–241, and the tolerance at line 248 reads only the run-to-run loop. Worse, it is unusable as recorded — `gen_golden.py:98-99` does `torch.manual_seed(0); prepare_inputs(..., device="cpu")` while `calibrate_tolerance.py:162-163` does the same at `device="cuda:0"`, so the two consume **different random draws**. Measured on `L1__067`: seed-0 CPU vs seed-0 CUDA `hidden_states` differ by 7.096e+00. Consequently **2302 of 2331 recorded goldens (98.8%) exceed the derived atol** and mean nothing.

### 2.6 The other half of the story: `recompile_limit = 8`

`reference/tb-candidates/variants.py:63` builds **one** module-level compiled callable per problem and reuses it for every workload:

```python
_solb_compiled = _solb_torch.compile(_solb_ref_run, mode={mode_arg}, dynamic=False)
```

`torch._dynamo.config.recompile_limit` prints as **8** in the container (`fail_on_recompile_limit_hit = False`). After the 8th guard configuration dynamo logs `torch._dynamo hit config.recompile_limit (8)` and **silently runs the frame eagerly**. Measured consequences:

* **225 of 235** problems have ≥9 distinct shape signatures; **2061 of 3957** workloads sit at or after the 9th.
* **522 of the 523** v2 failures are inside the compiled prefix; the sole exception is `L2__018_cu_seqlens_variable_length_vision_attention` at index 12, whose `total_seq_len=4096` duplicates index 0 (a guard hit) and which additionally graph-breaks on data-dependent control flow at `reference.py:109`.
* Pre-cliff fail rate **522 / 1776 = 29.4%**; post-cliff **1 / 1941 = 0.05%**.
* Re-run one workload per process, the "passes" are not passes: `L2__009` 8/8 fail, `Quant__004` 8/8 fail, `L2__058` 8/8 fail, `L1__067` 10/10 fail. **All five deep-dive problems fail 100% of their workloads when actually compiled.**
* **626 of 3717 published anchors** in `artifacts/06/authoritative` are labelled `v2_compile`/`v3_compile_max_autotune` and sit past the cliff — an eager latency wearing a compile label. Confirmation: on the 538 post-cliff workloads of the anchor-less problems that carry a recorded compile latency, `T_b / compile` is p50 **1.000**, p90 1.008, max 1.163.

**Note on the brief's premise for §6:** `time_tb_candidates.py:121-123` records `latency_ms_by_workload` **only for `status == PASSED`**. There is no recorded latency for a numerically-rejected workload. The DB rows that appear to be "FAILED with a latency" are the D28 inversion (§2.7).

### 2.7 Do not use `leaderboard/solbench.db` for any of this

```
db submission 2 FAILED rows: 585
artifacts/06 v2_compile FAILED: 523  PASSED: 3171
db FAILED that artifacts call FAILED: 0
db FAILED that artifacts call PASSED: 585
artifact failures with no db row at all: 523
```

Every one of the 585 carries a non-null `latency_ms` (`sum(latency_ms is null) = 0`), which today's `ingest.py` cannot produce. `db_built_utc = 2026-08-06`; TODO.md records the D28 fix as landing **2026-08-07**. The lead's framing — "585 FAILED spanning 69 problems" — describes the D28 artefact and names the *passing* workloads. Ground truth is `artifacts/06/candidates`. (The coincidence is real but unrelated: 69 is also the size of `v2 ∩ v3` numerical-failure problems.)

**But the live board is not affected, and an earlier version of this section invited the opposite conclusion.** `part_databases()` (`leaderboard/app.py:242`) reads `db/solbench-<PART>.db` first and falls back to the single-file layout only for a part the per-part layout has not produced. `db/solbench-MI350X.db` exists, built 2026-08-11T18:25Z, and carries `523` not-`PASSED` v2 rows — matching `artifacts/06` exactly — with `0` of them carrying a latency. Verified against the running server: `GET /api/v1/submissions/baseline-v2-compile/problems/L2__009_...` returns `{INCORRECT_NUMERICAL: 8, PASSED: 8}`. So the stale file is a pre-split leftover that nothing serves and that is gitignored, and no re-ingest is required. It is still worth deleting: if `db/` were wiped the fallback would serve it, and `run.sh`'s rebuild guard would count it as a database — only the freshness banner catches that.

---

## 3. Why these ~70 and not the other ~150

### 3.1 The population, exactly

| variant | INCORRECT_NUMERICAL | problems | other |
|---|---|---|---|
| `v1_eager` | 10 | 1 (`L2__051`) | 240 INVALID_REFERENCE (15 NVFP4) |
| `v2_compile` | **523** | **71** (L1 24, L2 41, Quant 6) | 240 INVALID_REFERENCE |
| `v3_compile_max_autotune` | **571** | **80** | 10 RUNTIME_ERROR + 240 INVALID_REFERENCE |
| `v4_contiguous` | 29 | 2 | 240 INVALID_REFERENCE |
| `v5_compile_contiguous` | 0 | — | **3717 RUNTIME_ERROR** (excluded from the board) |

`v2 ∩ v3` numerical = **69** problems; union = 82. **70 problems** carry no compile variant at all in `artifacts/06/authoritative`, the directory `build_manifest.py --t-b` reads. The 240 `INVALID_REFERENCE` are the deferred NVFP4 set and must never be pooled with the compile failures.

### 3.2 The discriminator, and how good it is

Measured on the pre-cliff population (the 1776 workloads torch.compile actually compiled):

| discriminator | yes: F/P | fail rate | no: F/P | fail rate | precision | recall | accuracy | OR |
|---|---|---|---|---|---|---|---|---|
| `rtol == fp32 eps` | 339 / 283 | **54.5%** | 183 / 971 | 15.9% | 0.545 | 0.649 | 0.738 | **6.36** |
| output floor `float32` | 339 / 317 | 51.7% | 183 / 937 | 16.3% | 0.517 | 0.649 | 0.718 | 5.48 |
| `atol == 0` (int/bool) | 11 / 29 | 27.5% | 511 / 1225 | 29.4% | — | 0.021 | — | 0.91 |

Fail rate by tolerance-floor dtype, pre-cliff:

| floor | n | failed | rate |
|---|---|---|---|
| float32 | 656 | 339 | **51.7%** |
| int64 | 24 | 11 | 45.8% |
| bfloat16 | 988 | 172 | 17.4% |
| float16 | 92 | 0 | **0.0%** |
| int32 / bool | 8 / 8 | 0 / 0 | 0.0% |

Problem level (220 scoreable problems): float32 **44 / 81 = 54.3%**, bfloat16 25 / 122 = 20.5%, int64 2 / 3, float16 0 / 12.

**So the best single-feature classifier is "the tolerance floored at fp32 epsilon", and it is a 6.4× odds shift with 73.8% accuracy — useful, and not sufficient.** It explains 339 of 523 failures (64.8%); both off-diagonal cells are large.

### 3.3 The exceptions, named — and the correction that matters

**Cell B — fp32-floored yet passes all 16 workloads: 37 problems, 592 workloads.** This is the deep-dives' "the tolerance forbids any reordering" claim, refuted by measurement:

* `L1__088_rotary_position_embedding_application` computes literally `query*cos + rotate_half(query)*sin` — the same expression `L1__067` fails on, same fp32, same FMA contraction. Measured on all 8 pre-cliff workloads: **`c/e = 4.768e-07`, identical to `L1__067`'s first-divergence figure — and `matched_ratio = 1.000000`, all PASS**, at `atol = 1.191e-07`.
* `L1__078_group_norm_fusion` / `07da4dac`: atol 1.695e-07, **413,789 / 1,310,720 elements bitwise different**, max_abs 1.907e-06, `mr = 0.996527` → PASS.
* `L1__054`: 328,986 / 2,097,152 differ, mr 0.999680 → PASS. `L2__051`: 341,604 / 1,048,576 differ, mr 0.998967 → PASS (while its *eager* variant fails 10/16 in a fresh process).

**Cell C — not fp32-floored yet fails: 27 problems, 184 workloads**, almost all bf16 at `rtol = 7.8125e-03`, including `L2__015` (atol 184–320) and `L2__056` (atol 0.18–1.39) where a 1-ULP story cannot apply.

**Cell D — zero tolerance.** `L2__049_group_limited_topk_routing` and `Quant__011_fp8_moe_gate_routing` return an int64 index tensor first, so `_dtype_floor` returns `{"atol":0.0,"rtol":0.0}` and that zero is applied to the fp32 output as well. Measured `L2__049 / d6d0eb83`: `topk_idx` **bit-identical (0 / 16,384 — no routing flip)**, `topk_weight` off by exactly one fp32 ULP (1.1920929e-07) on 4,488 / 16,384, `mr = 0.726074`, **`min_k_to_pass = None` searched to 2^20**. 11 workloads. Upstream gives `L2__049` atol 0.56–0.84 / rtol 0.01 / ratio 0.98.

**Cell E — threshold noise.** Both of `L2__015`'s recorded failures **pass** on isolated re-run: mr 0.990831 and 0.990732 against 0.99 required (max_abs 512.0 = one bf16 ULP at ref absmax 116,224). One verifier also measured `L2__015 / ee0f7e21` eager-vs-eager `= 6.400e+01`, contradicting the "the reference is deterministic" premise for that problem.

### 3.4 The discriminator I would actually state

The deep-dives' headline ("the tolerance is one ULP, therefore any reordering fails") is **refuted** by cell B and by these direct counterexamples:

* `L2__005_swiglu_mlp_backward` — same SwiGLU op, same bf16 floor, eager bit-deterministic — diverges by `max_rel = 1.065` (**106% relative on the worst element**) and gets `mr = 1.000000` PASS on 4/4 workloads.
* `L2__059_decoder_layer_full_block` — bf16, `F.silu`, 11 Inductor kernels, 0 graph breaks — diverges by 3.9062e-02 on **62.2%** of elements, *2.5× larger than `L2__058`'s failing divergence*, and passes at `mr = 0.994533`, 0/16 failures.

The mechanism is ubiquitous; the *verdict* is decided by the **fraction of elements pushed past `atol + rtol·|y|` against the 1% budget**. Because `rtol·|y| ≥ ulp(y)` by construction, a genuine one-ULP difference always passes. What fails is error landed on **sub-RMS elements**, where the RMS-scaled `atol` floor is many ULP too small for that element's own magnitude. Measured on `L2__058`: over all output elements median `|d| / ulp(ref) = 1.000`, p99 = 50; the failing 1.27% have median `|ref| = 9.326e-02 = 0.157 × RMS` and median `|d| = 6.348e-03 = 12 ULP of their own value`. That is why fp32 (floor 1.2e-07, deep reductions, heavy near-zero mass after a projection) fails at 52% and bf16 (floor 7.8e-03) at 17%, and why fp16 — floor 9.8e-04, mostly shallow elementwise problems — fails at 0%.

And a fourth reason some problems pass that has nothing to do with numerics: **Inductor compiled nothing.** `L1__044_moe_expert_computation` (bf16, `F.silu`) passes with `c/e = 0.000e+00` because `generated_kernel_count = 0, graph_breaks = 1`; `L2__010` likewise (4 kernels, 2 graph breaks, `c/e = 0.000e+00`).

---

## 4. The five examples

### 4.1 `L2__009_decoder_layer_with_residual_connections` — the pure fp32-floor case

**Computes:** a full decoder layer, fp32 end to end — 4× RMSNorm, 8+ `F.linear`, two attention matmuls, two softmaxes, top-8 MoE routing over 128 experts with `index_add_`.

**Numbers:** `atol` 1.223e-07 … 1.375e-07 (`= fp32eps × RMS|y|`), `rtol` 1.1920929e-07 exactly. Eager-vs-eager `0.000e+00` on all 16. Compiled-vs-eager 2.0e-06 … 5.8e-06 (8–40 ULP), `matched_ratio` 0.415–0.806. **16/16 fail** when each is compiled in its own process (the recorded 8 "passes" are dynamo fallbacks; isolated, they give mr 0.424–0.590). Tolerance multiple needed: **k=8** (mr 0.994260); k=4 still fails at 0.922.

**Where it diverges:** the first op, `rms_norm(...)` at `reference.py:71`. `x.pow(2)` bit-identical (0 / 2,097,152); `.mean(-1)` differs on 351 / 1024 rows. A verifier's decisive extra test: monkeypatching *only* the first `rms_norm` to return the Inductor tensor and running the remaining ~20 ops in eager gives max_abs 5.722e-06 / mr 0.479438 → FAIL. **One reduction is sufficient.**

**fp64 verdict — a coin flip.** RMS error vs a float64 CPU golden: eager **1.2443518e-06**, compiled **1.2442062e-06** (compiled better by 0.012%); max_abs eager 1.164e-05 vs compiled 1.020e-05; **970,753 elements closer under eager, 970,569 under compiled, 155,830 tied (50.005% / 49.995%)**. For scale: the reference's own RMS error is 9.05 ULP while the tolerance demands ~1.

**Two corrections the deep-dive did not make.** (a) There are **two independently sufficient sites**, not one: compiling *only* `apply_rotary_pos_emb` — purely pointwise, no reduction anywhere — fails on its own (max_abs 2.622604e-06, mr 0.616300), by FMA contraction, and against fp64 that fragment is **decisively better** than eager (RMS 4.802e-08 vs 5.190e-08; 397,952 elements closer vs 120,633). (b) **The "20-op chain" is not compiled.** With a fresh Inductor cache `torch.compile(run)` produces only 5 tiny graphs (three shape-specialisations of `rms_norm`, one `apply_rotary_pos_emb`, one `repeat_kv`), zero `extern_kernels.*`; every linear, both attention matmuls, both softmaxes, `topk` and the whole MoE loop run **eager**, because of a data-dependent graph break at `reference.py:187`. Compiling only those three helpers reproduces the full run to the last decimal (5.781650543212891e-06 / 0.41581106185913086). The deep-dive's prefix-localisation table therefore measures a *different program* from the one the board scores; its conclusion survives, its per-stage attribution does not.

### 4.2 `L1__062_kv_cache_update_with_rope_backward` — the atypical tail case

**Computes:** the backward of a RoPE'd KV-cache update: a gather, a bf16 elementwise multiply, an 8-term reduction over kv-heads, and two clone+scatter-zero cache grads.

**Numbers:** bf16 floor, `atol` 7.81e-03 … 8.26e-03, `rtol` 7.8125e-03 — **a thousand times looser than the `L2__009` case, and it still fails**. Eager-vs-eager 0.000e+00 on all 16. **1 of 16 fails** (`6c293638`, batch 1, new_seq 1, max_seq 256): `mr = 0.976562`, 3 bad elements of a **128-element** `grad_cos`. Over 10 seeds that same workload fails on **1 of 10**; the sibling 512-element workload drew 5 bad on seed 1 (mr 0.990234) — one element from failing.

**Where it diverges:** ops 1 and 2 are bit-exact; the divergence is born simultaneously at `grad_k1_total` and `grad_cos`, both consumers of an elided bf16 intermediate. Proved bit-exactly (`eager == bf16-rounded model: True`; `compiled == fp32-throughout model: True`) and then **causally**: `emulate_precision_casts=True` restores bit-identity on all six outputs. `index_put_` / functionalisation is cleanly exonerated — `grad_value_states`, `grad_key_cache_input`, `grad_value_cache_input` are all 0 differing. Amplifier: catastrophic cancellation, condition number 42.7 at the worst index (eight terms summing to 0.1186 with `sum|t| = 5.0656`).

**fp64 verdict — compiled is strictly better, and this is the sharpest single result in the whole investigation.** grad_cos RMS 4.7270e-03 (compiled) vs 6.8427e-03 (eager); grad_sin 4.6575e-03 vs 5.7942e-03; grad_key_states 2.4671e-03 vs 3.3696e-03. At the three elements that trip the harness, compiled is 20–350× closer (idx 9: golden 0.1186237335, compiled 0.11865234, eager 0.12890625). And when the fp64 golden is passed through the reference's own final `.to(torch.bfloat16)`, **the compiled output is bit-identical to it on all six tensors while eager is not** — compiled is the correctly-rounded bf16 answer and eager is the wrong one.

**Generalisation refuted.** `L1__062` is the **only problem in the corpus with exactly one failing workload**; its 1/16 = 0.0625 is the minimum of the distribution. Median failure fraction over the 71 problems is 0.50; 55 of 71 fail ≥ half their workloads; 17 fail 100%. The "128-element output, Poisson noise on the 1% budget" story explains this problem and nothing else. Minor: the deep-dive's "T_b for all 16 is v4_contiguous" is wrong — `winner_by_workload` is v1_eager on 11 and v4 on 5; the looseness on `9aa4bf2f` is 0.156700 / 0.029040 = **5.396×**, not 5.30×.

### 4.3 `Quant__004_fp8_moe_expert_linear` — **distinct mechanism: a downstream quantizer as amplifier**

**Computes:** an fp8 (e4m3, BlockWise1x128) MoE expert MLP — quantize hidden and gate_up, fp32 GEMM, `gated = F.silu(gate) * up` in **bf16 and unquantized**, requantize *that* to fp8, second GEMM, bf16 output.

**Numbers:** the "fp8 problems get loose tolerances" premise is false — `rtol = 7.8125e-03` is exactly bf16 eps (the floor) and `atol = 7.8125e-03 × RMS|y| = 4.75e-03`, because the *output* is bf16 and the tolerance keys off the output dtype, not the internal quantization. Eager-vs-eager 0.000e+00 on all 16. **16/16 fail** when actually compiled: max_abs 1.31e-01 … 2.03e-01 (**28–43× atol**), `mr` 0.734–0.743.

**Where it diverges and why it is different.** Every fp8 quantize, dequantize and both fp32 GEMMs of the first projection are bit-identical (`extern_kernels.mm`, `ALLOW_TF32=False`; isolated GEMM 0 / 4,194,304 differing; isolated fp8 codes 0 / 3,670,016). The divergence is elided bf16 rounding — but **not at one site**: barrier experiments show four (gemm1 cast, silu intermediate, the `silu*up` store, gemm2 cast). Restoring only the silu rounding leaves mr at 0.798480 (still FAIL); three of four are needed to pass; `codegen_upcast_to_fp32 = False` gives `n_diff 0/3,670,016, mr = 1.000000`. The deep-dive's quoted kernel `triton_poi_fused_mul_silu_0` does not exist in the real graph (`grep -c` = 0); the real one is `triton_red_fused__to_copy_abs_amax_clamp_div_mul_silu_split_unsqueeze_view_4`, and its 18-row stage table is invalidated by materialisation (returning an intermediate forces the very rounding being probed — in the real graph `gate_up_output` is never bf16; `buf5` is fp32).

**The amplifier — this is the callout.** A 1-bf16-ULP (2^-7) perturbation straddles fp8 e4m3 rounding boundaries: **2.93–3.12% of activation codes flip by exactly one code, and one e4m3 ULP is 2^-3 = 12.5% relative.** Same GEMM with the activation quantizer removed: output gap mean 1.102e-03 and 4.36% of elements over atol; with it, **4.832e-03 and 32.88%** — the requantizer multiplies the mean error **4.38×** and the over-tolerance fraction **7.5×**. That is why this problem misses by 30–40× atol where an otherwise-identical SwiGLU (`L2__005`, max_rel 1.065) passes at `mr = 1.000000`. All six failing Quant problems (004, 011, 012, 013, 016, 017) quantize an intermediate.

**fp64 verdict — compiled is better:** mean_abs 6.871507e-03 (compiled) vs 7.446866e-03 (eager), ratio **0.9227**; compiled also wins on max_abs and rms. Compiled closer on 46.93% of elements, eager on 41.67%, tied 11.40%.

**The counter-case that must go on the record.** `L1__074_fused_gated_mlp_silu` — the same SwiGLU pattern in **pure fp32**, no bf16 anywhere, so this mechanism cannot apply. Its `F.linear` is bit-identical (0 / 16,777,216); 100% of the divergence is `up*(gate*sigmoid(gate))` — 671,517 / 8,388,608 elements, max_abs 1.907e-06 — `tl.sigmoid` vs `aten::sigmoid`. Against fp64: eager RMS 8.601942e-07, compiled 8.602856e-07, **ratio 1.0000933 — compiled is very slightly worse**, and eager is closer on more elements (35.92% vs 35.84%).

### 4.4 `L2__058_mamba2_selective_scan` — **distinct mechanism: a long recurrence that does *not* amplify**

**Computes:** Mamba-2 chunked selective scan — in-proj, depthwise conv1d, SiLU, softplus dt, segment-sum/cumsum, four einsums, inter-chunk recurrence over 32 chunks at seq 4096, RMS gate, out-proj. bf16 in/out with fp32 internals.

**Numbers:** bf16 floor, `atol` 4.645e-03 … 4.666e-03, `rtol` 7.8125e-03. Eager-vs-eager 0.000e+00; `artifacts/05` records `run_to_run {max_abs: 0.0, max_rel: 0.0}, deterministic: true` on all 16. **16/16 fail** when compiled, `mr` **0.9854–0.9889 against a required 0.99** — a 1.3-point miss, not a blow-up. **k=2 clears it.**

**Where it diverges:** first at `(conv_out * sigmoid(conv_out))` — `01_projected` (matmul) and `05_conv_out` (conv1d) are **0.000e+00**, and at stage 06 `eager == bf16-intermediate SiLU` and `compiled == fp32-intermediate SiLU`, both bit-exact, differing by 3.125e-02 (one bf16 ULP **in the binade [4,8)** — the deep-dive said [8,16), which is wrong by one) on 26.34% of elements.

**The scan does not amplify — this was tested and falsified.** Along 32 chunks at seq 4096, the inter-chunk recurrence error relative to scale goes 1.419e-03 (chunk 1) → 9.047e-04 (4) → 4.587e-04 (16) → **6.142e-04 (32)**: an amplification factor of ~0.43×, i.e. it *decays*. Reductions average independent per-element perturbations down; the bf16 casts push it back to ~1 ULP. Anyone reaching for "recurrences accumulate error" as the explanation should stop.

**Correction on attribution.** The first divergence is not the dominant one. Emulating the fp32-intermediate policy one site at a time (mr vs the compiled output; baseline 0.987285): conv SiLU only → 0.989835; softplus only → 0.987597 (nothing); **the final gate/norm SiLU chain only → 0.995301**; all three → 0.997181. The tail contributes ~3× the conv SiLU. And the fullest hand emulation reaches mr 0.990262 — which would **pass** — while the true compiled sits at 0.987285, so ~0.3 points are unattributed by the stage story. **Cite the config control, not the stage table:** `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` → `mr = 1.000000 PASS`, 99.8168% bit-identity.

**fp64 verdict — compiled is better, and eager fails its own gate.** Workload `4c88b9e7`: eager vs golden RMS 2.7785e-03, compiled **2.1146e-03** (eager 31% worse); `c924af2c` reproduces at ratio 1.314 to two decimals. Compiled strictly closer on 41.73% of elements, eager on 24.26%, tied 34.01%. **Eager's matched_ratio against the golden, under this workload's own tolerance, is 0.986859 — below the 0.99 it enforces on others.** Compiled scores 0.998504.

Two Inductor differences that exist here and are *benign*, worth naming so nobody rediscovers them as contradictions: fp32 transcendentals differ from ATen (exp 53.3% bit-exact, sigmoid 78.0%, softplus 62.3%, rsqrt 89.1%, up to 11 fp32 ULP) and the fp32 cumsum uses a different order (16.8% bit-exact) — injected end-to-end they give mr 1.0 / 99.77% and 99.90% bit-exact.

### 4.5 `L1__067_flash_attention_gqa_ultralong` — the fp32 attention case, and the max-autotune third mechanism

**Computes:** GQA attention, fp32 throughout, seq 128 → 16384: QKV projections, RoPE, `repeat_kv`, QK matmul + scaling, 2-D causal mask, softmax, attn·V, out-proj.

**Numbers:** fp32 floor; **`atol` shrinks with sequence length — 3.446e-08 at s=128 down to 4.706e-09 at s=16384 — because it is `eps × RMS|y|`** and the output RMS falls as attention averages over more positions. Eager-vs-eager 0.000e+00 on all 18. **18/18 fail** when compiled (the recorded 10 passes are the post-cliff fallbacks; re-run in batches of five they give `c/e` 1.19e-06 … 1.91e-06, `mr` 0.249–0.481).

**Where it diverges:** `reference.py:41`, the RoPE line — 4.7684e-07 on 13.46% of elements, already **13.8× atol** at the first site. Growth: 9.54e-07 after the QK matmul (~2×, contracting 128 perturbed terms), *shrinks* to 2.38e-07 after softmax, then **2.03e-06 / 94.15% of elements after the out-proj** (4096-term contraction). All three F.linear and both bmms are `0.0000e+00` under default. **SDPA is falsified**: zero `scaled_dot_product` / `_flash_attention` symbols in the output code; the `fuse_attention` pattern cannot fire because the mask is 2-D while query is 4-D.

**fp64 verdict — a tie, and the reference is nowhere near its own tolerance.** RMS ratio compiled/eager **0.9995** (`b0c05812`) and 0.9997 (`fd115ea8`); **max_abs identical to all seven digits (8.094737e-06)**; elementwise 47.16% / 5.85% / 46.99%. **Eager's matched_ratio against the golden under this workload's own tolerance is 0.089854** — the reference misses its own gate by 11×. Median error vs truth is 26.6 ULP for both, against a tolerance demanding ~1.

**Three corrections.**
1. **Max-autotune is a different mechanism.** Under `max-autotune-no-cudagraphs` the extern bmm is replaced by `triton_tem_fused_bmm_0` and the matmuls are **not** bit-identical: qk raw max_abs **1.5259e-05 on 74.93% of elements**, qk×scaling 1.4305e-06, attn·V 2.3842e-07, versus 0.000e+00 for all three under default. Winner `triton_bmm_4 ... BLOCK_K=64, matrix_instr_nonkdim=16, ACC_TYPE='tl.float32', ALLOW_TF32=False`. So v3's extra 48 failures are partly a GEMM-implementation swap, not the fp32 story.
2. **The exact divergence figures are not reproducible across compile sessions.** Same uuid, same bitwise-identical inputs, default mode, three Inductor cache states: 2.0265579e-06 / mr 0.386650, 1.9073486e-06 / 0.386724, 1.5497208e-06 / 0.412093. The verdict never moves, but default-vs-max-autotune column differences of that size are inside compile-session noise and must not be attributed to the mode.
3. **The failure is over-determined.** With `cos=1, sin=0` — making the RoPE line arithmetically inert under any association or contraction, with the compiled graph otherwise unchanged — the end-to-end divergence is still 9.5367e-07 on **92.27%** of elements, ~28× atol. Killing the FMA contraction would not make this problem pass; softmax alone fails the gate.

---

## 5. Benchmark defect, or a real torch.compile failure?

**On every case where anyone actually adjudicated it against a float64 golden, eager is not the more accurate implementation. On five of six, it is the less accurate one.**

| problem | eager vs fp64 | compiled vs fp64 | verdict |
|---|---|---|---|
| `L2__009` | RMS 1.2443518e-06 | RMS 1.2442062e-06 | tie (970,753 / 970,569 / 155,830 tied) |
| `L1__062` | grad_cos RMS 6.8427e-03 | **4.7270e-03** | compiled better; **bit-identical to the correctly-rounded bf16 golden** |
| `Quant__004` | mean_abs 7.446866e-03 | **6.871507e-03** (0.9227×) | compiled better |
| `L2__058` | RMS 2.7785e-03 | **2.1146e-03** (0.761×) | compiled better |
| `L1__067` | RMS 6.123068e-07 | **6.120029e-07** (0.9995×) | tie |
| `L1__074` (control) | RMS 8.601942e-07 | 8.602856e-07 (1.0000933×) | **eager marginally better** |

And the reference fails its own gate against truth wherever that was checked: `L1__067` eager `matched_ratio = 0.089854` (required 0.99), `L2__058` eager 0.986859, and — with no golden involved at all — `v1_eager` re-run in a fresh process fails **10 of 16** workloads of `L2__051` under the tolerance task 05 derived for it.

**Conclusion, stated plainly: on the measured cases the benchmark is grading agreement-with-eager, not correctness.** The gate as shipped asks a submission to reproduce a particular kernel schedule's *rounding sequence*, including its errors, on 99% of elements. A numerically better implementation is scored `INCORRECT_NUMERICAL` for being better. That is a defect in the tolerance derivation, not a defect in `torch.compile`.

Three honest qualifications, because this conclusion is load-bearing:

1. **Coverage is thin.** fp64 goldens were computed for ~10 workloads across 6 problems, out of 523 failing workloads. The direction is consistent and the mechanisms (removing a rounding, contracting an FMA) predict it, but it is 2% of the population and one control already came out the other way.
2. **Not all failures are rounding.** `L2__049` / `Quant__011` have a *zero* tolerance from an `except TypeError` path — no accuracy argument applies, and no multiplicative widening can ever fix them (11 workloads). `L2__015`'s recorded failures do not reproduce (threshold noise). Under max-autotune, `L1__067`'s GEMM template swap is a genuine implementation change. Four distinct mechanisms, minimum.
3. **The other reading is coherent.** If the reference source *is* the specification — "PyTorch eager semantics, as written" — then eager is correct by definition and compiled deviates. Under that reading the benchmark is measuring rounding-sequence fidelity rather than correctness, and no optimizing compiler can pass it. Pick one; the current artifacts do not say which was intended, and `tasks/05` never asked.

---

## 6. What it means for scoring

`S(T_k) = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))`, and `T_b` is the fastest of `{v1_eager, v2_compile, v3_compile_max_autotune, v4_contiguous}` that passes **every workload of the problem** (`time_tb_candidates.py:148`: `if not r.get("ok") or not r.get("all_passed"): continue`). One failing workload disqualifies a variant on *all* of that problem's workloads.

**Exposure.** 70 problems have no compile variant in `artifacts/06/authoritative` → **1115 of 3717 scoreable workloads (30.0%)** are anchored to eager-class PyTorch only (`t_b_variant`: 605 `v1_eager`, 510 `v4_contiguous`). Corpus-wide, compile wins the anchor on 44.2% of workloads where it is eligible, so this is not a marginal variant.

**Correcting the brief:** `artifacts/06` records a latency **only for workloads that passed** (`time_tb_candidates.py:121-123`). There is no recorded compiled time for a numerically-rejected workload. What *can* be read off is the compile time on workloads a disqualified variant **passed** — i.e. times that were real measurements of a variant the harness then rejected problem-wide.

**Measured inflation.** On the 70 anchor-less problems, 577 of 1115 workloads carry such a latency. Splitting them by the recompile cliff (identical under both definitions — the 9th distinct shape, and raw index < 8):

| population | n | `T_b / compile` p50 | p90 | max | frac > 1.5 |
|---|---|---|---|---|---|
| pre-cliff — genuinely compiled | **39** | **2.023** | 5.318 | **6.288** | 0.615 |
| post-cliff — dynamo eager fallback | 538 | 1.000 | 1.008 | 1.163 | 0.000 |

The 538 confirm the fallback (they *are* eager times). The 39 are the honest measurement: **the published anchor is a median 2.02× and up to 6.29× slower than a real compiled time that was measured and then thrown away.** By problem (max ratio): `L1__061` 6.27×, `L1__062` 5.75×, `L1__094` 5.12×, `L1__014` 3.69×, `L2__054` 3.37×, `L2__040` 3.30×, `L2__007` 1.96×, `L2__015` 1.46×, then eight below 1.2×.

**Score effect at the anchor.** For those 39 workloads, an agent that exactly ties the published `T_b` scores 0.5 by construction; under the corrected anchor it would score **median 0.312, p25 0.181, min 0.113** — a median **0.188 points of pure inflation at S = 0.5**, and none of the corrected anchors falls below T_SOL.

**Extrapolation to the other 1076.** Not measured — on those workloads compile never both compiled *and* passed. Two bracketing estimates: the anchor tightening compile delivers on problems where it *was* eligible, pre-cliff, is p50 **1.243×** (p90 4.629×, max 14.918×, n=1208); the tightening measured on the 39 is **2.023×**. Applying each uniformly to all 1115:

| tightening | S at the published anchor becomes | workloads whose corrected T_b would fall **below T_SOL** |
|---|---|---|
| 1.243× | p50 0.439, mean 0.402 | 57 / 1115 |
| 2.023× | p50 0.318, mean 0.282 | **272 / 1115 (24%)** |
| 4.629× | p50 0.157, mean 0.142 | 458 / 1115 |

That last column is a second finding: on a quarter of these workloads a compiled-PyTorch anchor would sit **under the published T_SOL**, which means the bound there is not a lower bound. That overlaps D39/D42 and should be treated as further evidence for them.

**Effect on real published scores** (re-scoring the actual agent submissions in `leaderboard/solbench.db`, holding each submission's measured latency fixed and moving only `T_b` on the 70 problems; workloads whose corrected anchor falls below T_SOL are dropped from both means):

| submission | n scored | on the 70 | mean S now | @1.243× | @2.023× |
|---|---|---|---|---|---|
| 1 — PyTorch eager | 3644 / 3429 | 1058 / 843 | 0.4541 / 0.4513 | 0.4257 (**−0.0285**) | 0.3977 (**−0.0537**) |
| 5 — agent (largest run) | 373 / 362 | 103 / 92 | 0.6589 / 0.6641 | 0.6330 (−0.0259) | 0.6059 (**−0.0582**) |
| 6 — agent | 74 | 17 | 0.7757 | 0.7719 (−0.0038) | 0.7608 (−0.0149) |
| 7 — agent | 59 / 53 | 27 / 21 | 0.7011 / 0.7176 | 0.6677 (−0.0334) | 0.6689 (−0.0487) |

Restricted to the affected problems alone, submission 5's mean S there is **0.546 → 0.423 → 0.274**. Against a published benchmark score of 0.6341 whose entire re-timing debate turned on ±0.001 (STATE.md, the 8-wide/serial revert), **a −0.026 to −0.058 systematic inflation is 25–60× that margin.**

**Plus a labelling defect that inflates nothing but measures nothing either:** 626 of 3717 published anchors are stamped `v2_compile`/`v3_compile_max_autotune` and sit past the recompile cliff. Their `T_b` is an eager latency with a compile label, and the true compiled time on those 2061 workloads has never been measured, in either direction.

---

## 7. Options

Prime directive 7 applies to every one of these: none may be improvised. Whatever is chosen goes into `STATE.md` first, with the reasoning, and produces **manifest v1.3** — the frozen v1 and the served v1.2 are not edited.

**A. Fix `recompile_limit` and re-run task 06. Cost: low. Ambiguity: none.**
Set `torch._dynamo.config.recompile_limit = 64` (or build one compiled callable per workload) in `reference/tb-candidates/variants.py`, re-run the candidate sweep 8-way and the authoritative re-time on GPU 0. This is an unambiguous bug, not a methodology change — the harness intended to time `torch.compile` and timed eager on 2061 workloads. *Consequence to expect:* the failure count goes **up**, not down: all three index-8 workloads tested in fresh processes (`Quant__004/0278bef5`, `L1__074/0c1b78fe` mr 0.917327, `L1__022/678705cd` mr 0.691406) failed. Cost: ~5.5 h serial on GPU 0 at the measured 1.5 min/problem, plus the sharded sweep on GPUs 1–7. **Do this first and separately from anything else, so its effect is legible.**

**B. Calibrate the tolerance against reformulation, not reseeding.** Derive `max_atol`/`max_rtol` from the spread across the *four T_b formulations* (eager, contiguous, compile, max-autotune) plus seeds and processes — the perturbation class the harness is actually exposed to — rather than the same kernel schedule twice in one process. *Pro:* directly targets the defect; a tolerance the reference's own legal reformulations cannot meet is prima facie wrong. *Con:* it is exactly the "loosen a tolerance so a kernel passes" move `tasks/05` forbids, unless it is paired with a ceiling. *Pair it with:* a hard rule that the derived tolerance may never exceed some fraction (say 1/4) of the reference's own measured distance from a float64 golden — which turns "loosen" into "loosen only into the slack the reference already has", and is checkable. On `L2__009` that ceiling would be ~2.3 ULP against a needed k=8, so it would *not* let everything through — good, that is the point. *Re-measure:* all of task 05 (GPU, 235 problems × workloads × 4 variants × seeds), then task 06, then manifest v1.3, then re-ingest and re-score every submission (raw latencies unchanged; only S moves).

**C. Score against a float64 golden instead of eager.** *Pro:* it is the only formulation under which "correct" means correct, and it would end the whole class of dispute. *Con:* the golden infrastructure is currently unusable — the CPU/CUDA RNG mismatch means 2302 of 2331 recorded goldens are comparisons of two different problems, several problems raise on fp64 promotion (`L2__058` needs 6 `.float()` → `.to(float64)` edits and is arguably then not the reference), and `artifacts/golden/_report.json` already skips the largest workloads on an element cap. *Cost:* highest of any option, and it changes what the benchmark *is*.

**D. Fix the three narrow tolerance bugs regardless of A–C.** (i) `_dtype_floor`'s `except TypeError` path must not apply `atol=rtol=0` to a problem's *float* outputs; derive per-output tolerances. 76 workloads, ≥11 failures, and `L2__049` is unpassable by construction today. (ii) `gen_golden.py` must generate inputs on the same device as `calibrate_tolerance.py`. (iii) Record cross-process, not just in-process, run-to-run spread; `L2__051` is the ready-made reproducer (10/16 eager failures at an in-process-derived tolerance). All three are cheap and none is a methodology change in the directive-7 sense — they are defects with respect to what task 05 already says it does.

**E. Loosen `required_matched_ratio`.** Upstream used 0.98 on all 1299 L2 workloads and task 05 tightened it to 0.99 — a 2× reduction in permitted out-of-bound elements, on the category with the most failures. Reverting to 0.98 would recover the knife-edge cases (`L2__058` at 0.9854–0.9889, `L2__015` at 0.9905) and nothing else: `L2__009` sits at 0.42. *Cheap, and it addresses maybe 10% of the population.* It is a legitimate v1.3 item but not a fix.

**F. Accept and document.** Publish the anchor as "fastest *numerically eager-equivalent* PyTorch formulation", state that 30% of workloads have a softened anchor, and put the per-problem flag on the board next to `bound_quality`. *Pro:* costs nothing, breaks no comparability, and is honest. *Con:* it leaves a measured −0.026 to −0.058 mean-S inflation in every published number, and the leaderboard's own claim — that S=0.5 is "optimized PyTorch" — becomes false on 70 problems.

**What I would do: A, then D, then B, with F as the interim.** A is a plain bug with a plain fix and must not be bundled with anything else, because it moves the failure count in the opposite direction from B and the two effects would be inseparable afterwards. D is three small correctness fixes that are defensible under the existing task-05 spec. B is the real fix and the only one that makes `T_b` mean what the board says it means, but it costs a full task-05 + task-06 re-measurement and a new manifest, so it should be entered deliberately with the ceiling rule written down *before* the sweep starts, not chosen after seeing which tolerances it produces. Until B lands, ship F: mark the 70 problems on the board and in the manifest as `anchor_class: eager_only`, so a reader can see which scores rest on a soft anchor. Do not touch a tolerance to make a variant pass in the meantime.

---

## 8. What is still unknown

* **The true compiled latency on 1076 of the 1115 anchor-less workloads.** Compile never both compiled *and* passed there, so the T_b inflation on 96% of the affected population is an extrapolation (1.243×–2.023×), not a measurement. Option A produces the missing data as a side effect.
* **torch.compile has never been exercised on 2061 of 3957 workloads.** The failure count of 523 is a floor. Three index-8 probes all failed; that is 3, not 2061.
* **fp64 adjudication covers ~10 workloads of 523.** The "compiled is not less accurate" conclusion is measured on 6 problems, one of which (`L1__074`) came out the other way by 0.009%. There is no distributional claim here and I will not make one.
* **Whether the fp8-requantizer amplifier explains the other five failing Quant problems.** All six quantize an intermediate and all six use the SwiGLU shape, which is a strong prior. It was measured on `Quant__004` only.
* **`L2__015` is unresolved.** Its two recorded failures pass on isolated re-run (mr 0.990831 / 0.990732), and one verifier measured eager-vs-eager `= 6.400e+01` on `ee0f7e21` — which contradicts the recorded `run_to_run max_abs = 0.0` for that problem and, if real, means part of the failing population is nondeterminism the calibration did not see. That single measurement has not been reproduced.
* **The max-autotune GEMM-template mechanism** was measured on one problem (`L1__067`). It plausibly accounts for much of v3's extra 48 failing workloads; nobody has checked.
* **Compile-session reproducibility.** The same workload with a fresh Inductor cache gives divergences differing by 30% (2.03e-06 / 1.91e-06 / 1.55e-06). Autotuning on a co-tenanted GPU is the obvious suspect. No verdict changed in any observed instance, but it means small differences between recorded runs are not evidence of anything.
* **Whether raising `recompile_limit` changes any *published* `T_b`.** 626 anchors are mislabelled compile-but-eager; their times would move once compile really runs, in the tightening direction. Unquantified.
* **The 272 workloads (24% of the affected set) whose corrected anchor would fall below T_SOL.** Either the tightening estimate is too aggressive there, or those bounds are wrong. This intersects D39 (827 workloads at >100× headroom, bound never checked for tightness) and D42, and deserves its own look before B is designed.
* **D20 remains unexplained** and two upstream variance tests remain skipped behind it; nothing here touched it.

*Everything numeric above is reproducible from `artifacts/06/candidates`, `artifacts/06/authoritative`, `artifacts/05/workloads`, `artifacts/09/manifest-v1.2.json`, `leaderboard/solbench.db`, the census script at `/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/tolerance_census.py`, and the ~100 GPU artifacts under `/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/`. No repo file outside `artifacts/11/compile-diag/` was modified; `leaderboard/solbench.db` was read only.*