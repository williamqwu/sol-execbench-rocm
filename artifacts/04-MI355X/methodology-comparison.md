# Task 04 — `hip_events` vs `rocprof`

<!-- {"task": "04-methodology-comparison", "utc": "2026-08-14T20:42:36.520208+00:00", "git_sha": "a63ef836b2ab1edb301e5a1f2fb94db443b48992-dirty", "host": "mia1-p02-g46", "python": "3.12.3", "torch": {"available": true, "version": "2.9.1+rocm7.2.0.git7e1940d4", "hip": "7.2.26015-fc0010cf6a", "cuda": null, "device_count": 8, "devices": ["AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X"]}, "rocm": {"version": "7.2.0", "driver": "6.16.6", "amd_smi": "AMDSMI Tool: 26.2.1+fc0010cf6a | AMDSMI Library version: 26.2.1 | ROCm version: 7.2.0 | amdgpu version: 6.16.6 | hsmp version: N/A"}, "f_lock_mhz": null, "visible_devices": null} -->

Both methodologies time the same solution on the same inputs, back to back in one process, so this compares two ways of measuring and not two moments in the node's life. **Positive means `hip_events` read slower**, which is the expected direction: an event pair brackets the host launch and dispatch-level activity tracing does not.

Problems compared: **94** of 94; workload pairs: **1452**.

Conditions: arm order **hip_events,rocprof**. Settle state is not recorded in these artifacts; they predate the field.

| group | n | median | p10 | p90 |
|---|---|---|---|---|
| all workloads | 1452 | -2.62% | -79.3% | +2.9% |
| kernels >= 100 us | 990 | -2.06% | -80.3% | +0.6% |
| kernels < 100 us | 462 | -9.06% | -78.5% | +9.5% |

**Acceptance — median divergence on kernels >= 100 us: -2.06%** against a gate of 2%. FAIL.

**The median landed on the unexpected side of zero.** The predicted sign was positive — events include the launch, activity tracing does not — and the measured median is -2.06%, i.e. `rocprof` reads *slower*. The gap is about 2.1% at the median. It is reported rather than explained: whether about 2.1% is inside this node's run-to-run reproducibility is a question for that node's own stability measurement, which this script does not read and therefore does not quote.

Sub-100 us kernels are reported separately rather than folded in. There the median is -9.06%: a fixed launch overhead is a larger fraction of a smaller number, which is the finding, not an error.

## The tails are wide, and they are wide in both directions

451 of 1452 workload pairs differ by more than 20%, concentrated in 61 problems. The median is small because most iterations dispatch one kernel; the tail is where they do not.

* **`hip_events` much slower** (up to +26%) on problems whose iteration is many tiny kernels. The event pair contains the host-side work between them and the activity sum does not. This is the understood direction and is why short kernels score slightly low under the default methodology.
* **`rocprof` slower** (to -225% on `L1__018_fused_rope_with_qk_norm_and_kv_cache_update`) on some multi-dispatch iterations. Summing per-dispatch durations exceeds the wall clock whenever dispatches overlap, so the activity sum is not a wall-clock measurement for those. Stated as the hypothesis it is: it has not been confirmed against a dispatch timeline, and no number in this port depends on it, because `hip_events` is the default and every trace records its methodology.

| workloads > 20% apart | problem |
|---|---|
| 16 | L1__016_rope_inverse_frequency_computation |
| 16 | L1__021_vision_cu_seqlens_variable_length_attention |
| 16 | L1__034_flux_multi_axis_rope_frequency_computation |
| 16 | L1__051_attention_qkv_with_qk_norm_single_kernel_backward |
| 15 | L1__012_fused_cos_sin_embedding_generation |
| 15 | L1__044_moe_expert_computation |
| 15 | L1__058_moe_expert_token_radix_sort_with_prefix_sum |
| 14 | L1__014_rotary_embedding_with_attention_scaling_backward |
| 14 | L1__023_multimodal_rope_position_computation_with_grid_based_indexing |
| 14 | L1__024_vision_rotary_position_embedding_generation_backward |
| 14 | L1__042_moe_expert_load_balancing_and_token_capacity_backward |
| 13 | L1__011_rotary_position_embedding |

## What this licenses

`hip_events` stays the default and `Environment.methodology` is recorded on every trace. A trace taken under one and a trace taken under the other are not interchangeable — that is what the field is for. Mixing them silently is the failure this measurement exists to make impossible.

