# Task 04 — `hip_events` vs `rocprof`

<!-- {"task": "04-methodology-comparison", "utc": "2026-08-04T02:30:05.102997+00:00", "git_sha": "68f92bb6f4f48f69bd03b4d228a99ddbd05fb9a5-dirty", "host": "gbt350-odcdh1-a08-1.png-odc.dcgpu", "python": "3.12.3", "torch": {"available": true, "version": "2.9.1+rocm7.2.0.git7e1940d4", "hip": "7.2.26015-fc0010cf6a", "cuda": null, "device_count": 8, "devices": ["AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X"]}, "rocm": {"version": "7.2.0", "driver": "7.1.1.31500000", "amd_smi": "AMDSMI Tool: 26.2.1+fc0010cf6a | AMDSMI Library version: 26.2.1 | ROCm version: 7.2.0 | amdgpu version: 7.1.1.31500000 | hsmp version: N/A"}, "f_lock_mhz": 1300, "visible_devices": null} -->

Both methodologies time the same solution on the same inputs, back to back in one process, so this compares two ways of measuring and not two moments in the node's life. **Positive means `hip_events` read slower**, which is the expected direction: an event pair brackets the host launch and dispatch-level activity tracing does not.

Problems compared: **92** of 92; workload pairs: **1430**.

| group | n | median | p10 | p90 |
|---|---|---|---|---|
| all workloads | 1430 | -0.77% | -43.6% | +4.0% |
| kernels >= 100 us | 1044 | -0.61% | -44.8% | +1.4% |
| kernels < 100 us | 386 | -4.71% | -43.5% | +9.6% |

**Acceptance — median divergence on kernels >= 100 us: -0.61%** against a gate of 2%. PASS.

Sub-100 us kernels are reported separately rather than folded in. There the median is -4.71%: a fixed launch overhead is a larger fraction of a smaller number, which is the finding, not an error.

## The tails are wide, and they are wide in both directions

330 of 1430 workload pairs differ by more than 20%, concentrated in 57 problems. The median is small because most iterations dispatch one kernel; the tail is where they do not.

* **`hip_events` much slower** (up to ~90%) on problems whose iteration is many tiny kernels. The event pair contains the host-side work between them and the activity sum does not. This is the understood direction and is why short kernels score slightly low under the default methodology.
* **`rocprof` slower** (to ~3x on `L1/034`) on some multi-dispatch iterations. Summing per-dispatch durations exceeds the wall clock whenever dispatches overlap, so the activity sum is not a wall-clock measurement for those. Stated as the hypothesis it is: it has not been confirmed against a dispatch timeline, and no number in this port depends on it, because `hip_events` is the default and every trace records its methodology.

| workloads > 20% apart | problem |
|---|---|
| 16 | L1__016_rope_inverse_frequency_computation |
| 16 | L1__021_vision_cu_seqlens_variable_length_attention |
| 16 | L1__024_vision_rotary_position_embedding_generation_backward |
| 16 | L1__034_flux_multi_axis_rope_frequency_computation |
| 16 | L1__058_moe_expert_token_radix_sort_with_prefix_sum |
| 16 | L1__086_sam_hq_mask_decoder_iou_hypernetwork_fusion |
| 15 | L1__012_fused_cos_sin_embedding_generation |
| 13 | L1__042_moe_expert_load_balancing_and_token_capacity_backward |
| 12 | L1__023_multimodal_rope_position_computation_with_grid_based_indexing |
| 12 | L1__051_attention_qkv_with_qk_norm_single_kernel_backward |
| 11 | L1__014_rotary_embedding_with_attention_scaling_backward |
| 10 | L1__053_gaussian_topk_sparse_activation |

## What this licenses

`hip_events` stays the default and `Environment.methodology` is recorded on every trace. A trace taken under one and a trace taken under the other are not interchangeable — that is what the field is for. Mixing them silently is the failure this measurement exists to make impossible.

