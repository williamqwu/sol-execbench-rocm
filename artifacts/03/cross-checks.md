# Task 03 — T_SOL cross-checks

<!-- {"task": "03-cross-checks", "utc": "2026-08-04T01:26:45.505045+00:00", "git_sha": "68f92bb6f4f48f69bd03b4d228a99ddbd05fb9a5-dirty", "host": "gbt350-odcdh1-a08-1.png-odc.dcgpu", "python": "3.12.3", "torch": {"available": true, "version": "2.9.1+rocm7.2.0.git7e1940d4", "hip": "7.2.26015-fc0010cf6a", "cuda": null, "device_count": 8, "devices": ["AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X"]}, "rocm": {"version": "7.2.0", "driver": "7.1.1.31500000", "amd_smi": "AMDSMI Tool: 26.2.1+fc0010cf6a | AMDSMI Library version: 26.2.1 | ROCm version: 7.2.0 | amdgpu version: 7.1.1.31500000 | hsmp version: N/A"}, "f_lock_mhz": 1300, "visible_devices": null} -->

Upstream's B200 SOL times are not used as a comparison anywhere in this document. The shipped dataset carries no per-workload SOL figures, so there is nothing to compare against that was not invented here — and an invented comparison would be worse than none. The three checks below are internal to this platform and are stronger for it.

## A — SOLAR's memory term vs the problem's own declared traffic

Every definition states the shape and dtype of each input and output. Their sum is what any correct kernel must move at least once. A memory term below it is not a bound.

* checked: **1415** workloads
* below declared minimum: **164**
* not checkable (unresolved symbol or dtype): 1583

The shortfall is concentrated: **13 problems**, not a scatter across the set. Two mechanisms produce it, and only one of them is benign.

*Benign* — a preallocated cache or table that the kernel indexes rather than streams. `L1/018` declares a KV cache of 131072 positions and touches one sequence's worth of it, so the declared total is not a floor for that kernel and the ratio near zero is expected.

*Not benign* — a graph SOLAR traced incompletely, which shows up as a missing weight matrix. Where the largest declared tensor is a weight and the ratio is ~0.001, SOLAR did not see the matmul that consumes it.

**Direction of the error.** A T_SOL below the true bound makes `(T_b − T_SOL)` too large, so scores computed against it are *understated*, not inflated. That is the safe direction — no kernel is flattered by it — but it is still wrong, and these problems carry the ratio into the manifest so a consumer can see which bounds are loose.

| worst ratio | workloads | problem | declared bytes | SOLAR bytes |
|---|---|---|---|---|
| 0.0000 | 13 | L1__018_fused_rope_with_qk_norm_and_kv_cache_update | 8,590,156,584 | 427,280 |
| 0.0003 | 16 | L2__031_flux_timestep_guidance_projection_embedding | 56,676,868 | 16,900 |
| 0.0003 | 16 | L1__086_sam_hq_mask_decoder_iou_hypernetwork_fusion | 7,510,048 | 2,576 |
| 0.0042 | 6 | Quant__023_fp8_mamba2_ssm_discretization | 34,431,599,360 | 145,818,624 |
| 0.0105 | 16 | L1__037_flux_feedforward_gelu_approximate | 305,270,784 | 3,219,456 |
| 0.0409 | 13 | L1__020_vision_patch_merger_spatial_shuffle_mlp | 122,053,680 | 4,993,028 |
| 0.5000 | 16 | L2__075_sam_hq_mask_decoder_two_way_transformer | 12,640,256 | 6,320,128 |
| 0.5046 | 14 | L2__038_audio_relative_position_attention | 455,455,552 | 229,832,448 |
| 0.5294 | 16 | L2__030_flux_concatenated_sequence_processing_with_split | 641,728,512 | 339,738,624 |
| 0.7831 | 1 | L1__073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection | 861,696 | 674,816 |
| 0.8316 | 16 | L2__044_mamba_discretization_and_segsum | 505,937,952 | 420,741,152 |
| 0.9241 | 5 | L1__079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward | 350,407,680 | 323,814,912 |
| 0.9609 | 16 | L1__009_expert_token_scatter_with_weighted_forward_backward | 805,317,120 | 773,850,112 |

## B — rates implied by T_SOL

Arch config: DRAM 8.00 TB/s at 1.3 GHz.

* checked: **2998** workloads
* implied bandwidth above DRAM peak: **0**
* implied FLOPS above the precision's peak: **0**

## C — hand-derived MAC counts

Eleven single matmuls and seven pure memory kernels, counted by hand from the reference source. A pure memory kernel's expected count is zero, and zero is a real prediction: MACs reported for one would mean SOLAR found arithmetic that is not in the kernel.

| problem | hand-derived MACs | SOLAR MACs | verdict |
|---|---|---|---|
| FlashInfer-Bench__001_fused_add_rmsnorm_h2048 | 0 | 0 | exact |
| FlashInfer-Bench__002_fused_add_rmsnorm_h4096 | 0 | 0 | exact |
| FlashInfer-Bench__003_fused_add_rmsnorm_h7168 | 0 | 0 | exact |
| FlashInfer-Bench__004_gemm_n128_k2048 | 2,097,152 | 2,097,152 | exact |
| FlashInfer-Bench__005_gemm_n256_k7168 | 100,925,440 | 100,925,440 | exact |
| FlashInfer-Bench__006_gemm_n2048_k4096 | 16,777,216 | 16,777,216 | exact |
| FlashInfer-Bench__007_gemm_n4096_k4096 | 117,440,512 | 117,440,512 | exact |
| FlashInfer-Bench__008_gemm_n4096_k14336 | 5,637,144,576 | 5,637,144,576 | exact |
| FlashInfer-Bench__009_gemm_n5120_k2048 | 20,971,520 | 20,971,520 | exact |
| FlashInfer-Bench__010_gemm_n6144_k4096 | 6,442,450,944 | 6,442,450,944 | exact |
| FlashInfer-Bench__011_gemm_n28672_k4096 | 11,274,289,152 | 11,274,289,152 | exact |
| L1__003_lm_head_projection_with_logit_slicing | 858,993,459,200 | 858,993,459,200 | exact |
| L1__025_video_latent_gelu_activation | 0 | 0 | exact |
| L1__046_attention_softmax_with_softcapping_and_dropout | 0 | 0 | exact |
| L1__069_rms_norm | 0 | 0 | exact |
| L1__077_whisper_decoder_output_projection | 33,990,901,760 | 33,990,901,760 | exact |
| L1__085_geglu_activation | 0 | 0 | exact |
| L2__030_flux_concatenated_sequence_processing_with_split | 3,623,878,656 | 3,623,878,656 | exact |

MISMATCHes: **0**

## D — T_SOL <= best measured time

PENDING — needs task 06 (`--t-b artifacts/06/authoritative`)

