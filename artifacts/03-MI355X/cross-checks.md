# Task 03 — T_SOL cross-checks

<!-- {"task": "03-cross-checks", "utc": "2026-08-15T03:58:35.940899+00:00", "git_sha": "e974e70565a5e6a94874d11448d3bfaf6f889ee0-dirty", "host": "mia1-p02-g46", "python": "3.12.3", "torch": {"available": true, "version": "2.9.1+rocm7.2.0.git7e1940d4", "hip": "7.2.26015-fc0010cf6a", "cuda": null, "device_count": 8, "devices": ["AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X"]}, "rocm": {"version": "7.2.0", "driver": "6.16.6", "amd_smi": "AMDSMI Tool: 26.2.1+fc0010cf6a | AMDSMI Library version: 26.2.1 | ROCm version: 7.2.0 | amdgpu version: 6.16.6 | hsmp version: N/A"}, "f_lock_mhz": null, "visible_devices": null} -->

Upstream's B200 SOL times are not used as a comparison anywhere in this document. The shipped dataset carries no per-workload SOL figures, so there is nothing to compare against that was not invented here — and an invented comparison would be worse than none. The three checks below are internal to this platform and are stronger for it.

## A — SOLAR's memory term vs the problem's own declared traffic

Every definition states the shape and dtype of each input and output. Their sum is what any correct kernel must move at least once. A memory term below it is not a bound.

* checked: **2998** workloads
* below declared minimum: **1021**
* not checkable (unresolved symbol or dtype): 0

The shortfall is concentrated: **66 problems**, not a scatter across the set. Two mechanisms produce it, and only one of them is benign.

*Benign* — a preallocated cache or table that the kernel indexes rather than streams. `L1/018` declares a KV cache of 131072 positions and touches one sequence's worth of it, so the declared total is not a floor for that kernel and the ratio near zero is expected.

*Not benign* — a graph SOLAR traced incompletely, which shows up as a missing weight matrix. Where the largest declared tensor is a weight and the ratio is ~0.001, SOLAR did not see the matmul that consumes it.

**Direction of the error.** A T_SOL below the true bound makes `(T_b − T_SOL)` too large, so scores computed against it are *understated*, not inflated. That is the safe direction — no kernel is flattered by it — but it is still wrong, and these problems carry the ratio into the manifest so a consumer can see which bounds are loose.

| worst ratio | workloads | problem | declared bytes | SOLAR bytes |
|---|---|---|---|---|
| 0.0000 | 13 | L1__018_fused_rope_with_qk_norm_and_kv_cache_update | 8,590,156,584 | 213,640 |
| 0.0003 | 16 | L2__031_flux_timestep_guidance_projection_embedding | 56,676,868 | 16,900 |
| 0.0003 | 16 | L1__086_sam_hq_mask_decoder_iou_hypernetwork_fusion | 7,510,048 | 2,576 |
| 0.0021 | 6 | Quant__023_fp8_mamba2_ssm_discretization | 34,431,599,360 | 72,909,312 |
| 0.0078 | 16 | L1__087_embedding_with_initial_layernorm_backward | 543,181,824 | 4,211,712 |
| 0.0105 | 16 | L1__037_flux_feedforward_gelu_approximate | 305,270,784 | 3,219,456 |
| 0.0118 | 16 | L1__029_mamba_conv1d_with_gating | 547,586,304 | 6,455,808 |
| 0.0135 | 16 | L2__032_dual_stream_attention_with_conditional_cross_attention | 485,699,584 | 6,561,792 |
| 0.0205 | 15 | L1__020_vision_patch_merger_spatial_shuffle_mlp | 122,053,680 | 2,496,514 |
| 0.0208 | 16 | L1__036_flux_output_norm_projection_chain | 77,926,656 | 1,617,920 |
| 0.0394 | 16 | L1__075_grouped_query_self_attention_with_rope | 1,117,814,912 | 44,056,704 |
| 0.0502 | 16 | L1__080_adaptive_layernorm_continuous_with_modulation | 49,836,544 | 2,500,096 |
| 0.0504 | 16 | L1__021_vision_cu_seqlens_variable_length_attention | 27,627,536 | 1,392,644 |
| 0.0518 | 16 | L1__057_mtp_shifted_embedding_with_dual_rms_norm_fusion | 1,356,874,752 | 70,271,232 |
| 0.0593 | 16 | L1__005_conv_gated_projection_with_causal_conv | 35,688,448 | 2,117,632 |
| 0.0725 | 16 | L1__064_latent_kv_expansion_with_split | 578,814,976 | 41,944,064 |
| 0.1001 | 16 | L2__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block | 3,542,272 | 354,592 |
| 0.1111 | 16 | L1__027_video_spatial_attention_with_rope_3d | 18,891,008 | 2,098,768 |
| 0.1401 | 16 | L2__006_multimodal_rope_position_calculation | 656,096 | 91,888 |
| 0.1805 | 16 | Quant__026_nvfp4_mamba2_out_projection | 73,835,520 | 13,324,800 |
| 0.1998 | 16 | L1__008_expert_output_weighted_index_add_accumulation | 8,057,024 | 1,609,728 |
| 0.3229 | 15 | L2__055_audio_encoder_conv_positional_layer_stack | 851,912,960 | 275,114,240 |
| 0.3514 | 16 | L1__043_mla_fused_qkv_rope_split | 635,309,056 | 223,218,688 |
| 0.3916 | 16 | L1__073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection | 861,696 | 337,408 |
| 0.4286 | 16 | L1__013_fused_residual_rms_norm_backward | 18,792,599,552 | 8,054,122,496 |
| 0.4321 | 16 | L2__067_patch_embed_to_joint_attention_input | 193,026,564 | 83,401,732 |
| 0.4486 | 16 | L2__034_vision_language_cross_attention_fusion | 91,315,392 | 40,962,064 |
| 0.4669 | 16 | L1__092_gqa_attention_with_qk_norm | 220,295,680 | 102,859,264 |
| 0.4923 | 16 | L1__042_moe_expert_load_balancing_and_token_capacity_backward | 2,181,040,133 | 1,073,742,848 |
| 0.5000 | 16 | L2__075_sam_hq_mask_decoder_two_way_transformer | 12,640,256 | 6,320,128 |
| 0.5046 | 14 | L2__038_audio_relative_position_attention | 455,455,552 | 229,832,448 |
| 0.5202 | 16 | L2__070_basic_transformer_block | 246,000,640 | 127,979,520 |
| 0.5294 | 16 | L2__030_flux_concatenated_sequence_processing_with_split | 641,728,512 | 339,738,624 |
| 0.5407 | 16 | L1__052_altup_hidden_state_collapse_with_magnitude_normalization | 463,208,448 | 250,478,592 |
| 0.5749 | 16 | L1__081_joint_attention_context_projection | 315,819,520 | 181,573,120 |
| 0.6002 | 16 | L2__072_region_aware_self_attention_with_edit_bias_backward | 5,772,191,232 | 3,464,355,392 |
| 0.6050 | 14 | L1__045_fused_linear_gelu_grn_linear | 1,333,760 | 806,912 |
| 0.6243 | 16 | L1__014_rotary_embedding_with_attention_scaling_backward | 131,840 | 82,304 |
| 0.6667 | 16 | L1__085_geglu_activation | 7,864,320 | 5,242,880 |
| 0.6667 | 16 | L1__090_batched_2d_rope_position_encoding_backward | 786,432 | 524,288 |
| 0.6667 | 16 | L1__006_hyena_depthwise_conv1d_split_gate | 100,675,584 | 67,118,080 |
| 0.6677 | 16 | Quant__003_fp8_mlp_gate_up_projection | 497,770,880 | 332,370,016 |
| 0.6973 | 16 | Quant__022_fp8_mamba2_out_projection | 69,803,748 | 48,674,745 |
| 0.7525 | 16 | L2__057_residual_coupling_flow_block | 813,309,440 | 611,982,848 |
| 0.8033 | 16 | Quant__015_fp8_mla_attention_output_projection | 629,174,272 | 505,420,800 |
| 0.8076 | 16 | L2__078_fused_final_layer_upsample_with_adaptive_norm | 523,496,400 | 422,779,856 |
| 0.8167 | 16 | L1__051_attention_qkv_with_qk_norm_single_kernel_backward | 97,177,664 | 79,362,592 |
| 0.8174 | 16 | L2__015_audio_sinusoidal_position_embedding_with_conv_projection | 16,466,944 | 13,460,480 |
| 0.8301 | 16 | L2__066_resnet_block_with_time_embedding | 9,680,640 | 8,035,840 |
| 0.8316 | 16 | L2__044_mamba_discretization_and_segsum | 505,937,952 | 420,741,152 |
| 0.8578 | 16 | L2__041_kv_shared_attention_with_dual_rope | 177,275,264 | 152,060,576 |
| 0.8587 | 16 | L1__001_attention_softmax_dropout_value_matmul_backward | 9,495,904,256 | 8,153,726,976 |
| 0.8599 | 11 | L2__050_vae_decoder_mid_block_attention_resnet | 45,131,776 | 38,807,552 |
| 0.8889 | 16 | L1__066_masked_softmax_with_attention_dropout_backward | 28,311,552 | 25,165,824 |
| 0.9136 | 14 | L2__036_convnextv2_layer_with_nhwc_persistence_backward | 1,183,788,800 | 1,081,454,720 |
| 0.9235 | 16 | L2__042_ffn_gelu_projection_fused_backward | 3,506,978,816 | 3,238,543,360 |
| 0.9241 | 5 | L1__079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward | 350,407,680 | 323,814,912 |
| 0.9292 | 15 | L2__019_decoder_layer_fused_attention_mlp | 936,294,400 | 870,019,072 |
| 0.9454 | 16 | Quant__005_fp8_moe_router_projection | 31,785,072 | 30,048,284 |
| 0.9534 | 16 | L2__011_moe_sparse_routing_and_dispatch_backward | 744,751,104 | 710,017,024 |
| 0.9609 | 16 | L1__009_expert_token_scatter_with_weighted_forward_backward | 805,317,120 | 773,850,112 |
| 0.9917 | 16 | L1__024_vision_rotary_position_embedding_generation_backward | 10,813,584 | 10,723,724 |
| 0.9961 | 15 | FlashInfer-Bench__016_gqa_ragged_prefill_causal_h32_kv4_d128 | 18,576 | 18,504 |
| 0.9965 | 21 | FlashInfer-Bench__017_gqa_ragged_prefill_causal_h32_kv8_d128 | 20,624 | 20,552 |
| 0.9985 | 15 | Quant__013_fp8_mla_kv_compression_projection | 1,353,450,496 | 1,351,353,344 |
| 0.9987 | 16 | FlashInfer-Bench__018_mla_paged_decode_h16_ckv512_kpe64_ps1 | 1,142,631,848 | 1,141,186,140 |

## B — rates implied by T_SOL

Arch config: DRAM 8.00 TB/s at 2.4 GHz.

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

2546/2694 workloads satisfy T_SOL <= T_b — **148 VIOLATIONS**, each one a config error

| problem | workload | T_SOL ms | T_b ms | variant |
|---|---|---|---|---|
| L1__006_hyena_depthwise_conv1d_split_gate | `ec71ab53` | 0.01542 | 0.0132 | v2_compile |
| L1__006_hyena_depthwise_conv1d_split_gate | `b9d99d9d` | 0.246 | 0.05304 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `71d9a820` | 0.2496 | 0.08162 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `62d61aa9` | 0.03078 | 0.01688 | v2_compile |
| L1__006_hyena_depthwise_conv1d_split_gate | `9cb591a3` | 0.49176 | 0.093221 | v2_compile |
| L1__006_hyena_depthwise_conv1d_split_gate | `efc0661b` | 0.24768 | 0.0634605 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `8ce027b6` | 0.49344 | 0.0992805 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `63c2e8ad` | 0.0615 | 0.0233605 | v2_compile |
| L1__006_hyena_depthwise_conv1d_split_gate | `384d57b8` | 0.49248 | 0.091441 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `66b15fa6` | 0.0177 | 0.01552 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `d698ad85` | 0.1248 | 0.0474805 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `da31d8e3` | 0.12336 | 0.03368 | v2_compile |
| L1__006_hyena_depthwise_conv1d_split_gate | `950dc0ec` | 0.12384 | 0.03812 | v3_compile_max_autotune |
| L1__006_hyena_depthwise_conv1d_split_gate | `7de2c122` | 0.06192 | 0.02566 | v2_compile |
| L1__029_mamba_conv1d_with_gating | `71b7f4bc` | 0.182044 | 0.168441 | v1_eager |
| L1__029_mamba_conv1d_with_gating | `b2e443d9` | 3.07769 | 1.39301 | v4_contiguous |
| L1__029_mamba_conv1d_with_gating | `4a61237f` | 2.91271 | 1.23225 | v4_contiguous |
| L1__029_mamba_conv1d_with_gating | `5aafa38b` | 11.6508 | 4.15497 | v3_compile_max_autotune |
| L1__029_mamba_conv1d_with_gating | `1b0a2e46` | 11.6508 | 3.85063 | v3_compile_max_autotune |
| L1__029_mamba_conv1d_with_gating | `63f6f33c` | 11.6508 | 4.44283 | v3_compile_max_autotune |
| L1__035_flux_ada_layer_norm_zero_modulation_extraction | `81f42cda` | 0.49152 | 0.417123 | v4_contiguous |
| L1__035_flux_ada_layer_norm_zero_modulation_extraction | `2879d7a9` | 0.20256 | 0.201601 | v2_compile |
| L1__035_flux_ada_layer_norm_zero_modulation_extraction | `8e261dd1` | 0.43104 | 0.389263 | v2_compile |
| L1__035_flux_ada_layer_norm_zero_modulation_extraction | `51fef589` | 0.35808 | 0.321462 | v4_contiguous |
| L1__035_flux_ada_layer_norm_zero_modulation_extraction | `2f570f4d` | 0.18432 | 0.164361 | v2_compile |

## D-published — the bound a score is actually computed against

Section D above audits the SOLAR tier alone. The manifest publishes
max(SOLAR, declared-traffic) and rejects a tier that exceeds the
measured T_b, so the tier count overstates the shipped damage.

3660/3701 PUBLISHED workloads satisfy T_SOL <= T_b — **41 VIOLATIONS across 4 problems**. Scores on those problems are not results.

| problem | workload | T_SOL ms | T_b ms | bound tier |
|---|---|---|---|---|
| L1__006_hyena_depthwise_conv1d_split_gate | `384d57b8` | 0.371216 | 0.091441 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `9cb591a3` | 0.372389 | 0.093221 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `8ce027b6` | 0.37194 | 0.0992805 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `b9d99d9d` | 0.185505 | 0.05304 | declared_traffic |
| L1__057_mtp_shifted_embedding_with_dual_rms_norm_fusion | `9fec4efe` | 0.169873 | 0.0495 | solar_fused |
| L1__029_mamba_conv1d_with_gating | `25cc310d` | 11.6768 | 3.83507 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `1b0a2e46` | 11.6833 | 3.85063 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `4d8dcc2c` | 2.79769 | 0.936326 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `bc1dc4bf` | 11.6508 | 3.90895 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `efc0661b` | 0.186928 | 0.0634605 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `5aafa38b` | 11.5355 | 4.15497 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `da31d8e3` | 0.0928683 | 0.03368 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `ee62552e` | 1.32731 | 0.481423 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `63f6f33c` | 11.6638 | 4.44283 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `2e529adb` | 1.94675 | 0.747465 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `950dc0ec` | 0.0935426 | 0.03812 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `bfb88d94` | 1.28502 | 0.542084 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `71d9a820` | 0.188141 | 0.08162 | declared_traffic |
| L2__035_convnextv2_block_with_grn | `65bf886b` | 0.670879 | 0.304603 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `088e7f41` | 2.62403 | 1.21407 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `4a61237f` | 2.5488 | 1.23225 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `e36c7969` | 0.730716 | 0.365762 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `63c2e8ad` | 0.0463568 | 0.0233605 | declared_traffic |
| L1__006_hyena_depthwise_conv1d_split_gate | `d698ad85` | 0.0940704 | 0.0474805 | declared_traffic |
| L1__029_mamba_conv1d_with_gating | `b2e443d9` | 2.68143 | 1.39301 | declared_traffic |

