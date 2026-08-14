# Delivery index — 40 problems (MI355X)

Selection: proportional by category, first N sorted, largest-remainder apportionment

| # | category | problem | reference | tolerance | bound | T_b | scoreable |
|---|---|---|---|---|---|---|---|
| 1 | L1 | `001_attention_softmax_dropout_value_matmul_backward` | yes | yes | - | - | - |
| 2 | L1 | `002_vae_conv3x3_groupnorm_silu_residual_fused` | yes | yes | - | - | - |
| 3 | L1 | `003_lm_head_projection_with_logit_slicing` | yes | yes | - | - | - |
| 4 | L1 | `004_attention_output_projection_with_reshape_backward` | yes | yes | - | - | - |
| 5 | L1 | `005_conv_gated_projection_with_causal_conv` | yes | yes | - | - | - |
| 6 | L1 | `006_hyena_depthwise_conv1d_split_gate` | yes | yes | - | - | - |
| 7 | L1 | `007_hyena_fft_size_padding_rfft` | yes | yes | - | - | - |
| 8 | L1 | `008_expert_output_weighted_index_add_accumulation` | yes | yes | - | - | - |
| 9 | L1 | `009_expert_token_scatter_with_weighted_forward_backward` | yes | yes | - | - | - |
| 10 | L1 | `010_attention_value_projection_with_transpose` | yes | yes | - | - | - |
| 11 | L1 | `011_rotary_position_embedding` | yes | yes | - | - | - |
| 12 | L1 | `012_fused_cos_sin_embedding_generation` | yes | yes | - | - | - |
| 13 | L1 | `013_fused_residual_rms_norm_backward` | yes | yes | - | - | - |
| 14 | L1 | `014_rotary_embedding_with_attention_scaling_backward` | yes | yes | - | - | - |
| 15 | L1 | `015_grouped_query_attention_with_rope_and_qk_norm` | yes | yes | - | - | - |
| 16 | L1 | `016_rope_inverse_frequency_computation` | yes | yes | - | - | - |
| 17 | L2 | `001_fused_vision_multihead_attention_with_norms_backward` | yes | yes | - | - | - |
| 18 | L2 | `002_decoder_layer_full_block` | yes | yes | - | - | - |
| 19 | L2 | `003_grouped_query_attention_with_rope_backward` | yes | yes | - | - | - |
| 20 | L2 | `004_fused_residual_rms_mlp` | yes | yes | - | - | - |
| 21 | L2 | `005_swiglu_mlp_backward` | yes | yes | - | - | - |
| 22 | L2 | `006_multimodal_rope_position_calculation` | yes | yes | - | - | - |
| 23 | L2 | `007_multimodal_rotary_embedding_attention` | yes | yes | - | - | - |
| 24 | L2 | `008_moe_sparse_routing_and_dispatch` | yes | yes | - | - | - |
| 25 | L2 | `009_decoder_layer_with_residual_connections` | yes | yes | - | - | - |
| 26 | L2 | `010_moe_expert_computation_with_weighted_accumulation` | yes | yes | - | - | - |
| 27 | L2 | `011_moe_sparse_routing_and_dispatch_backward` | yes | yes | - | - | - |
| 28 | L2 | `012_moe_expert_batched_execution_with_capacity_factor` | yes | yes | - | - | - |
| 29 | L2 | `013_expert_weighted_aggregation_with_shared_expert` | yes | yes | - | - | - |
| 30 | L2 | `014_audio_encoder_varlen_attention_with_chunking_backward` | yes | yes | - | - | - |
| 31 | Quant | `001_fp8_attention_output_projection` | yes | yes | - | - | - |
| 32 | Quant | `002_fp8_attention_qkv_projection` | yes | yes | - | - | - |
| 33 | Quant | `003_fp8_mlp_gate_up_projection` | yes | yes | - | - | - |
| 34 | Quant | `004_fp8_moe_expert_linear` | yes | yes | - | - | - |
| 35 | Quant | `005_fp8_moe_router_projection` | yes | yes | - | - | - |
| 36 | Quant | `006_fp8_vision_attention_output_projection` | yes | yes | - | - | - |
| 37 | FlashInfer-Bench | `001_fused_add_rmsnorm_h2048` | yes | yes | - | - | - |
| 38 | FlashInfer-Bench | `002_fused_add_rmsnorm_h4096` | yes | yes | - | - | - |
| 39 | FlashInfer-Bench | `003_fused_add_rmsnorm_h7168` | yes | yes | - | - | - |
| 40 | FlashInfer-Bench | `004_gemm_n128_k2048` | yes | yes | - | - | - |

Counts: reference 40/40, tolerance 40/40, bound 0/40, t_b 0/40, scoreable 0/40
