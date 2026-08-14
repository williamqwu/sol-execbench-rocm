# Delivery index — 40 problems (MI355X)

Selection: proportional by category, first N sorted, largest-remainder apportionment

| # | category | problem | reference | tolerance | bound | T_b | scoreable |
|---|---|---|---|---|---|---|---|
| 1 | L1 | `001_attention_softmax_dropout_value_matmul_backward` | yes | yes | yes | yes | yes |
| 2 | L1 | `004_attention_output_projection_with_reshape_backward` | yes | yes | yes | yes | yes |
| 3 | L1 | `006_hyena_depthwise_conv1d_split_gate` | yes | yes | yes | yes | yes |
| 4 | L1 | `008_expert_output_weighted_index_add_accumulation` | yes | yes | yes | yes | yes |
| 5 | L1 | `009_expert_token_scatter_with_weighted_forward_backward` | yes | yes | yes | yes | yes |
| 6 | L1 | `011_rotary_position_embedding` | yes | yes | yes | yes | yes |
| 7 | L1 | `012_fused_cos_sin_embedding_generation` | yes | yes | yes | yes | yes |
| 8 | L1 | `016_rope_inverse_frequency_computation` | yes | yes | yes | yes | yes |
| 9 | L1 | `017_moe_expert_swiglu_with_down_projection_backward` | yes | yes | yes | yes | yes |
| 10 | L1 | `018_fused_rope_with_qk_norm_and_kv_cache_update` | yes | yes | yes | yes | yes |
| 11 | L1 | `019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward` | yes | yes | yes | yes | yes |
| 12 | L1 | `021_vision_cu_seqlens_variable_length_attention` | yes | yes | yes | yes | yes |
| 13 | L1 | `022_vision_mlp_gelu_backward` | yes | yes | yes | yes | yes |
| 14 | L1 | `023_multimodal_rope_position_computation_with_grid_based_indexing` | yes | yes | yes | yes | yes |
| 15 | L1 | `024_vision_rotary_position_embedding_generation_backward` | yes | yes | yes | yes | yes |
| 16 | L1 | `028_hybrid_attention_mask_preparation` | yes | yes | yes | yes | yes |
| 17 | L1 | `031_repeat_kv_attention_matmul` | yes | yes | yes | yes | yes |
| 18 | L2 | `002_decoder_layer_full_block` | yes | yes | yes | yes | yes |
| 19 | L2 | `006_multimodal_rope_position_calculation` | yes | yes | yes | yes | yes |
| 20 | L2 | `008_moe_sparse_routing_and_dispatch` | yes | yes | yes | yes | yes |
| 21 | L2 | `009_decoder_layer_with_residual_connections` | yes | yes | yes | yes | yes |
| 22 | L2 | `010_moe_expert_computation_with_weighted_accumulation` | yes | yes | yes | yes | yes |
| 23 | L2 | `011_moe_sparse_routing_and_dispatch_backward` | yes | yes | yes | yes | yes |
| 24 | L2 | `013_expert_weighted_aggregation_with_shared_expert` | yes | yes | yes | yes | yes |
| 25 | L2 | `014_audio_encoder_varlen_attention_with_chunking_backward` | yes | yes | yes | yes | yes |
| 26 | L2 | `016_moe_expert_mlp_with_load_balancing` | yes | yes | yes | yes | yes |
| 27 | L2 | `018_cu_seqlens_variable_length_vision_attention` | yes | yes | yes | yes | yes |
| 28 | L2 | `019_decoder_layer_fused_attention_mlp` | yes | yes | yes | yes | yes |
| 29 | L2 | `021_cross_attention_text_video_conditioning_backward` | yes | yes | yes | yes | yes |
| 30 | L2 | `022_video_latent_denoising_unet_block` | yes | yes | yes | yes | yes |
| 31 | L2 | `024_moe_expert_parallel_execution` | yes | yes | yes | yes | yes |
| 32 | L2 | `025_moe_expert_parallel_execution_backward` | yes | yes | yes | yes | yes |
| 33 | Quant | `003_fp8_mlp_gate_up_projection` | yes | yes | yes | yes | yes |
| 34 | Quant | `004_fp8_moe_expert_linear` | yes | yes | yes | yes | yes |
| 35 | Quant | `005_fp8_moe_router_projection` | yes | yes | yes | yes | yes |
| 36 | Quant | `006_fp8_vision_attention_output_projection` | yes | yes | yes | yes | yes |
| 37 | FlashInfer-Bench | `001_fused_add_rmsnorm_h2048` | yes | yes | yes | yes | yes |
| 38 | FlashInfer-Bench | `002_fused_add_rmsnorm_h4096` | yes | yes | yes | yes | yes |
| 39 | FlashInfer-Bench | `012_gqa_paged_decode_h32_kv4_d128_ps1` | yes | yes | yes | yes | yes |
| 40 | FlashInfer-Bench | `013_gqa_paged_decode_h32_kv8_d128_ps1` | yes | yes | yes | yes | yes |

Counts: reference 40/40, tolerance 40/40, bound 40/40, t_b 40/40, t_b_authoritative 40/40, bound_reclockable 40/40, scoreable 40/40
