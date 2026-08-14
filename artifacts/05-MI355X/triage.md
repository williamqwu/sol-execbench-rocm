# Task 05 — tolerance triage

<!-- {"task": "05-tolerance-triage", "utc": "2026-08-14T17:26:39.739395+00:00", "git_sha": null, "host": "mia1-p02-g46", "python": "3.12.3", "torch": {"available": true, "version": "2.9.1+rocm7.2.0.git7e1940d4", "hip": "7.2.26015-fc0010cf6a", "cuda": null, "device_count": 8, "devices": ["AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X", "AMD Instinct MI355X"]}, "rocm": {"version": "7.2.0", "driver": "6.16.6", "amd_smi": "AMDSMI Tool: 26.2.1+fc0010cf6a | AMDSMI Library version: 26.2.1 | ROCm version: 7.2.0 | amdgpu version: 6.16.6 | hsmp version: N/A"}, "f_lock_mhz": null, "visible_devices": null} -->

AMD-derived tolerances written for **3717 of 3957** workload instances.

Every number below was derived from reference-vs-reference variance on MI350X and nothing else. No B200 value was used as a source; upstream's values appear only as a comparison column.

## Structurally nondeterministic references

These disagree with themselves run to run on identical inputs. That is a property of the kernels (atomics-ordered accumulation, library algorithm selection), not a bug, and it is why their tolerances are wider than their neighbours'. Each one is listed rather than absorbed silently, because a wide tolerance is exactly what would let a wrong kernel through.

| problem | workload | run-to-run max_abs (float outputs) | run-to-run max_abs (int/bool outputs) | derived atol | derived rtol |
|---|---|---|---|---|---|
| L1__008_expert_output_weighted_index_add_accumulation | `01dfd338` | 0.1875 | 0 | 0.234375 | 0.394737 |
| L1__008_expert_output_weighted_index_add_accumulation | `0bda2a65` | 0.1875 | 0 | 0.234375 | 0.5 |
| L1__008_expert_output_weighted_index_add_accumulation | `0f250bfd` | 0.25 | 0 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `172afa25` | 0.1875 | 0 | 0.234375 | 0.4375 |
| L1__008_expert_output_weighted_index_add_accumulation | `28f8046d` | 0.1875 | 0 | 0.234375 | 0.416667 |
| L1__008_expert_output_weighted_index_add_accumulation | `2c1d8396` | 0.1875 | 0 | 0.234375 | 0.416667 |
| L1__008_expert_output_weighted_index_add_accumulation | `5a674481` | 0.1875 | 0 | 0.234375 | 0.416667 |
| L1__008_expert_output_weighted_index_add_accumulation | `611e32dd` | 0.25 | 0 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `8b677344` | 0.1875 | 0 | 0.234375 | 0.5 |
| L1__008_expert_output_weighted_index_add_accumulation | `8c5c94b5` | 0.25 | 0 | 0.3125 | 0.5 |
| L1__008_expert_output_weighted_index_add_accumulation | `a57cc129` | 0.25 | 0 | 0.3125 | 0.365854 |
| L1__008_expert_output_weighted_index_add_accumulation | `ada14c03` | 0.25 | 0 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `b1b7efca` | 0.25 | 0 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `b61ee9a7` | 0.25 | 0 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `c4a319ff` | 0.25 | 0 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `edc192ac` | 0.21875 | 0 | 0.273438 | 0.428571 |
| L1__024_vision_rotary_position_embedding_generation_backward | `0cda4941` | 0.0119629 | 0 | 0.0149536 | 0.000142202 |
| L1__024_vision_rotary_position_embedding_generation_backward | `199e4d8e` | 6.10352e-05 | 0 | 7.62939e-05 | 1.00241e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `229e6489` | 0.00195312 | 0 | 0.00244141 | 3.76474e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `472b424d` | 0.000152588 | 0 | 0.000190735 | 2.0571e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `61d8b841` | 0.00878906 | 0 | 0.0109863 | 2.29048e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `7579ff98` | 0.000610352 | 0 | 0.000762939 | 1.42126e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `85f40a68` | 0.000732422 | 0 | 0.000915527 | 5.95012e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `96bc140b` | 0.000610352 | 0 | 0.000762939 | 1.15531e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `a1a49d6d` | 0.00585938 | 0 | 0.00732422 | 6.04948e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `a292f1ef` | 6.86646e-05 | 0 | 8.58307e-05 | 1.00151e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `bca3cee8` | 0.000366211 | 0 | 0.000457764 | 2.5853e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `dbd8f288` | 0.000732422 | 0 | 0.000915527 | 0.000139329 |
| L1__024_vision_rotary_position_embedding_generation_backward | `e12534d6` | 0.000366211 | 0 | 0.000457764 | 2.82584e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `e2e6995a` | 0.000488281 | 0 | 0.000610352 | 0.000406792 |
| L1__024_vision_rotary_position_embedding_generation_backward | `e56dd6a6` | 0.000732422 | 0 | 0.000915527 | 6.96174e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `fc1c68f0` | 0.00012207 | 0 | 0.000152588 | 1.34809e-05 |
| L1__087_embedding_with_initial_layernorm_backward | `5d6fcf80` | 0.00390625 | 0 | 0.00960052 | 0.0078125 |
| L1__087_embedding_with_initial_layernorm_backward | `a6d899cd` | 0.00390625 | 0 | 0.00960052 | 0.00892857 |
| L1__087_embedding_with_initial_layernorm_backward | `a70755cd` | 0.000244141 | 0 | 0.00677592 | 0.0078125 |
| L1__087_embedding_with_initial_layernorm_backward | `b54a2121` | 1.52588e-05 | 0 | 0.00583629 | 0.0078125 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `106f0092` | 0.0195312 | 0 | 0.0244141 | 0.4 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `16755515` | 0.0195312 | 0 | 0.0244141 | 0.3 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `18872872` | 0.0195312 | 0 | 0.0244141 | 0.277778 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `2e7d2e79` | 0.0195312 | 0 | 0.0244141 | 0.3 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `2fe16676` | 0.0234375 | 0 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `3744fdcd` | 0.0234375 | 0 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `598f9037` | 0.0195312 | 0 | 0.0244141 | 0.3125 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `604e6e46` | 0.0234375 | 0 | 0.0292969 | 0.25 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `61022909` | 0.0195312 | 0 | 0.0244141 | 0.3 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `9a4bee24` | 0.0234375 | 0 | 0.0292969 | 0.243902 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `ab49eedb` | 0.015625 | 0 | 0.0195312 | 0.3125 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `b983758c` | 0.0195312 | 0 | 0.0244141 | 0.31875 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `bf41eb28` | 0.0195312 | 0 | 0.0244141 | 0.3125 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `c2a09e88` | 0.015625 | 0 | 0.0195312 | 0.5 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `d36f3c8b` | 0.0195312 | 0 | 0.0244141 | 0.25 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `f850f2b7` | 0.0195312 | 0 | 0.0244141 | 0.2875 |
| L2__015_audio_sinusoidal_position_embedding_with_conv_projection | `d982c737` | 256 | 0 | 320 | 0.00968992 |
| L2__024_moe_expert_parallel_execution | `00c0833a` | 0.00390625 | 0 | 0.00488281 | 0.0078125 |
| L2__024_moe_expert_parallel_execution | `032fc60b` | 0.0078125 | 0 | 0.00976562 | 0.0094697 |
| L2__024_moe_expert_parallel_execution | `0887b5af` | 0.0078125 | 0 | 0.00976562 | 0.00811688 |
| L2__024_moe_expert_parallel_execution | `1367e254` | 0.00390625 | 0 | 0.00488281 | 0.00912409 |
| L2__024_moe_expert_parallel_execution | `154b4946` | 0.0078125 | 0 | 0.00976562 | 0.00961538 |
| L2__024_moe_expert_parallel_execution | `2b597deb` | 0.00195312 | 0 | 0.00348051 | 0.00954198 |
| L2__024_moe_expert_parallel_execution | `2ec82922` | 0.0078125 | 0 | 0.00976562 | 0.00919118 |
| L2__024_moe_expert_parallel_execution | `3ade55ce` | 0.00195312 | 0 | 0.00347743 | 0.00899281 |
| L2__024_moe_expert_parallel_execution | `60097c14` | 0.00390625 | 0 | 0.00488281 | 0.00954198 |
| L2__024_moe_expert_parallel_execution | `7492600c` | 0.00390625 | 0 | 0.00488281 | 0.00856164 |
| L2__024_moe_expert_parallel_execution | `af9f7c32` | 0.00195312 | 0 | 0.00351065 | 0.00976562 |
| L2__024_moe_expert_parallel_execution | `b1b5c374` | 0.00195312 | 0 | 0.00348992 | 0.00856164 |
| L2__024_moe_expert_parallel_execution | `ba19612d` | 0.00390625 | 0 | 0.00488281 | 0.00961538 |
| L2__024_moe_expert_parallel_execution | `de4a688a` | 0.00390625 | 0 | 0.00488281 | 0.0093985 |
| L2__024_moe_expert_parallel_execution | `df563cc8` | 0.00390625 | 0 | 0.00488281 | 0.00954198 |
| L2__024_moe_expert_parallel_execution | `fa621ee3` | 0.00195312 | 0 | 0.00347346 | 0.00968992 |
| L2__033_multi_scale_feature_pyramid | `c880a2e4` | 557056 | 0 | 696320 | 0.411765 |
| L2__033_multi_scale_feature_pyramid | `cda7c568` | 278528 | 0 | 348160 | 0.0953388 |
| L2__050_vae_decoder_mid_block_attention_resnet | `67bdd5bc` | 7.82013e-05 | 0 | 9.77516e-05 | 0.38835 |
| L2__050_vae_decoder_mid_block_attention_resnet | `f009abdb` | 0.000110626 | 0 | 0.000138283 | 0.558333 |
| L2__057_residual_coupling_flow_block | `1652542c` | 0.00146484 | 0 | 0.00183105 | 0.120192 |
| L2__057_residual_coupling_flow_block | `3c84b2d6` | 0.00146484 | 0 | 0.00183105 | 0.0405093 |
| L2__057_residual_coupling_flow_block | `40d72809` | 0.00146484 | 0 | 0.00183105 | 0.208333 |
| L2__057_residual_coupling_flow_block | `5d0f4920` | 0.00146484 | 0 | 0.00183105 | 0.166667 |
| L2__057_residual_coupling_flow_block | `6128fbf5` | 0.00146484 | 0 | 0.00183105 | 0.159574 |
| L2__057_residual_coupling_flow_block | `f274d392` | 0.00146484 | 0 | 0.00183105 | 0.0833333 |
| L2__066_resnet_block_with_time_embedding | `1d09327e` | 2.28882e-05 | 0 | 2.86102e-05 | 0.166667 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `0fee490a` | 1.78814e-07 | 0 | 2.23517e-07 | 0.00883557 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `1247f979` | 9.53674e-07 | 0 | 1.19209e-06 | 0.000667379 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `2568fb2d` | 2.68221e-07 | 0 | 3.35276e-07 | 0.0026824 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `3a57d7f0` | 2.38419e-07 | 0 | 2.98023e-07 | 0.00265393 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `4a0f1201` | 2.98023e-07 | 0 | 3.72529e-07 | 0.06875 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `568d2852` | 7.15256e-07 | 0 | 8.9407e-07 | 0.00324675 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `570c1f47` | 4.76837e-07 | 0 | 5.96046e-07 | 0.00136911 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `85ac048c` | 1.49012e-07 | 0 | 1.86265e-07 | 0.00055991 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `c829153c` | 2.38419e-07 | 0 | 2.98023e-07 | 0.00480077 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `c856ce66` | 1.19209e-07 | 0 | 1.49012e-07 | 0.0487329 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `cb4ff597` | 1.19209e-07 | 0 | 1.49012e-07 | 0.0154746 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `d9ef480c` | 8.9407e-08 | 0 | 1.11759e-07 | 0.000543951 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `e89e9a82` | 1.19209e-07 | 0 | 1.49012e-07 | 0.000638407 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `ec61cccd` | 1.19209e-07 | 0 | 1.49012e-07 | 0.00206392 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `efdfcb05` | 1.78814e-07 | 0 | 2.23517e-07 | 0.00448994 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `fa1f8e5c` | 1.19209e-07 | 0 | 1.49012e-07 | 4.54976e-05 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `069dd70d` | 0.5 | 0 | 0.625 | 0.0078125 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `14705bec` | 0.5 | 0 | 0.625 | 0.015625 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `19a1a2c5` | 0.25 | 0 | 0.459528 | 0.0212514 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `1e32a29c` | 0.25 | 0 | 0.401379 | 0.0178571 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `205a19bb` | 0.03125 | 0 | 0.571741 | 0.0117925 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `3b134f80` | 0.25 | 0 | 0.463509 | 0.00811688 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `4a79e495` | 0.5 | 0 | 0.625 | 0.015625 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `a74598f8` | 0.25 | 0 | 0.409667 | 0.0195312 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `bf757f4d` | 1 | 0 | 1.25 | 0.1875 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c562ba02` | 0.25 | 0 | 0.39778 | 0.0245503 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c87730f3` | 0.125 | 0 | 0.388443 | 0.0125702 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `e0e443ab` | 0.5 | 0 | 0.625 | 0.0173611 |

## Tolerances more than 2.0x B200's

Each needs a reason. A 10x looser tolerance usually means something is wrong, not that CDNA4 is noisy.

Grouped by mechanism, because a flat list of hundreds of rows is a backlog rather than a triage. Only the last group needs a person.

### floor: bit-exact, atol = dtype epsilon — 895 workloads

| problem | workload | B200 atol | AMD atol | ratio |
|---|---|---|---|---|
| L2__033_multi_scale_feature_pyramid | `77f6632e` | 0.063 | 14874.6 | 236105.0x |
| L2__033_multi_scale_feature_pyramid | `e6b2118b` | 0.071 | 14783.2 | 208214.7x |
| L2__033_multi_scale_feature_pyramid | `13858024` | 0.072 | 14771.8 | 205163.9x |
| L2__033_multi_scale_feature_pyramid | `09b7ea3c` | 0.086 | 15397.4 | 179040.0x |
| L2__033_multi_scale_feature_pyramid | `5271f48c` | 0.088 | 15436.5 | 175414.5x |
| L2__033_multi_scale_feature_pyramid | `13a04ee9` | 0.08 | 13951 | 174387.5x |
| L2__033_multi_scale_feature_pyramid | `37f62c95` | 0.08 | 13941.9 | 174273.8x |
| L2__033_multi_scale_feature_pyramid | `fb8e5980` | 0.1 | 16068.6 | 160686.0x |
| L2__033_multi_scale_feature_pyramid | `8ddd08db` | 0.1 | 15871.9 | 158718.8x |
| L2__033_multi_scale_feature_pyramid | `a4fdc5ea` | 0.1 | 15730.6 | 157305.6x |
| L2__033_multi_scale_feature_pyramid | `207eeef1` | 0.1 | 15454.4 | 154544.3x |
| L2__033_multi_scale_feature_pyramid | `bf8e1e76` | 0.099 | 15296.7 | 154512.4x |
| L2__033_multi_scale_feature_pyramid | `a873acb1` | 0.11 | 14834.6 | 134860.2x |
| L2__033_multi_scale_feature_pyramid | `615b0564` | 0.12 | 15235.4 | 126961.5x |
| L1__004_attention_output_projection_with_reshape_backward | `1124cf38` | 1e-05 | 0.117843 | 11784.3x |
| L1__079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward | `31f3d024` | 1e-05 | 0.110907 | 11090.7x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `b7d2ce3a` | 1e-05 | 0.0940942 | 9409.4x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `314bcaf9` | 1e-05 | 0.0940547 | 9405.5x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `77a66754` | 1e-05 | 0.0939269 | 9392.7x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `405de7ad` | 1e-05 | 0.0939102 | 9391.0x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `b16ebbcb` | 1e-05 | 0.093852 | 9385.2x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `46445623` | 1e-05 | 0.0936005 | 9360.1x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `213a8256` | 1e-05 | 0.093579 | 9357.9x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `019db6f8` | 1e-05 | 0.0935701 | 9357.0x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `34c7ec75` | 1e-05 | 0.0935676 | 9356.8x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `6c45c19e` | 1e-05 | 0.0935467 | 9354.7x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `cccd1cb3` | 1e-05 | 0.0932876 | 9328.8x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `cfca350b` | 1e-05 | 0.0932034 | 9320.3x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `d947725d` | 1e-05 | 0.0931943 | 9319.4x |
| L1__070_mamba2_fused_intra_chunk_diagonal_computation | `336c7221` | 1e-05 | 0.0929186 | 9291.9x |
| L1__079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward | `3c008d2a` | 1e-05 | 0.0766363 | 7663.6x |
| L2__044_mamba_discretization_and_segsum | `f7c98ff5` | 1e-05 | 0.0755028 | 7550.3x |
| L2__044_mamba_discretization_and_segsum | `388d7529` | 1e-05 | 0.0753256 | 7532.6x |
| L2__044_mamba_discretization_and_segsum | `e9a72dde` | 1e-05 | 0.0753256 | 7532.6x |
| L2__044_mamba_discretization_and_segsum | `70903611` | 1e-05 | 0.0753204 | 7532.0x |
| L2__044_mamba_discretization_and_segsum | `147b8cd4` | 1e-05 | 0.0752686 | 7526.9x |
| L2__044_mamba_discretization_and_segsum | `0fd73416` | 1e-05 | 0.0751674 | 7516.7x |
| L2__044_mamba_discretization_and_segsum | `5c7731b0` | 1e-05 | 0.07498 | 7498.0x |
| L2__044_mamba_discretization_and_segsum | `0b325782` | 1e-05 | 0.07498 | 7498.0x |
| L2__044_mamba_discretization_and_segsum | `28b1adc9` | 1e-05 | 0.07498 | 7498.0x |
| ... and 855 more | | | | |

### measured run-to-run variance — 57 workloads

| problem | workload | B200 atol | AMD atol | ratio |
|---|---|---|---|---|
| L2__033_multi_scale_feature_pyramid | `c880a2e4` | 0.074 | 696320 | 9409729.7x |
| L2__033_multi_scale_feature_pyramid | `cda7c568` | 0.063 | 348160 | 5526349.2x |
| L1__087_embedding_with_initial_layernorm_backward | `5d6fcf80` | 1e-05 | 0.00960052 | 960.1x |
| L1__087_embedding_with_initial_layernorm_backward | `a6d899cd` | 1e-05 | 0.00960052 | 960.1x |
| L1__087_embedding_with_initial_layernorm_backward | `a70755cd` | 1e-05 | 0.00677592 | 677.6x |
| L1__087_embedding_with_initial_layernorm_backward | `b54a2121` | 1e-05 | 0.00583629 | 583.6x |
| L2__015_audio_sinusoidal_position_embedding_with_conv_projection | `d982c737` | 1.1 | 320 | 290.9x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `bf757f4d` | 0.014 | 1.25 | 89.3x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `069dd70d` | 0.013 | 0.625 | 48.1x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `e0e443ab` | 0.017 | 0.625 | 36.8x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `205a19bb` | 0.019 | 0.571741 | 30.1x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `4a79e495` | 0.021 | 0.625 | 29.8x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `19a1a2c5` | 0.016 | 0.459528 | 28.7x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `3b134f80` | 0.017 | 0.463509 | 27.3x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c562ba02` | 0.016 | 0.39778 | 24.9x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `14705bec` | 0.028 | 0.625 | 22.3x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `1e32a29c` | 0.019 | 0.401379 | 21.1x |
| L2__024_moe_expert_parallel_execution | `154b4946` | 0.00048 | 0.00976562 | 20.3x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `a74598f8` | 0.022 | 0.409667 | 18.6x |
| L2__024_moe_expert_parallel_execution | `032fc60b` | 0.00057 | 0.00976562 | 17.1x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c87730f3` | 0.023 | 0.388443 | 16.9x |
| L2__024_moe_expert_parallel_execution | `0887b5af` | 0.00059 | 0.00976562 | 16.6x |
| L2__024_moe_expert_parallel_execution | `2ec82922` | 0.00065 | 0.00976562 | 15.0x |
| L2__024_moe_expert_parallel_execution | `00c0833a` | 0.00042 | 0.00488281 | 11.6x |
| L2__024_moe_expert_parallel_execution | `de4a688a` | 0.00045 | 0.00488281 | 10.9x |
| L2__024_moe_expert_parallel_execution | `7492600c` | 0.00045 | 0.00488281 | 10.9x |
| L2__024_moe_expert_parallel_execution | `df563cc8` | 0.00048 | 0.00488281 | 10.2x |
| L2__024_moe_expert_parallel_execution | `60097c14` | 0.00049 | 0.00488281 | 10.0x |
| L2__024_moe_expert_parallel_execution | `1367e254` | 0.00051 | 0.00488281 | 9.6x |
| L2__024_moe_expert_parallel_execution | `2b597deb` | 0.00047 | 0.00348051 | 7.4x |
| L2__024_moe_expert_parallel_execution | `ba19612d` | 0.0007 | 0.00488281 | 7.0x |
| L2__024_moe_expert_parallel_execution | `b1b5c374` | 0.00051 | 0.00348992 | 6.8x |
| L2__024_moe_expert_parallel_execution | `3ade55ce` | 0.00053 | 0.00347743 | 6.6x |
| L2__024_moe_expert_parallel_execution | `af9f7c32` | 0.00057 | 0.00351065 | 6.2x |
| L2__024_moe_expert_parallel_execution | `fa621ee3` | 0.00057 | 0.00347346 | 6.1x |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `bf41eb28` | 0.0059 | 0.0244141 | 4.1x |
| L1__008_expert_output_weighted_index_add_accumulation | `0f250bfd` | 0.078 | 0.3125 | 4.0x |
| L1__008_expert_output_weighted_index_add_accumulation | `ada14c03` | 0.087 | 0.3125 | 3.6x |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `18872872` | 0.0068 | 0.0244141 | 3.6x |
| L1__008_expert_output_weighted_index_add_accumulation | `a57cc129` | 0.09 | 0.3125 | 3.5x |
| ... and 17 more | | | | |


## Tolerances TIGHTER than B200's

1904 workloads. Not a problem — a tighter tolerance rejects more, not less — but worth seeing: it means the AMD reference is more reproducible than B200's calibration assumed, usually because the kernel is bit-exact here and the derived value fell to the dtype epsilon floor.

## Exact matches with B200

0 workloads have tolerances numerically identical to upstream's.

This is reported because prime directive 2 forbids copying an NVIDIA constant into an AMD artifact, and an automated check cannot tell a copy from a coincidence. Here it is a coincidence with a mechanism: for a reference that is bit-exact on both platforms, both procedures floor at the same dtype epsilon and therefore agree. Agreement of that kind is the correct answer, not a smell.
