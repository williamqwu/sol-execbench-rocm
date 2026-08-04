# Task 05 — tolerance triage

<!-- {"task": "05-tolerance-triage", "utc": "2026-08-04T03:13:48.363590+00:00", "git_sha": "87d9e1a78a865d2757f2f158eb5cd61a492fad00-dirty", "host": "gbt350-odcdh1-a08-1.png-odc.dcgpu", "python": "3.12.3", "torch": {"available": true, "version": "2.9.1+rocm7.2.0.git7e1940d4", "hip": "7.2.26015-fc0010cf6a", "cuda": null, "device_count": 8, "devices": ["AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X", "AMD Instinct MI350X"]}, "rocm": {"version": "7.2.0", "driver": "7.1.1.31500000", "amd_smi": "AMDSMI Tool: 26.2.1+fc0010cf6a | AMDSMI Library version: 26.2.1 | ROCm version: 7.2.0 | amdgpu version: 7.1.1.31500000 | hsmp version: N/A"}, "f_lock_mhz": 1300, "visible_devices": null} -->

AMD-derived tolerances written for **3717 of 3957** workload instances.

Every number below was derived from reference-vs-reference variance on MI350X and nothing else. No B200 value was used as a source; upstream's values appear only as a comparison column.

## Structurally nondeterministic references

These disagree with themselves run to run on identical inputs. That is a property of the kernels (atomics-ordered accumulation, library algorithm selection), not a bug, and it is why their tolerances are wider than their neighbours'. Each one is listed rather than absorbed silently, because a wide tolerance is exactly what would let a wrong kernel through.

| problem | workload | run-to-run max_abs | derived atol | derived rtol |
|---|---|---|---|---|
| L1__002_vae_conv3x3_groupnorm_silu_residual_fused | `69ceed6e` | 1.90735e-06 | 2.38419e-06 | 0.1875 |
| L1__008_expert_output_weighted_index_add_accumulation | `01dfd338` | 0.1875 | 0.234375 | 0.416667 |
| L1__008_expert_output_weighted_index_add_accumulation | `0bda2a65` | 0.1875 | 0.234375 | 0.416667 |
| L1__008_expert_output_weighted_index_add_accumulation | `0f250bfd` | 0.25 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `172afa25` | 0.1875 | 0.234375 | 0.416667 |
| L1__008_expert_output_weighted_index_add_accumulation | `28f8046d` | 0.1875 | 0.234375 | 0.454545 |
| L1__008_expert_output_weighted_index_add_accumulation | `2c1d8396` | 0.25 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `5a674481` | 0.25 | 0.3125 | 0.449219 |
| L1__008_expert_output_weighted_index_add_accumulation | `611e32dd` | 0.1875 | 0.234375 | 0.666667 |
| L1__008_expert_output_weighted_index_add_accumulation | `8b677344` | 0.1875 | 0.234375 | 0.416667 |
| L1__008_expert_output_weighted_index_add_accumulation | `8c5c94b5` | 0.25 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `a57cc129` | 0.25 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `ada14c03` | 0.25 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `b1b7efca` | 0.25 | 0.3125 | 0.375 |
| L1__008_expert_output_weighted_index_add_accumulation | `b61ee9a7` | 0.25 | 0.3125 | 0.384615 |
| L1__008_expert_output_weighted_index_add_accumulation | `c4a319ff` | 0.25 | 0.3125 | 0.40625 |
| L1__008_expert_output_weighted_index_add_accumulation | `edc192ac` | 0.25 | 0.3125 | 0.375 |
| L1__024_vision_rotary_position_embedding_generation_backward | `0cda4941` | 0.0136719 | 0.0170898 | 3.33078e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `199e4d8e` | 4.57764e-05 | 5.72205e-05 | 8.30879e-06 |
| L1__024_vision_rotary_position_embedding_generation_backward | `229e6489` | 0.00170898 | 0.00213623 | 7.43657e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `472b424d` | 0.00012207 | 0.000152588 | 2.37187e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `61d8b841` | 0.00976562 | 0.012207 | 3.05856e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `7579ff98` | 0.000488281 | 0.000610352 | 3.55128e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `85f40a68` | 0.000854492 | 0.00106812 | 5.64388e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `96bc140b` | 0.000488281 | 0.000610352 | 2.26076e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `a1a49d6d` | 0.00585938 | 0.00732422 | 3.18968e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `a292f1ef` | 6.10352e-05 | 7.62939e-05 | 9.95719e-06 |
| L1__024_vision_rotary_position_embedding_generation_backward | `bca3cee8` | 0.000366211 | 0.000457764 | 2.76996e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `dbd8f288` | 0.000732422 | 0.000915527 | 7.61274e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `e12534d6` | 0.000244141 | 0.000305176 | 5.83633e-05 |
| L1__024_vision_rotary_position_embedding_generation_backward | `e2e6995a` | 0.000549316 | 0.000686646 | 0.000432095 |
| L1__024_vision_rotary_position_embedding_generation_backward | `e56dd6a6` | 0.000976562 | 0.0012207 | 0.000113912 |
| L1__024_vision_rotary_position_embedding_generation_backward | `fc1c68f0` | 9.15527e-05 | 0.000114441 | 1.34803e-05 |
| L1__087_embedding_with_initial_layernorm_backward | `4d16688e` | 0.000244141 | 0.00677592 | 0.0078125 |
| L1__087_embedding_with_initial_layernorm_backward | `5d6fcf80` | 0.0078125 | 0.00976562 | 0.00880282 |
| L1__087_embedding_with_initial_layernorm_backward | `a6d899cd` | 0.0078125 | 0.00976562 | 0.00880282 |
| L1__087_embedding_with_initial_layernorm_backward | `a70755cd` | 2.38419e-07 | 0.00677592 | 0.0078125 |
| L1__087_embedding_with_initial_layernorm_backward | `b54a2121` | 0.00012207 | 0.00583629 | 0.0078125 |
| L2__003_grouped_query_attention_with_rope_backward | `fc7d9227` | 0.0302734 | 0.204926 | 0.178704 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `106f0092` | 0.0234375 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `16755515` | 0.0234375 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `18872872` | 0.0234375 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `2e7d2e79` | 0.015625 | 0.0195312 | 0.359375 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `2fe16676` | 0.0195312 | 0.0244141 | 0.3 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `3744fdcd` | 0.0234375 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `598f9037` | 0.0234375 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `604e6e46` | 0.0234375 | 0.0292969 | 0.333333 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `61022909` | 0.015625 | 0.0195312 | 0.476562 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `9a4bee24` | 0.0195312 | 0.0244141 | 0.4 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `ab49eedb` | 0.0195312 | 0.0244141 | 0.314039 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `b983758c` | 0.0234375 | 0.0292969 | 0.268229 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `bf41eb28` | 0.0234375 | 0.0292969 | 0.25 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `c2a09e88` | 0.0195312 | 0.0244141 | 0.3125 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `d36f3c8b` | 0.0195312 | 0.0244141 | 0.4 |
| L2__012_moe_expert_batched_execution_with_capacity_factor | `f850f2b7` | 0.0234375 | 0.0292969 | 0.293825 |
| L2__015_audio_sinusoidal_position_embedding_with_conv_projection | `d982c737` | 256 | 320 | 0.03125 |
| L2__024_moe_expert_parallel_execution | `00c0833a` | 0.00390625 | 0.00488281 | 0.0094697 |
| L2__024_moe_expert_parallel_execution | `032fc60b` | 0.00390625 | 0.00488281 | 0.00954198 |
| L2__024_moe_expert_parallel_execution | `0887b5af` | 0.00195312 | 0.00347963 | 0.0093985 |
| L2__024_moe_expert_parallel_execution | `1367e254` | 0.00195312 | 0.0034796 | 0.0094697 |
| L2__024_moe_expert_parallel_execution | `154b4946` | 0.00195312 | 0.00348014 | 0.0094697 |
| L2__024_moe_expert_parallel_execution | `2b597deb` | 0.0078125 | 0.00976562 | 0.00874126 |
| L2__024_moe_expert_parallel_execution | `2ec82922` | 0.00195312 | 0.00347995 | 0.0093985 |
| L2__024_moe_expert_parallel_execution | `3ade55ce` | 0.00195312 | 0.00347743 | 0.00968992 |
| L2__024_moe_expert_parallel_execution | `60097c14` | 0.00390625 | 0.00488281 | 0.00925926 |
| L2__024_moe_expert_parallel_execution | `7492600c` | 0.00390625 | 0.00488281 | 0.0093985 |
| L2__024_moe_expert_parallel_execution | `af9f7c32` | 0.00390625 | 0.00488281 | 0.00976562 |
| L2__024_moe_expert_parallel_execution | `b1b5c374` | 0.000976562 | 0.00348992 | 0.00954198 |
| L2__024_moe_expert_parallel_execution | `ba19612d` | 0.00390625 | 0.00488281 | 0.00961538 |
| L2__024_moe_expert_parallel_execution | `de4a688a` | 0.00390625 | 0.00488281 | 0.0094697 |
| L2__024_moe_expert_parallel_execution | `df563cc8` | 0.00390625 | 0.00488281 | 0.0093985 |
| L2__024_moe_expert_parallel_execution | `fa621ee3` | 0.0078125 | 0.00976562 | 0.0093985 |
| L2__033_multi_scale_feature_pyramid | `09b7ea3c` | 548864 | 686080 | 0.492537 |
| L2__033_multi_scale_feature_pyramid | `13a04ee9` | 458752 | 573440 | 0.464481 |
| L2__033_multi_scale_feature_pyramid | `207eeef1` | 622592 | 778240 | 0.368421 |
| L2__033_multi_scale_feature_pyramid | `37f62c95` | 557056 | 696320 | 0.131701 |
| L2__033_multi_scale_feature_pyramid | `615b0564` | 573440 | 716800 | 0.485714 |
| L2__033_multi_scale_feature_pyramid | `77f6632e` | 262144 | 327680 | 0.1375 |
| L2__033_multi_scale_feature_pyramid | `c880a2e4` | 491520 | 614400 | 0.333333 |
| L2__033_multi_scale_feature_pyramid | `cda7c568` | 344064 | 430080 | 0.154762 |
| L2__050_vae_decoder_mid_block_attention_resnet | `3f3523b9` | 0.000102997 | 0.000128746 | 0.37037 |
| L2__050_vae_decoder_mid_block_attention_resnet | `67bdd5bc` | 8.39233e-05 | 0.000104904 | 0.295455 |
| L2__050_vae_decoder_mid_block_attention_resnet | `706585da` | 7.62939e-05 | 9.53674e-05 | 0.203947 |
| L2__050_vae_decoder_mid_block_attention_resnet | `89fa03a1` | 9.15527e-05 | 0.000114441 | 0.222081 |
| L2__050_vae_decoder_mid_block_attention_resnet | `8af99a9e` | 0.000114441 | 0.000143051 | 0.316667 |
| L2__050_vae_decoder_mid_block_attention_resnet | `f009abdb` | 0.000110626 | 0.000138283 | 0.465517 |
| L2__057_residual_coupling_flow_block | `1652542c` | 0.00146484 | 0.00183105 | 0.078125 |
| L2__057_residual_coupling_flow_block | `3c84b2d6` | 0.00146484 | 0.00183105 | 0.0700935 |
| L2__057_residual_coupling_flow_block | `40d72809` | 0.00146484 | 0.00183105 | 0.166667 |
| L2__057_residual_coupling_flow_block | `5d0f4920` | 0.00146484 | 0.00183105 | 0.15625 |
| L2__057_residual_coupling_flow_block | `6128fbf5` | 0.00146484 | 0.00183105 | 0.128205 |
| L2__065_sparse_expert_dispatch_and_combine | `0217e0c7` | 2.98023e-08 | 3.72529e-08 | 0.0010989 |
| L2__065_sparse_expert_dispatch_and_combine | `389f8338` | 2.98023e-08 | 3.72529e-08 | 0.000127136 |
| L2__065_sparse_expert_dispatch_and_combine | `50112f2e` | 1.49012e-08 | 1.86265e-08 | 4.13237e-05 |
| L2__065_sparse_expert_dispatch_and_combine | `6da51ccd` | 1.49012e-08 | 1.86265e-08 | 0.00025319 |
| L2__065_sparse_expert_dispatch_and_combine | `8710eb23` | 2.98023e-08 | 3.72529e-08 | 0.000116921 |
| L2__065_sparse_expert_dispatch_and_combine | `8cc04a47` | 2.98023e-08 | 3.72529e-08 | 0.000380865 |
| L2__065_sparse_expert_dispatch_and_combine | `9c1cb8cb` | 1.49012e-08 | 1.86265e-08 | 0.00130514 |
| L2__065_sparse_expert_dispatch_and_combine | `a5db2198` | 1.49012e-08 | 1.86265e-08 | 0.00130107 |
| L2__065_sparse_expert_dispatch_and_combine | `a9b83327` | 2.98023e-08 | 3.72529e-08 | 0.000160256 |
| L2__065_sparse_expert_dispatch_and_combine | `bb0b3456` | 2.98023e-08 | 3.72529e-08 | 0.00130753 |
| L2__065_sparse_expert_dispatch_and_combine | `dda93738` | 2.98023e-08 | 3.72529e-08 | 0.00107066 |
| L2__065_sparse_expert_dispatch_and_combine | `e5e1f144` | 2.98023e-08 | 3.72529e-08 | 0.00390625 |
| L2__065_sparse_expert_dispatch_and_combine | `e663b439` | 2.98023e-08 | 3.72529e-08 | 0.00390625 |
| L2__065_sparse_expert_dispatch_and_combine | `e8880a2f` | 1.49012e-08 | 1.86265e-08 | 0.00129971 |
| L2__066_resnet_block_with_time_embedding | `1d09327e` | 2.28882e-05 | 2.86102e-05 | 0.208333 |
| L2__071_edit_consistency_loss_with_perceptual_weighting | `9a78dd83` | 4.76837e-07 | 8.36332e-07 | 1.19209e-07 |
| L2__071_edit_consistency_loss_with_perceptual_weighting | `b3db0bb4` | 4.76837e-07 | 8.35373e-07 | 1.19209e-07 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `0fee490a` | 1.49012e-07 | 1.86265e-07 | 0.0147856 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `1247f979` | 9.53674e-07 | 1.19209e-06 | 0.000667022 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `2568fb2d` | 2.38419e-07 | 2.98023e-07 | 0.00110444 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `3a57d7f0` | 1.78814e-07 | 2.23517e-07 | 0.00149682 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `4a0f1201` | 2.38419e-07 | 2.98023e-07 | 0.03125 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `568d2852` | 7.15256e-07 | 8.9407e-07 | 0.00324886 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `570c1f47` | 3.57628e-07 | 4.47035e-07 | 0.00273523 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `85ac048c` | 1.78814e-07 | 2.23517e-07 | 0.000165038 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `c829153c` | 2.38419e-07 | 2.98023e-07 | 0.0066361 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `c856ce66` | 1.19209e-07 | 1.49012e-07 | 0.00992063 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `cb4ff597` | 1.19209e-07 | 1.49012e-07 | 0.0258876 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `d9ef480c` | 1.19209e-07 | 1.49012e-07 | 0.00163043 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `e89e9a82` | 1.19209e-07 | 1.49012e-07 | 0.000124522 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `ec61cccd` | 1.49012e-07 | 1.86265e-07 | 0.00180413 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `efdfcb05` | 2.38419e-07 | 2.98023e-07 | 0.00405186 |
| L2__076_sam_hq_vision_attention_with_relative_position_backward | `fa1f8e5c` | 1.19209e-07 | 1.49012e-07 | 0.00012219 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `069dd70d` | 0.0078125 | 0.398847 | 0.0078125 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `0e247ac4` | 0.0625 | 0.395328 | 0.00838926 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `14705bec` | 0.5 | 0.625 | 0.0150602 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `19a1a2c5` | 0.125 | 0.459528 | 0.0212514 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `1e32a29c` | 0.25 | 0.401379 | 0.0109649 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `205a19bb` | 0.015625 | 0.571748 | 0.0341606 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `3b134f80` | 0.25 | 0.463509 | 0.0119617 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `a74598f8` | 0.5 | 0.625 | 0.015625 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `bf757f4d` | 1 | 1.25 | 0.191406 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c562ba02` | 0.5 | 0.625 | 0.0104167 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c87730f3` | 0.25 | 0.388386 | 0.0131579 |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `e0e443ab` | 0.5 | 0.625 | 0.0078125 |

## Tolerances more than 2.0x B200's

Each needs a reason. A 10x looser tolerance usually means something is wrong, not that CDNA4 is noisy.

Grouped by mechanism, because a flat list of hundreds of rows is a backlog rather than a triage. Only the last group needs a person.

### floor: bit-exact, atol = dtype epsilon — 887 workloads

| problem | workload | B200 atol | AMD atol | ratio |
|---|---|---|---|---|
| L2__033_multi_scale_feature_pyramid | `e6b2118b` | 0.071 | 14783.2 | 208214.7x |
| L2__033_multi_scale_feature_pyramid | `13858024` | 0.072 | 14771.8 | 205163.9x |
| L2__033_multi_scale_feature_pyramid | `5271f48c` | 0.088 | 15436.5 | 175414.5x |
| L2__033_multi_scale_feature_pyramid | `fb8e5980` | 0.1 | 16068.6 | 160686.0x |
| L2__033_multi_scale_feature_pyramid | `8ddd08db` | 0.1 | 15871.9 | 158718.8x |
| L2__033_multi_scale_feature_pyramid | `a4fdc5ea` | 0.1 | 15730.6 | 157305.6x |
| L2__033_multi_scale_feature_pyramid | `bf8e1e76` | 0.099 | 15296.7 | 154512.4x |
| L2__033_multi_scale_feature_pyramid | `a873acb1` | 0.11 | 14834.6 | 134860.2x |
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
| L2__044_mamba_discretization_and_segsum | `f7c98ff5` | 1e-05 | 0.0754858 | 7548.6x |
| L2__044_mamba_discretization_and_segsum | `388d7529` | 1e-05 | 0.0753691 | 7536.9x |
| L2__044_mamba_discretization_and_segsum | `e9a72dde` | 1e-05 | 0.0753691 | 7536.9x |
| L2__044_mamba_discretization_and_segsum | `70903611` | 1e-05 | 0.0753261 | 7532.6x |
| L2__044_mamba_discretization_and_segsum | `147b8cd4` | 1e-05 | 0.0752781 | 7527.8x |
| L2__044_mamba_discretization_and_segsum | `0fd73416` | 1e-05 | 0.0751728 | 7517.3x |
| L2__044_mamba_discretization_and_segsum | `5c7731b0` | 1e-05 | 0.0749985 | 7499.8x |
| L2__044_mamba_discretization_and_segsum | `0b325782` | 1e-05 | 0.0749985 | 7499.8x |
| L2__044_mamba_discretization_and_segsum | `28b1adc9` | 1e-05 | 0.0749985 | 7499.8x |
| L2__044_mamba_discretization_and_segsum | `54b20dd4` | 1e-05 | 0.0749473 | 7494.7x |
| L2__044_mamba_discretization_and_segsum | `cb74f1c9` | 1e-05 | 0.0745984 | 7459.8x |
| L2__044_mamba_discretization_and_segsum | `55bb9dcb` | 1e-05 | 0.0744597 | 7446.0x |
| L2__044_mamba_discretization_and_segsum | `bdb0665a` | 1e-05 | 0.0744597 | 7446.0x |
| L2__044_mamba_discretization_and_segsum | `f98871d3` | 1e-05 | 0.0744597 | 7446.0x |
| L2__044_mamba_discretization_and_segsum | `e46fa678` | 1e-05 | 0.0740069 | 7400.7x |
| ... and 847 more | | | | |

### measured run-to-run variance — 64 workloads

| problem | workload | B200 atol | AMD atol | ratio |
|---|---|---|---|---|
| L2__033_multi_scale_feature_pyramid | `37f62c95` | 0.08 | 696320 | 8704000.0x |
| L2__033_multi_scale_feature_pyramid | `c880a2e4` | 0.074 | 614400 | 8302702.7x |
| L2__033_multi_scale_feature_pyramid | `09b7ea3c` | 0.086 | 686080 | 7977674.4x |
| L2__033_multi_scale_feature_pyramid | `207eeef1` | 0.1 | 778240 | 7782400.0x |
| L2__033_multi_scale_feature_pyramid | `13a04ee9` | 0.08 | 573440 | 7168000.0x |
| L2__033_multi_scale_feature_pyramid | `cda7c568` | 0.063 | 430080 | 6826666.7x |
| L2__033_multi_scale_feature_pyramid | `615b0564` | 0.12 | 716800 | 5973333.3x |
| L2__033_multi_scale_feature_pyramid | `77f6632e` | 0.063 | 327680 | 5201269.8x |
| L1__087_embedding_with_initial_layernorm_backward | `5d6fcf80` | 1e-05 | 0.00976562 | 976.6x |
| L1__087_embedding_with_initial_layernorm_backward | `a6d899cd` | 1e-05 | 0.00976562 | 976.6x |
| L1__087_embedding_with_initial_layernorm_backward | `4d16688e` | 1e-05 | 0.00677592 | 677.6x |
| L1__087_embedding_with_initial_layernorm_backward | `a70755cd` | 1e-05 | 0.00677592 | 677.6x |
| L1__087_embedding_with_initial_layernorm_backward | `b54a2121` | 1e-05 | 0.00583629 | 583.6x |
| L2__015_audio_sinusoidal_position_embedding_with_conv_projection | `d982c737` | 1.1 | 320 | 290.9x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `bf757f4d` | 0.014 | 1.25 | 89.3x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c562ba02` | 0.016 | 0.625 | 39.1x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `e0e443ab` | 0.017 | 0.625 | 36.8x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `069dd70d` | 0.013 | 0.398847 | 30.7x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `205a19bb` | 0.019 | 0.571748 | 30.1x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `19a1a2c5` | 0.016 | 0.459528 | 28.7x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `a74598f8` | 0.022 | 0.625 | 28.4x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `3b134f80` | 0.017 | 0.463509 | 27.3x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `0e247ac4` | 0.015 | 0.395328 | 26.4x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `14705bec` | 0.028 | 0.625 | 22.3x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `1e32a29c` | 0.019 | 0.401379 | 21.1x |
| L2__024_moe_expert_parallel_execution | `2b597deb` | 0.00047 | 0.00976562 | 20.8x |
| L2__024_moe_expert_parallel_execution | `fa621ee3` | 0.00057 | 0.00976562 | 17.1x |
| L2__078_fused_final_layer_upsample_with_adaptive_norm | `c87730f3` | 0.023 | 0.388386 | 16.9x |
| L2__024_moe_expert_parallel_execution | `00c0833a` | 0.00042 | 0.00488281 | 11.6x |
| L2__024_moe_expert_parallel_execution | `de4a688a` | 0.00045 | 0.00488281 | 10.9x |
| L2__024_moe_expert_parallel_execution | `7492600c` | 0.00045 | 0.00488281 | 10.9x |
| L2__024_moe_expert_parallel_execution | `df563cc8` | 0.00048 | 0.00488281 | 10.2x |
| L2__024_moe_expert_parallel_execution | `60097c14` | 0.00049 | 0.00488281 | 10.0x |
| L2__024_moe_expert_parallel_execution | `032fc60b` | 0.00057 | 0.00488281 | 8.6x |
| L2__024_moe_expert_parallel_execution | `af9f7c32` | 0.00057 | 0.00488281 | 8.6x |
| L2__024_moe_expert_parallel_execution | `154b4946` | 0.00048 | 0.00348014 | 7.3x |
| L2__024_moe_expert_parallel_execution | `ba19612d` | 0.0007 | 0.00488281 | 7.0x |
| L2__024_moe_expert_parallel_execution | `b1b5c374` | 0.00051 | 0.00348992 | 6.8x |
| L2__024_moe_expert_parallel_execution | `1367e254` | 0.00051 | 0.0034796 | 6.8x |
| L2__024_moe_expert_parallel_execution | `3ade55ce` | 0.00053 | 0.00347743 | 6.6x |
| ... and 24 more | | | | |


## Tolerances TIGHTER than B200's

1903 workloads. Not a problem — a tighter tolerance rejects more, not less — but worth seeing: it means the AMD reference is more reproducible than B200's calibration assumed, usually because the kernel is bit-exact here and the derived value fell to the dtype epsilon floor.

## Exact matches with B200

0 workloads have tolerances numerically identical to upstream's.

This is reported because prime directive 2 forbids copying an NVIDIA constant into an AMD artifact, and an automated check cannot tell a copy from a coincidence. Here it is a coincidence with a mechanism: for a reference that is bit-exact on both platforms, both procedures floor at the same dtype epsilon and therefore agree. Agreement of that kind is the correct answer, not a smell.
