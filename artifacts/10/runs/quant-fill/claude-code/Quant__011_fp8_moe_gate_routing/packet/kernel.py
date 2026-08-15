import torch
import triton
import triton.language as tl


@triton.jit
def fp8_gemm_kernel(
    # Pointers
    X_ptr, W_ptr, Out_ptr,
    scale_x_ptr, scale_w_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Scale strides
    stride_sx_m, stride_sx_k,
    stride_sw_n, stride_sw_k,
    # Block size
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in blocks of 128
    for k_block in range(0, K, 128):
        offs_k = k_block + tl.arange(0, 128)

        # Load FP8 values
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Load scales for this block
        # scale_x: [M, K//128], scale_w: [N//128, K//128]
        scale_x_idx_m = offs_m
        scale_x_idx_k = k_block // 128
        scale_w_idx_n = offs_n // 128
        scale_w_idx_k = k_block // 128

        scale_x_ptrs = scale_x_ptr + scale_x_idx_m[:, None] * stride_sx_m + scale_x_idx_k * stride_sx_k
        scale_w_ptrs = scale_w_ptr + scale_w_idx_n[:, None] * stride_sw_n + scale_w_idx_k * stride_sw_k

        scale_x_mask = offs_m[:, None] < M
        scale_w_mask = offs_n[:, None] < N

        scale_x = tl.load(scale_x_ptrs, mask=scale_x_mask, other=1.0)
        scale_w = tl.load(scale_w_ptrs, mask=scale_w_mask, other=1.0)

        # Dequantize
        x_deq = x * scale_x
        w_deq = w * scale_w[:, None]

        # Accumulate
        acc += tl.dot(x_deq, tl.trans(w_deq))

    # Store output
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    FP8-quantized MoE gating with top-k expert selection.
    """
    # Constants
    n_routed_experts = 256
    num_experts_per_tok = 8
    n_group = 8
    topk_group = 4

    num_tokens = hidden_states.shape[0]
    hidden_size = 7168

    # Step 1: Convert to FP32 and apply FP8 scaling
    E4M3_MAX = 448.0

    # Scale and quantize hidden_states
    hidden_states_fp32 = hidden_states.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)

    # Apply blockwise scaling for activation (BlockWise1x128)
    # Reshape to apply block scales
    h_reshaped = hidden_states_fp32.reshape(num_tokens, hidden_size // 128, 128)
    scale_x_expanded = scale_x.unsqueeze(2)  # [num_tokens, hidden_blocks, 1]
    x_scaled = h_reshaped / scale_x_expanded
    x_scaled = torch.clamp(x_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    x_scaled = x_scaled.reshape(num_tokens, hidden_size)
    qx = x_scaled.to(torch.float8_e4m3fn)

    # Apply blockwise scaling for weight (BlockWise128x128)
    # weight is [256, 7168], transpose to [7168, 256] for blockwise scaling
    weight_t = weight_fp32.T  # [7168, 256]
    w_reshaped = weight_t.reshape(hidden_size // 128, 128, n_routed_experts // 128, 128)
    scale_w_expanded = scale_w.unsqueeze(1).unsqueeze(3)  # [hidden_blocks, 1, expert_blocks, 1]
    w_scaled = w_reshaped / scale_w_expanded
    w_scaled = torch.clamp(w_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    w_scaled = w_scaled.reshape(hidden_size, n_routed_experts)
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # [256, 7168]

    # Step 2: FP8 GEMM with dequantization
    # Dequantize and compute: qx @ qw.T
    x_reshaped = qx.to(torch.float32).reshape(num_tokens, hidden_size // 128, 128)
    scale_x_expanded = scale_x.unsqueeze(2)
    x_deq = x_reshaped * scale_x_expanded
    x_deq = x_deq.reshape(num_tokens, hidden_size)

    qw_t = qw.T.to(torch.float32)  # [7168, 256]
    w_reshaped = qw_t.reshape(hidden_size // 128, 128, n_routed_experts // 128, 128)
    scale_w_expanded = scale_w.unsqueeze(1).unsqueeze(3)
    w_deq = w_reshaped * scale_w_expanded
    w_deq = w_deq.reshape(hidden_size, n_routed_experts)

    logits = x_deq @ w_deq
    logits = logits.to(torch.bfloat16)

    # Step 3: Sigmoid activation
    scores = torch.sigmoid(logits.to(torch.float32))

    # Step 4: Add score correction bias
    scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)

    # Step 5: Group-based top-k selection
    experts_per_group = n_routed_experts // n_group  # 32
    group_scores_reshaped = scores_for_choice.view(num_tokens, n_group, experts_per_group)

    # Select top-2 experts per group and sum their scores
    group_scores = group_scores_reshaped.topk(2, dim=-1)[0].sum(dim=-1)

    # Select top-4 groups out of 8
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]

    # Create group mask
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Expand mask to expert level
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, experts_per_group)
        .reshape(num_tokens, n_routed_experts)
    )

    # Step 6: Mask out non-selected groups and perform final top-k
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(tmp_scores, k=num_experts_per_tok, dim=-1, sorted=False)

    # Step 7: Gather final weights and normalize
    topk_weight = scores.gather(1, topk_idx)

    # Normalize weights
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator

    # Apply routing scaling factor
    topk_weight = topk_weight * routed_scaling_factor

    return topk_idx, topk_weight
