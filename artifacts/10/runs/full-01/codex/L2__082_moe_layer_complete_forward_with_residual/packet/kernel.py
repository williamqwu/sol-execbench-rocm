import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import aiter


@triton.jit
def _moe_stage1_kernel(
    hidden,
    gate_weight,
    up_weight,
    sorted_tokens,
    sorted_experts,
    num_valid,
    intermediate,
    n_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    n_m_blocks = (tl.load(num_valid) + BLOCK_M - 1) // BLOCK_M
    n_tiles = n_m_blocks * N_BLOCKS
    rows_in_block = tl.arange(0, BLOCK_M)
    cols_in_block = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    while tile < n_tiles:
        n_block = tile // n_m_blocks
        m_block = tile - n_block * n_m_blocks
        expert = tl.load(sorted_experts + m_block)
        rows = m_block * BLOCK_M + rows_in_block
        encoded_tokens = tl.load(sorted_tokens + rows)
        tokens = encoded_tokens & 0xFFFFFF
        valid_tokens = tokens < n_tokens
        cols = n_block * BLOCK_N + cols_in_block
        valid_cols = cols < intermediate_size

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, hidden_size, BLOCK_K):
            ks = k_start + k_offsets
            activations = tl.load(
                hidden + tokens[:, None] * hidden_size + ks[None, :],
                mask=valid_tokens[:, None],
                other=0.0,
            )
            weight_offsets = (
                expert * intermediate_size * hidden_size
                + cols[None, :] * hidden_size
                + ks[:, None]
            )
            gate_tile = tl.load(
                gate_weight + weight_offsets,
                mask=valid_cols[None, :],
                other=0.0,
            )
            up_tile = tl.load(
                up_weight + weight_offsets,
                mask=valid_cols[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(activations, gate_tile)
            up_acc += tl.dot(activations, up_tile)

        gate_bf16 = gate_acc.to(tl.bfloat16)
        up_bf16 = up_acc.to(tl.bfloat16)
        silu_bf16 = (
            gate_bf16.to(tl.float32)
            * tl.sigmoid(gate_bf16.to(tl.float32))
        ).to(tl.bfloat16)
        activated = (
            silu_bf16.to(tl.float32) * up_bf16.to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(
            intermediate
            + rows[:, None] * intermediate_size
            + cols[None, :],
            activated,
            mask=valid_tokens[:, None] & valid_cols[None, :],
        )
        tile += NUM_PROGRAMS


@triton.jit
def _moe_stage2_kernel(
    intermediate,
    down_weight,
    sorted_tokens,
    sorted_experts,
    num_valid,
    expert_output,
    inverse,
    n_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    n_m_blocks = (tl.load(num_valid) + BLOCK_M - 1) // BLOCK_M
    n_tiles = n_m_blocks * N_BLOCKS
    rows_in_block = tl.arange(0, BLOCK_M)
    cols_in_block = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    while tile < n_tiles:
        n_block = tile // n_m_blocks
        m_block = tile - n_block * n_m_blocks
        expert = tl.load(sorted_experts + m_block)
        rows = m_block * BLOCK_M + rows_in_block
        encoded_tokens = tl.load(sorted_tokens + rows)
        tokens = encoded_tokens & 0xFFFFFF
        slots = encoded_tokens >> 24
        valid_tokens = tokens < n_tokens
        cols = n_block * BLOCK_N + cols_in_block
        valid_cols = cols < hidden_size

        accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, intermediate_size, BLOCK_K):
            ks = k_start + k_offsets
            activations = tl.load(
                intermediate
                + rows[:, None] * intermediate_size
                + ks[None, :],
                mask=valid_tokens[:, None],
                other=0.0,
            )
            weight_offsets = (
                expert * hidden_size * intermediate_size
                + cols[None, :] * intermediate_size
                + ks[:, None]
            )
            weight_tile = tl.load(
                down_weight + weight_offsets,
                mask=valid_cols[None, :],
                other=0.0,
            )
            accum += tl.dot(activations, weight_tile)

        tl.store(
            expert_output + rows[:, None] * hidden_size + cols[None, :],
            accum.to(tl.bfloat16),
            mask=valid_tokens[:, None] & valid_cols[None, :],
        )
        # The sorter's compact token encoding gives both the token and its
        # original top-k slot.  One N tile constructs the inverse map for the
        # final eight-way reduction.
        if n_block == 0:
            tl.store(
                inverse + tokens * TOPK + slots,
                rows,
                mask=valid_tokens,
            )
        tile += NUM_PROGRAMS


@triton.jit
def _invert_permutation_kernel(
    permutation,
    inverse,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    original_offsets = tl.load(permutation + offsets, mask=mask)
    tl.store(inverse + original_offsets, offsets, mask=mask)


@triton.jit
def _weighted_residual_kernel(
    expert_output,
    inverse,
    topk_weights,
    shared_output,
    output,
    n_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    token_mask = token_offsets < n_tokens
    hidden_mask = hidden_offsets < hidden_size
    value_mask = token_mask[:, None] & hidden_mask[None, :]

    accum = tl.zeros((BLOCK_T, BLOCK_H), tl.float32)
    for k in range(TOPK):
        route_offsets = token_offsets * TOPK + k
        sorted_offsets = tl.load(inverse + route_offsets, mask=token_mask, other=0)
        values = tl.load(
            expert_output + sorted_offsets[:, None] * hidden_size + hidden_offsets[None, :],
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            topk_weights + route_offsets, mask=token_mask, other=0.0
        ).to(tl.float32)
        accum += values * weights[:, None]

    # The routed sum is rounded to bf16 before the residual addition in the
    # reference implementation.
    routed = accum.to(tl.bfloat16)
    shared = tl.load(
        shared_output + token_offsets[:, None] * hidden_size + hidden_offsets[None, :],
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    combined = routed.to(tl.float32) + shared
    tl.store(
        output + token_offsets[:, None] * hidden_size + hidden_offsets[None, :],
        combined,
        mask=value_mask,
    )


@torch.no_grad()
def run(
    hidden_states,
    router_weight,
    e_score_correction_bias,
    expert_gate_projs,
    expert_up_projs,
    expert_down_projs,
    shared_gate_proj_weight,
    shared_up_proj_weight,
    shared_down_proj_weight,
    routed_scaling_factor,
    norm_topk_prob,
):
    n_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    topk = 8
    n_experts = 160

    # Preserve the reference's fp32 router GEMM and selection semantics.
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    scores_for_choice = torch.sigmoid(router_logits)
    scores_for_choice.add_(e_score_correction_bias)
    topk_weights, topk_indices = torch.topk(
        scores_for_choice, k=topk, dim=-1, sorted=False
    )
    if norm_topk_prob:
        topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights.mul_(routed_scaling_factor)

    # Compact the selected token/expert pairs into 128-row expert tiles.  The
    # sorter encodes (top-k slot, token) in each int32 token entry.
    block_m = 128
    num_routes = topk_indices.numel()
    max_sorted = num_routes + n_experts * block_m
    ids32 = topk_indices.to(torch.int32)
    sorted_tokens = torch.empty(
        max_sorted, device=hidden_states.device, dtype=torch.int32
    )
    sorted_weights = torch.empty(
        max_sorted, device=hidden_states.device, dtype=torch.float32
    )
    sorted_experts = torch.empty(
        triton.cdiv(max_sorted, block_m),
        device=hidden_states.device,
        dtype=torch.int32,
    )
    num_valid = torch.empty(1, device=hidden_states.device, dtype=torch.int32)
    sort_workspace = torch.empty_like(ids32)
    aiter.moe_sorting_fwd(
        ids32,
        topk_weights,
        sorted_tokens,
        sorted_weights,
        sorted_experts,
        num_valid,
        sort_workspace,
        n_experts,
        block_m,
    )

    intermediate_size = expert_gate_projs.shape[1]
    intermediate = torch.empty(
        (max_sorted, intermediate_size),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    num_programs = 512
    _moe_stage1_kernel[(num_programs,)](
        hidden_states,
        expert_gate_projs,
        expert_up_projs,
        sorted_tokens,
        sorted_experts,
        num_valid,
        intermediate,
        n_tokens,
        hidden_size,
        intermediate_size,
        N_BLOCKS=triton.cdiv(intermediate_size, 64),
        NUM_PROGRAMS=num_programs,
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=8,
        num_stages=1,
    )

    expert_output = torch.empty(
        (max_sorted, hidden_size),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    inverse = torch.empty(
        num_routes, device=hidden_states.device, dtype=torch.int32
    )
    _moe_stage2_kernel[(num_programs,)](
        intermediate,
        expert_down_projs,
        sorted_tokens,
        sorted_experts,
        num_valid,
        expert_output,
        inverse,
        n_tokens,
        hidden_size,
        intermediate_size,
        N_BLOCKS=triton.cdiv(hidden_size, 128),
        NUM_PROGRAMS=num_programs,
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=64,
        num_warps=8,
        num_stages=2,
    )

    # Shared expert.  In-place elementwise operations preserve the two bf16
    # rounding points while avoiding another intermediate allocation.
    shared_gate = F.linear(hidden_states, shared_gate_proj_weight)
    shared_up = F.linear(hidden_states, shared_up_proj_weight)
    F.silu(shared_gate, inplace=True)
    shared_gate.mul_(shared_up)
    shared_output = F.linear(shared_gate, shared_down_proj_weight)

    output = torch.empty_like(hidden_states)
    _weighted_residual_kernel[
        (triton.cdiv(n_tokens, 4), triton.cdiv(hidden_size, 256))
    ](
        expert_output,
        inverse,
        topk_weights,
        shared_output,
        output,
        n_tokens,
        hidden_size,
        TOPK=topk,
        BLOCK_T=4,
        BLOCK_H=256,
        num_warps=8,
    )
    return output
