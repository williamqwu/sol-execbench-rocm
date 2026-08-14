import torch
import triton
import triton.language as tl

from aiter.fused_moe import fused_moe
from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton


@triton.jit
def _pack_weights(
    gate_ptr,
    up_ptr,
    down_ptr,
    gate_up_out_ptr,
    down_out_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_ORDER: tl.constexpr,
):
    """Directly form [gate, up] and CK's 16x16 tiled weight layout."""
    pid = tl.program_id(0)
    E: tl.constexpr = 256
    I: tl.constexpr = 1024
    GU_N: tl.constexpr = 2048
    GU_K: tl.constexpr = 4096
    D_N: tl.constexpr = 4096
    D_K: tl.constexpr = 1024
    GU_NT: tl.constexpr = GU_N // BLOCK_N
    GU_KT: tl.constexpr = GU_K // BLOCK_K
    GU_PROGRAMS: tl.constexpr = E * GU_NT * GU_KT

    linear = tl.arange(0, BLOCK_N * BLOCK_K)
    if OUTPUT_ORDER:
        rem = linear
        local_k_lane = rem % 8
        rem = rem // 8
        local_n_lane = rem % 16
        rem = rem // 16
        local_k_group = rem % 4
        rem = rem // 4
        local_k_block = rem % (BLOCK_K // 32)
        local_n_block = rem // (BLOCK_K // 32)
        local_n = local_n_block * 16 + local_n_lane
        local_k = local_k_block * 32 + local_k_group * 8 + local_k_lane
    else:
        local_n = linear // BLOCK_K
        local_k = linear % BLOCK_K

    if pid < GU_PROGRAMS:
        expert = pid // (GU_NT * GU_KT)
        tile = pid % (GU_NT * GU_KT)
        tile_n = tile // GU_KT
        tile_k = tile % GU_KT
        n = tile_n * BLOCK_N + local_n
        k = tile_k * BLOCK_K + local_k

        is_up = n >= I
        source_n = tl.where(is_up, n - I, n)
        source_base = tl.where(is_up, up_ptr, gate_ptr)
        value = tl.load(source_base + expert * I * GU_K + source_n * GU_K + k)

        n_block = n // 16
        n_lane = n % 16
        k_block = k // 32
        k_group = (k % 32) // 8
        k_lane = k % 8
        output_offset = (
            (((((expert * (GU_N // 16) + n_block) * (GU_K // 32) + k_block)
               * 4 + k_group) * 16 + n_lane) * 8 + k_lane)
        )
        tl.store(gate_up_out_ptr + output_offset, value)
    else:
        down_pid = pid - GU_PROGRAMS
        D_NT: tl.constexpr = D_N // BLOCK_N
        D_KT: tl.constexpr = D_K // BLOCK_K
        expert = down_pid // (D_NT * D_KT)
        tile = down_pid % (D_NT * D_KT)
        tile_n = tile // D_KT
        tile_k = tile % D_KT
        n = tile_n * BLOCK_N + local_n
        k = tile_k * BLOCK_K + local_k
        value = tl.load(down_ptr + expert * D_N * D_K + n * D_K + k)

        n_block = n // 16
        n_lane = n % 16
        k_block = k // 32
        k_group = (k % 32) // 8
        k_lane = k % 8
        output_offset = (
            (((((expert * (D_N // 16) + n_block) * (D_K // 32) + k_block)
               * 4 + k_group) * 16 + n_lane) * 8 + k_lane)
        )
        tl.store(down_out_ptr + output_offset, value)


def _pack(
    gate_weights,
    up_weights,
    down_weights,
    block_n=16,
    block_k=512,
    num_warps=8,
    output_order=True,
):
    gate_up = torch.empty(
        (256, 2048, 4096), dtype=gate_weights.dtype, device=gate_weights.device
    )
    down = torch.empty_like(down_weights)
    programs = (
        256 * (2048 // block_n) * (4096 // block_k)
        + 256 * (4096 // block_n) * (1024 // block_k)
    )
    _pack_weights[(programs,)](
        gate_weights,
        up_weights,
        down_weights,
        gate_up,
        down,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        OUTPUT_ORDER=output_order,
        num_warps=num_warps,
    )
    gate_up.is_shuffled = True
    down.is_shuffled = True
    return gate_up, down


@triton.jit
def _raw_moe_kernel(
    x_ptr,
    gate_ptr,
    up_ptr,
    down_ptr,
    route_weight_ptr,
    sorted_route_ptr,
    sorted_expert_ptr,
    padded_count_ptr,
    intermediate_ptr,
    route_output_ptr,
    valid_routes: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_O: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    H: tl.constexpr = 4096
    I: tl.constexpr = 1024
    TOP_K: tl.constexpr = 8

    start_block = tl.program_id(0)
    padded_count = tl.load(padded_count_ptr)
    num_blocks = tl.cdiv(padded_count, BLOCK_M)
    blocks_for_program = num_blocks // NUM_PROGRAMS
    if start_block < num_blocks % NUM_PROGRAMS:
        blocks_for_program += 1

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_i = tl.arange(0, BLOCK_I)
    offs_o = tl.arange(0, BLOCK_O)
    block = start_block

    for _ in range(blocks_for_program):
        sorted_pos = block * BLOCK_M + offs_m
        route = tl.load(sorted_route_ptr + sorted_pos)
        route_mask = route < valid_routes
        token = route // TOP_K
        expert = tl.load(sorted_expert_ptr + block)

        # First projection.  Gate and up remain in their supplied contiguous
        # layouts, so there is no multi-GiB preprocessing pass.
        for tile_i in range(I // BLOCK_I):
            a_ptrs = x_ptr + token[:, None] * H + offs_k[None, :]
            row_i = tile_i * BLOCK_I + offs_i
            gate_ptrs = (
                gate_ptr
                + expert * I * H
                + offs_k[:, None]
                + row_i[None, :] * H
            )
            up_ptrs = (
                up_ptr
                + expert * I * H
                + offs_k[:, None]
                + row_i[None, :] * H
            )
            gate_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
            for _k in range(H // BLOCK_K):
                a = tl.load(a_ptrs, mask=route_mask[:, None], other=0.0)
                gate = tl.load(gate_ptrs)
                up = tl.load(up_ptrs)
                gate_acc = tl.dot(a, gate, acc=gate_acc)
                up_acc = tl.dot(a, up, acc=up_acc)
                a_ptrs += BLOCK_K
                gate_ptrs += BLOCK_K
                up_ptrs += BLOCK_K

            silu = gate_acc / (1.0 + tl.exp2(-(gate_acc * 1.44269504089)))
            activated = (silu * up_acc).to(tl.bfloat16)
            inter_ptrs = intermediate_ptr + route[:, None] * I + row_i[None, :]
            tl.store(inter_ptrs, activated, mask=route_mask[:, None])

        tl.debug_barrier()

        # Down projection, producing one BF16 result for every routed pair.
        for tile_o in range(H // BLOCK_O):
            output_col = tile_o * BLOCK_O + offs_o
            inter_ptrs = intermediate_ptr + route[:, None] * I + offs_i[None, :]
            down_ptrs = (
                down_ptr
                + expert * H * I
                + offs_i[:, None]
                + output_col[None, :] * I
            )
            output_acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)
            for _i in range(I // BLOCK_I):
                activated = tl.load(
                    inter_ptrs, mask=route_mask[:, None], other=0.0
                )
                down = tl.load(down_ptrs)
                output_acc = tl.dot(activated, down, acc=output_acc)
                inter_ptrs += BLOCK_I
                down_ptrs += BLOCK_I

            route_weight = tl.load(
                route_weight_ptr + route, mask=route_mask, other=0.0
            ).to(tl.float32)
            output_acc *= route_weight[:, None]
            out_ptrs = route_output_ptr + route[:, None] * H + output_col[None, :]
            tl.store(out_ptrs, output_acc.to(tl.bfloat16), mask=route_mask[:, None])

        block += NUM_PROGRAMS


def _run_raw(
    hidden_states,
    topk_idx,
    topk_weight,
    gate_weights,
    up_weights,
    down_weights,
):
    block_m = 64
    num_tokens = hidden_states.shape[0]
    valid_routes = topk_idx.numel()
    max_padded = valid_routes + 256 * (block_m - 1)
    sorted_routes = torch.full(
        (max_padded,), valid_routes, dtype=torch.int32, device=hidden_states.device
    )
    sorted_experts = torch.empty(
        (triton.cdiv(max_padded, block_m),),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    padded_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
    moe_align_block_size_triton(
        topk_idx,
        256,
        block_m,
        sorted_routes,
        sorted_experts,
        padded_count,
    )

    intermediate = torch.empty(
        (valid_routes, 1024), dtype=torch.bfloat16, device=hidden_states.device
    )
    route_output = torch.empty(
        (valid_routes, 4096), dtype=torch.bfloat16, device=hidden_states.device
    )
    _raw_moe_kernel[(512,)](
        hidden_states,
        gate_weights,
        up_weights,
        down_weights,
        topk_weight,
        sorted_routes,
        sorted_experts,
        padded_count,
        intermediate,
        route_output,
        valid_routes=valid_routes,
        BLOCK_M=block_m,
        BLOCK_I=64,
        BLOCK_K=64,
        BLOCK_O=64,
        NUM_PROGRAMS=512,
        num_warps=4,
    )
    return route_output.view(num_tokens, 8, 4096).sum(dim=1)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    gate_up, down = _pack(gate_weights, up_weights, down_weights)

    # The routing kernel uses i32 expert IDs and f32 routing weights.  The two
    # grouped GEMMs accumulate MFMA products in f32 and return the weighted
    # expert reduction directly in BF16.
    return fused_moe(
        hidden_states,
        gate_up,
        down,
        topk_weight.float(),
        topk_idx.int(),
    )
