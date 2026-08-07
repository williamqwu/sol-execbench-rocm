import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from aiter.ops.triton.moe.moe_align_block_size import (
    moe_align_block_size_triton,
)
from aiter.ops.triton.moe.moe_op import fused_moe
from aiter import moe_align_block_size
from aiter.fused_moe import moe_sorting


_MOE_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 32,
    "kpack": 1,
}


@triton.jit
def _route_top8_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    sorting_weights_ptr,
    indices_ptr,
    scale,
    NUM_EXPERTS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, NUM_EXPERTS)
    logits = tl.load(logits_ptr + row * NUM_EXPERTS + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols)
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias

    out_cols = tl.arange(0, 8)
    values = tl.zeros((8,), dtype=tl.float32)
    expert_ids = tl.zeros((8,), dtype=tl.int32)
    for k in tl.static_range(8):
        value = tl.max(scores, axis=0)
        expert = tl.argmax(scores, axis=0)
        values = tl.where(out_cols == k, value, values)
        expert_ids = tl.where(out_cols == k, expert, expert_ids)
        scores = tl.where(cols == expert, -float("inf"), scores)

    values = values / (tl.sum(values, axis=0) + 1.0e-20)
    values = values * scale
    tl.store(weights_ptr + row * 8 + out_cols, values)
    tl.store(sorting_weights_ptr + row * 8 + out_cols, values)
    tl.store(indices_ptr + row * 8 + out_cols, expert_ids)


def _route_top8(logits, bias, scale):
    rows = logits.shape[0]
    weights = torch.empty(
        (rows, 8), dtype=torch.bfloat16, device=logits.device
    )
    sorting_weights = torch.empty(
        (rows, 8), dtype=torch.float32, device=logits.device
    )
    indices = torch.empty(
        (rows, 8), dtype=torch.int32, device=logits.device
    )
    _route_top8_kernel[(rows,)](
        logits,
        bias,
        weights,
        sorting_weights,
        indices,
        scale,
        NUM_EXPERTS=128,
        num_warps=4,
    )
    return weights, sorting_weights, indices


@triton.jit
def _decode_sorted_ids_kernel(
    sorted_ids_ptr,
    num_ids,
    num_tokens,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_ids
    encoded = tl.load(sorted_ids_ptr + offsets, mask=mask)
    token = encoded & 0x00FFFFFF
    slot = encoded >> 24
    route = token * TOPK + slot
    route = tl.where(token < num_tokens, route, num_tokens * TOPK)
    tl.store(sorted_ids_ptr + offsets, route, mask=mask)


def _decode_sorted_ids_(sorted_ids, num_tokens):
    num_ids = sorted_ids.numel()
    _decode_sorted_ids_kernel[(triton.cdiv(num_ids, 256),)](
        sorted_ids,
        num_ids,
        num_tokens,
        TOPK=8,
        BLOCK=256,
        num_warps=4,
    )


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    result = (activated.to(tl.float32) * up).to(tl.bfloat16)
    tl.store(gate_ptr + offsets, result, mask=mask)


def _silu_mul_(gate, up):
    n_elements = gate.numel()
    _silu_mul_kernel[(triton.cdiv(n_elements, 1024),)](
        gate, up, n_elements, BLOCK=1024, num_warps=8
    )


@triton.jit
def _weighted_reduce_add_kernel(
    routes_ptr,
    weights_ptr,
    shared_ptr,
    output_ptr,
    n_elements,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tokens = offsets // HIDDEN
    cols = offsets - tokens * HIDDEN
    route_base = tokens * (8 * HIDDEN) + cols
    weight_base = tokens * 8
    routed = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(8):
        value = tl.load(
            routes_ptr + route_base + k * HIDDEN, mask=mask
        ).to(tl.float32)
        weight = tl.load(weights_ptr + weight_base + k, mask=mask).to(
            tl.float32
        )
        weighted = (value * weight).to(tl.bfloat16)
        routed += weighted.to(tl.float32)
    routed = routed.to(tl.bfloat16)
    shared = tl.load(shared_ptr + offsets, mask=mask).to(tl.float32)
    result = (routed.to(tl.float32) + shared).to(tl.bfloat16)
    tl.store(output_ptr + offsets, result, mask=mask)


def _weighted_reduce_add(routes, weights, shared):
    output = torch.empty_like(shared)
    n_elements = shared.numel()
    _weighted_reduce_add_kernel[(triton.cdiv(n_elements, 512),)](
        routes,
        weights,
        shared,
        output,
        n_elements,
        HIDDEN=4096,
        BLOCK=512,
        num_warps=8,
    )
    return output


@torch.no_grad()
def run(
    hidden_states,
    router_weight,
    expert_gate_weights,
    expert_up_weights,
    expert_down_weights,
    shared_gate_weight,
    shared_up_weight,
    shared_down_weight,
    e_score_correction_bias,
    routed_scaling_factor,
):
    router_logits = F.linear(hidden_states, router_weight)
    topk_weights, sorting_weights, topk_indices = _route_top8(
        router_logits, e_score_correction_bias, routed_scaling_factor
    )

    # Align the ragged routes into expert-homogeneous blocks.  The sentinel
    # initialization is required for the unused lanes in each block.
    num_tokens = hidden_states.shape[0]
    num_routes = num_tokens * 8
    block_m = 128
    (
        sorted_token_ids,
        unused_sorted_weights,
        expert_ids,
        num_tokens_post_padded,
        unused_workspace,
    ) = moe_sorting(
        topk_indices,
        sorting_weights,
        128,
        4096,
        torch.bfloat16,
        block_size=block_m,
    )
    _decode_sorted_ids_(sorted_token_ids, num_tokens)

    gate = torch.empty(
        (num_tokens, 8, 1408),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    up = torch.empty_like(gate)
    common = (
        None,
        None,
        None,
        topk_weights,
        topk_indices,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )

    fused_moe(
        hidden_states,
        expert_gate_weights,
        gate,
        *common,
        False,
        8,
        tl.bfloat16,
        False,
        False,
        False,
        config=_MOE_CONFIG,
    )
    fused_moe(
        hidden_states,
        expert_up_weights,
        up,
        *common,
        False,
        8,
        tl.bfloat16,
        False,
        False,
        False,
        config=_MOE_CONFIG,
    )
    _silu_mul_(gate, up)

    expert_output = torch.empty(
        (num_routes, 1, 4096),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    fused_moe(
        gate.reshape(num_routes, 1408),
        expert_down_weights,
        expert_output,
        *common,
        False,
        1,
        tl.bfloat16,
        False,
        False,
        False,
        config=_MOE_CONFIG,
    )
    expert_output = expert_output.view(num_tokens, 8, 4096)

    shared_gate = F.linear(hidden_states, shared_gate_weight)
    shared_up = F.linear(hidden_states, shared_up_weight)
    _silu_mul_(shared_gate, shared_up)
    shared_output = F.linear(shared_gate, shared_down_weight)
    return _weighted_reduce_add(
        expert_output, topk_weights, shared_output
    )
