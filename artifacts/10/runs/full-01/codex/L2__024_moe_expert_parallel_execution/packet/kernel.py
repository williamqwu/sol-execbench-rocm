import torch
import triton
import triton.language as tl

from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.moe.moe_op import fused_moe, moe_set_use_persistent_kernel


COMMON_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}

LARGE_CONFIG = {**COMMON_CONFIG, "BLOCK_SIZE_M": 256}

# A fixed 2-CU-resident grid loops over grouped-GEMM tiles. This trims launch
# and scheduling overhead for the thousands of tiles in each projection.
moe_set_use_persistent_kernel(True)


@triton.jit
def _sum_top8_kernel(inp, out, n_elements: tl.constexpr,
                     hidden_size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    token = offsets // hidden_size
    col = offsets - token * hidden_size
    base = token * (8 * hidden_size) + col
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in range(8):
        acc += tl.load(inp + base + k * hidden_size, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + offsets, acc, mask=mask)


@triton.jit
def _silu_mul_kernel(gate, up, out, n_elements: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = g / (1.0 + tl.exp2(-(g * 1.44269504089)))
    tl.store(out + offsets, silu * u, mask=mask)


@torch.no_grad()
def run(hidden_states, topk_indices, topk_weights,
        gate_proj_weights, up_proj_weights, down_proj_weights):
    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]
    intermediate_size = gate_proj_weights.shape[1]
    top_k = topk_indices.shape[1]
    num_routes = num_tokens * top_k
    config = LARGE_CONFIG if num_tokens >= 6000 else COMMON_CONFIG
    block_m = config["BLOCK_SIZE_M"]

    # The grouped GEMMs consume expert-contiguous, BLOCK_M-padded route lists.
    max_sorted = num_routes + num_experts * (block_m - 1)
    sorted_ids = torch.full(
        (max_sorted,), num_routes, dtype=torch.int32, device=hidden_states.device
    )
    expert_ids = torch.empty(
        (triton.cdiv(max_sorted, block_m),),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    num_post_pad = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
    moe_align_block_size_triton(
        topk_indices, num_experts, block_m,
        sorted_ids, expert_ids, num_post_pad,
    )

    # Project directly from the two source tensors. Packing them would copy
    # 8 GiB per invocation for these model dimensions.
    gate = torch.empty(
        (num_routes, 1, intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    up = torch.empty_like(gate)
    fused_moe(
        hidden_states, gate_proj_weights, gate,
        None, None, None,
        topk_weights, topk_indices,
        sorted_ids, expert_ids, num_post_pad,
        False, top_k, tl.bfloat16,
        False, False, False,
        config=config,
    )
    fused_moe(
        hidden_states, up_proj_weights, up,
        None, None, None,
        topk_weights, topk_indices,
        sorted_ids, expert_ids, num_post_pad,
        False, top_k, tl.bfloat16,
        False, False, False,
        config=config,
    )
    intermediate = torch.empty(
        (num_routes, intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    n_intermediate = num_routes * intermediate_size
    _silu_mul_kernel[(triton.cdiv(n_intermediate, 256),)](
        gate, up, intermediate, n_intermediate, BLOCK=256, num_warps=4,
    )

    # Treat each routed intermediate as an independent row in stage 2. The
    # original sorting is already in route-index order, so top_k=1 makes the
    # grouped kernel gather the correct intermediate row. Routing weights are
    # applied in its float accumulator before the BF16 contribution is stored.
    contributions = torch.empty(
        (num_routes, 1, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    fused_moe(
        intermediate, down_proj_weights, contributions,
        None, None, None,
        topk_weights.reshape(num_routes, 1),
        topk_indices.reshape(num_routes, 1),
        sorted_ids, expert_ids, num_post_pad,
        True, 1, tl.bfloat16,
        False, False, False,
        config=config,
    )

    output = torch.empty_like(hidden_states)
    n_elements = num_tokens * hidden_size
    _sum_top8_kernel[(triton.cdiv(n_elements, 256),)](
        contributions, output, n_elements, hidden_size, BLOCK=256,
        num_warps=4,
    )
    return output
