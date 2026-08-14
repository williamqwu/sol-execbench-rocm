import torch
import triton
import triton.language as tl


@triton.jit
def _build_routes(
    selected_ptr,
    token_map_ptr,
    assignment_pos_ptr,
    num_tokens,
    CAPACITY: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Build stable, capacity-limited routes one expert per program."""
    expert = tl.program_id(0)

    # Mark padding rows.  The same program later overwrites admitted rows, so
    # no separate memset or synchronization is needed.
    c = tl.arange(0, BLOCK_C)
    tl.store(token_map_ptr + expert * CAPACITY + c, -1, mask=c < CAPACITY)

    base = tl.zeros((1,), tl.int32)
    for start in tl.range(0, num_tokens, BLOCK_T, loop_unroll_factor=1):
        tok = start + tl.arange(0, BLOCK_T)
        live = tok < num_tokens
        off = tok * 8

        e0 = tl.load(selected_ptr + off + 0, mask=live, other=-1)
        e1 = tl.load(selected_ptr + off + 1, mask=live, other=-1)
        e2 = tl.load(selected_ptr + off + 2, mask=live, other=-1)
        e3 = tl.load(selected_ptr + off + 3, mask=live, other=-1)
        e4 = tl.load(selected_ptr + off + 4, mask=live, other=-1)
        e5 = tl.load(selected_ptr + off + 5, mask=live, other=-1)
        e6 = tl.load(selected_ptr + off + 6, mask=live, other=-1)
        e7 = tl.load(selected_ptr + off + 7, mask=live, other=-1)

        m0 = live & (e0 == expert)
        m1 = live & (e1 == expert)
        m2 = live & (e2 == expert)
        m3 = live & (e3 == expert)
        m4 = live & (e4 == expert)
        m5 = live & (e5 == expert)
        m6 = live & (e6 == expert)
        m7 = live & (e7 == expert)
        hit = (m0 | m1) | (m2 | m3) | (m4 | m5) | (m6 | m7)

        inclusive = tl.cumsum(hit.to(tl.int32), axis=0)
        pos = base + inclusive - 1
        admitted = hit & (pos < CAPACITY)
        tl.store(
            token_map_ptr + expert * CAPACITY + pos,
            tok,
            mask=admitted,
        )

        stored_pos = tl.where(pos < CAPACITY, pos, -1)
        tl.store(assignment_pos_ptr + off + 0, stored_pos, mask=m0)
        tl.store(assignment_pos_ptr + off + 1, stored_pos, mask=m1)
        tl.store(assignment_pos_ptr + off + 2, stored_pos, mask=m2)
        tl.store(assignment_pos_ptr + off + 3, stored_pos, mask=m3)
        tl.store(assignment_pos_ptr + off + 4, stored_pos, mask=m4)
        tl.store(assignment_pos_ptr + off + 5, stored_pos, mask=m5)
        tl.store(assignment_pos_ptr + off + 6, stored_pos, mask=m6)
        tl.store(assignment_pos_ptr + off + 7, stored_pos, mask=m7)
        base += tl.sum(hit.to(tl.int32), axis=0)


@triton.jit
def _gather_expert_inputs(
    hidden_ptr,
    token_map_ptr,
    expert_input_ptr,
    H: tl.constexpr,
    ROUTE_CAPACITY: tl.constexpr,
    OUTPUT_CAPACITY: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    group = tl.program_id(0)
    expert = group // M_BLOCKS
    mb = group - expert * M_BLOCKS
    m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    h = tl.arange(0, BLOCK_H) + tl.program_id(1) * BLOCK_H
    tok = tl.load(
        token_map_ptr + expert * ROUTE_CAPACITY + m,
        mask=m < ROUTE_CAPACITY,
        other=-1,
    )
    mask = (m[:, None] < OUTPUT_CAPACITY) & (h[None, :] < H)
    value = tl.load(
        hidden_ptr + tok[:, None] * H + h[None, :],
        mask=mask & (tok[:, None] >= 0),
        other=0.0,
    )
    tl.store(
        expert_input_ptr + (expert * OUTPUT_CAPACITY + m[:, None]) * H + h[None, :],
        value,
        mask=mask,
    )


@triton.jit
def _aggregate(
    expert_output_ptr,
    selected_ptr,
    routing_weight_ptr,
    assignment_pos_ptr,
    result_ptr,
    num_tokens,
    H: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    live = tok < num_tokens
    acc = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
    base = tok * 8

    for k in range(8):
        expert = tl.load(selected_ptr + base + k, mask=live, other=0)
        pos = tl.load(assignment_pos_ptr + base + k, mask=live, other=-1)
        weight = tl.load(routing_weight_ptr + base + k, mask=live, other=0.0)
        value = tl.load(
            expert_output_ptr
            + (expert[:, None] * CAPACITY + pos[:, None]) * H
            + h[None, :],
            mask=live[:, None] & (pos[:, None] >= 0) & (h[None, :] < H),
            other=0.0,
        )
        # PyTorch forms this product in bfloat16 before index_add_.
        weighted = (value * weight[:, None]).to(tl.bfloat16)
        acc += weighted.to(tl.float32)

    tl.store(
        result_ptr + tok[:, None] * H + h[None, :],
        acc,
        mask=live[:, None] & (h[None, :] < H),
    )


@triton.jit
def _swiglu_inplace(
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elements
    gate = tl.load(gate_ptr + off, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + off, mask=mask)
    # F.silu(bf16) rounds before the in-place bf16 multiplication.
    silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
    activated = (silu * up).to(tl.bfloat16)
    tl.store(gate_ptr + off, activated, mask=mask)


@torch.no_grad()
def run(
    hidden_states,
    selected_experts,
    routing_weights,
    expert_gate_weights,
    expert_up_weights,
    expert_down_weights,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts = expert_gate_weights.shape[0]
    capacity = max(int((num_tokens * 8 / num_experts) * 1.25), 1)
    # rocBLAS has sharp algorithm boundaries in its M dimension.  These
    # zero-padded sizes are faster than the immediately smaller exact sizes.
    if capacity == 80:
        compute_capacity = 82
    elif capacity == 88:
        compute_capacity = 94
    elif capacity == 193:
        compute_capacity = 224
    else:
        compute_capacity = capacity
    device = hidden_states.device

    token_map = torch.empty((num_experts, capacity), dtype=torch.int32, device=device)
    assignment_pos = torch.empty((num_tokens, 8), dtype=torch.int32, device=device)
    expert_inputs = torch.empty(
        (num_experts, compute_capacity, hidden_size),
        dtype=hidden_states.dtype,
        device=device,
    )

    block_c = triton.next_power_of_2(capacity)
    _build_routes[(num_experts,)](
        selected_experts,
        token_map,
        assignment_pos,
        num_tokens,
        CAPACITY=capacity,
        BLOCK_T=1024,
        BLOCK_C=block_c,
        num_warps=8,
    )

    block_m = 16
    block_h = 512
    m_blocks = triton.cdiv(compute_capacity, block_m)
    _gather_expert_inputs[(num_experts * m_blocks, triton.cdiv(hidden_size, block_h))](
        hidden_states,
        token_map,
        expert_inputs,
        H=hidden_size,
        ROUTE_CAPACITY=capacity,
        OUTPUT_CAPACITY=compute_capacity,
        M_BLOCKS=m_blocks,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=8,
    )

    gate_out = torch.bmm(expert_inputs, expert_gate_weights)
    up_out = torch.bmm(expert_inputs, expert_up_weights)
    activation_elements = gate_out.numel()
    _swiglu_inplace[(triton.cdiv(activation_elements, 2048),)](
        gate_out,
        up_out,
        activation_elements,
        BLOCK=2048,
        num_warps=4,
    )
    down_capacity = compute_capacity
    expert_outputs = torch.bmm(gate_out, expert_down_weights)

    result = torch.empty_like(hidden_states)
    agg_m = 16
    agg_h = 256
    _aggregate[(triton.cdiv(num_tokens, agg_m), triton.cdiv(hidden_size, agg_h))](
        expert_outputs,
        selected_experts,
        routing_weights,
        assignment_pos,
        result,
        num_tokens,
        H=hidden_size,
        CAPACITY=down_capacity,
        BLOCK_M=agg_m,
        BLOCK_H=agg_h,
        num_warps=4,
    )
    return result
