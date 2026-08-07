import torch
import triton
import triton.language as tl


@triton.jit
def _fill_counts(count_ptr, num_tokens: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(count_ptr + offsets, 0, mask=offsets < num_tokens)


@triton.jit
def _build_buckets(
    bucket_ptr,
    index_ptr,
    num_tokens: tl.constexpr,
    num_selected: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    selected = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = selected < num_selected
    token = tl.load(index_ptr + selected, mask=mask, other=0)
    slot = tl.atomic_add(bucket_ptr + token, 1, mask=mask)
    tl.store(
        bucket_ptr + num_tokens + token * CAPACITY + slot,
        selected,
        mask=mask & (slot < CAPACITY),
    )


@triton.jit
def _gather_buckets(
    output_ptr,
    input_ptr,
    expert_ptr,
    index_ptr,
    bucket_ptr,
    num_tokens: tl.constexpr,
    num_selected: tl.constexpr,
    hidden_size: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    block_h = tl.program_id(0)
    token = tl.program_id(1)
    cols = block_h * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.load(input_ptr + token * hidden_size + cols).to(tl.float32)
    count = tl.load(bucket_ptr + token)
    slot = 0
    while (slot + 1 < count) & (count <= CAPACITY):
        selected0 = tl.load(bucket_ptr + num_tokens + token * CAPACITY + slot)
        selected1 = tl.load(bucket_ptr + num_tokens + token * CAPACITY + slot + 1)
        values0 = tl.load(expert_ptr + selected0 * hidden_size + cols)
        values1 = tl.load(expert_ptr + selected1 * hidden_size + cols)
        acc += values0 + values1
        slot += 2
    while (slot < count) & (count <= CAPACITY):
        selected = tl.load(bucket_ptr + num_tokens + token * CAPACITY + slot)
        values = tl.load(expert_ptr + selected * hidden_size + cols)
        acc += values
        slot += 1

    selected = 0
    while (selected < num_selected) & (count > CAPACITY):
        selected_token = tl.load(index_ptr + selected)
        values = tl.load(
            expert_ptr + selected * hidden_size + cols,
            mask=selected_token == token,
            other=0.0,
        )
        acc += values
        selected += 1
    tl.store(output_ptr + token * hidden_size + cols, acc)


@triton.jit
def _gather_token_1024(
    output_ptr,
    input_ptr,
    expert_ptr,
    index_ptr,
    bucket_ptr,
    num_tokens: tl.constexpr,
    num_selected: tl.constexpr,
    hidden_size: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PAIR_ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    input_row = input_ptr + token * hidden_size + cols

    rows_base = num_tokens + token * CAPACITY

    acc0 = tl.load(input_row + 0 * BLOCK_H).to(tl.float32)
    acc1 = tl.load(input_row + 1 * BLOCK_H).to(tl.float32)
    acc2 = tl.load(input_row + 2 * BLOCK_H).to(tl.float32)

    count = tl.load(bucket_ptr + token)
    slot = 0
    if PAIR_ROWS:
        while (slot + 1 < count) & (count <= CAPACITY):
            selected0 = tl.load(bucket_ptr + rows_base + slot)
            selected1 = tl.load(bucket_ptr + rows_base + slot + 1)
            expert_row0 = expert_ptr + selected0 * hidden_size + cols
            expert_row1 = expert_ptr + selected1 * hidden_size + cols
            acc0 += tl.load(expert_row0 + 0 * BLOCK_H) + tl.load(
                expert_row1 + 0 * BLOCK_H
            )
            acc1 += tl.load(expert_row0 + 1 * BLOCK_H) + tl.load(
                expert_row1 + 1 * BLOCK_H
            )
            acc2 += tl.load(expert_row0 + 2 * BLOCK_H) + tl.load(
                expert_row1 + 2 * BLOCK_H
            )
            slot += 2
    while (slot < count) & (count <= CAPACITY):
        selected = tl.load(bucket_ptr + rows_base + slot)
        expert_row = expert_ptr + selected * hidden_size + cols
        acc0 += tl.load(expert_row + 0 * BLOCK_H)
        acc1 += tl.load(expert_row + 1 * BLOCK_H)
        acc2 += tl.load(expert_row + 2 * BLOCK_H)
        slot += 1

    selected = 0
    while (selected < num_selected) & (count > CAPACITY):
        selected_token = tl.load(index_ptr + selected)
        expert_row = expert_ptr + selected * hidden_size + cols
        mask = selected_token == token
        acc0 += tl.load(expert_row + 0 * BLOCK_H, mask=mask, other=0.0)
        acc1 += tl.load(expert_row + 1 * BLOCK_H, mask=mask, other=0.0)
        acc2 += tl.load(expert_row + 2 * BLOCK_H, mask=mask, other=0.0)
        selected += 1

    output_row = output_ptr + token * hidden_size + cols
    tl.store(output_row + 0 * BLOCK_H, acc0)
    tl.store(output_row + 1 * BLOCK_H, acc1)
    tl.store(output_row + 2 * BLOCK_H, acc2)


@triton.jit
def _gather_token_512(
    output_ptr,
    input_ptr,
    expert_ptr,
    bucket_ptr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    input_row = input_ptr + token * hidden_size + cols

    acc0 = tl.load(input_row + 0 * BLOCK_H).to(tl.float32)
    acc1 = tl.load(input_row + 1 * BLOCK_H).to(tl.float32)
    acc2 = tl.load(input_row + 2 * BLOCK_H).to(tl.float32)
    acc3 = tl.load(input_row + 3 * BLOCK_H).to(tl.float32)
    acc4 = tl.load(input_row + 4 * BLOCK_H).to(tl.float32)
    acc5 = tl.load(input_row + 5 * BLOCK_H).to(tl.float32)

    count = tl.load(bucket_ptr + token)
    slot = 0
    while slot < count:
        selected = tl.load(bucket_ptr + num_tokens + token * CAPACITY + slot)
        expert_row = expert_ptr + selected * hidden_size + cols
        acc0 += tl.load(expert_row + 0 * BLOCK_H)
        acc1 += tl.load(expert_row + 1 * BLOCK_H)
        acc2 += tl.load(expert_row + 2 * BLOCK_H)
        acc3 += tl.load(expert_row + 3 * BLOCK_H)
        acc4 += tl.load(expert_row + 4 * BLOCK_H)
        acc5 += tl.load(expert_row + 5 * BLOCK_H)
        slot += 1

    output_row = output_ptr + token * hidden_size + cols
    tl.store(output_row + 0 * BLOCK_H, acc0)
    tl.store(output_row + 1 * BLOCK_H, acc1)
    tl.store(output_row + 2 * BLOCK_H, acc2)
    tl.store(output_row + 3 * BLOCK_H, acc3)
    tl.store(output_row + 4 * BLOCK_H, acc4)
    tl.store(output_row + 5 * BLOCK_H, acc5)


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    output = torch.empty_like(final_hidden_states)
    num_tokens = final_hidden_states.shape[0]
    num_selected = expert_outputs.shape[0]
    hidden_size = expert_outputs.shape[1]

    capacity = 32
    buckets = torch.empty(
        num_tokens * (capacity + 1),
        dtype=torch.int32,
        device=final_hidden_states.device,
    )
    block = 256 if num_tokens <= 1024 else 1024
    _fill_counts[(triton.cdiv(num_tokens, block),)](
        buckets, num_tokens=num_tokens, BLOCK=block, num_warps=4
    )
    _build_buckets[(triton.cdiv(num_selected, block),)](
        buckets,
        token_indices,
        num_tokens=num_tokens,
        num_selected=num_selected,
        CAPACITY=capacity,
        BLOCK=block,
        num_warps=4,
    )

    if num_tokens <= 256:
        block_h = 1024
        _gather_buckets[(triton.cdiv(hidden_size, block_h), num_tokens)](
            output,
            final_hidden_states,
            expert_outputs,
            token_indices,
            buckets,
            num_tokens=num_tokens,
            num_selected=num_selected,
            hidden_size=hidden_size,
            CAPACITY=capacity,
            BLOCK_H=block_h,
            num_warps=8,
        )
    else:
        block_h = 1024
        warps = 4 if 1024 < num_tokens < 4096 else 8
        _gather_token_1024[(num_tokens,)](
            output,
            final_hidden_states,
            expert_outputs,
            token_indices,
            buckets,
            num_tokens=num_tokens,
            num_selected=num_selected,
            hidden_size=hidden_size,
            CAPACITY=capacity,
            BLOCK_H=block_h,
            PAIR_ROWS=num_tokens < 8192,
            num_warps=warps,
        )
    return output
