import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    sm_scale,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // 32
    q_head = pid - batch * 32
    kv_head = q_head // 8

    token_begin = tl.load(indptr_ptr + batch)
    token_end = tl.load(indptr_ptr + batch + 1)
    token_count = token_end - token_begin

    d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + (batch * 32 + q_head) * HEAD_DIM + d).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)

    token_base = 0
    while token_base < token_count:
        n = token_base + tl.arange(0, BLOCK_N)
        valid = n < token_count
        page = tl.load(indices_ptr + token_begin + n, mask=valid, other=0)

        k_offsets = page[:, None] * (4 * HEAD_DIM) + kv_head * HEAD_DIM + d[None, :]
        k = tl.load(k_ptr + k_offsets, mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(valid, scores, -float("inf"))

        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)

        v_offsets = page[:, None] * (4 * HEAD_DIM) + kv_head * HEAD_DIM + d[None, :]
        v = tl.load(v_ptr + v_offsets, mask=valid[:, None], other=0.0).to(tl.float32)
        accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * v, axis=0)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max
        token_base += BLOCK_N

    nonempty = token_count > 0
    result = tl.where(nonempty, accumulator / running_sum, 0.0)
    logsumexp2 = tl.where(
        nonempty,
        (running_max + tl.log(running_sum)) * 1.4426950408889634,
        -float("inf"),
    )
    tl.store(out_ptr + (batch * 32 + q_head) * HEAD_DIM + d, result)
    tl.store(lse_ptr + batch * 32 + q_head, logsumexp2)


@triton.jit
def _paged_decode_grouped(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    sm_scale,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_H: tl.constexpr,
    USE_BF16_P: tl.constexpr,
):
    pid = tl.program_id(0)
    groups_per_kv = 8 // GROUP_H
    groups_per_batch = 4 * groups_per_kv
    batch = pid // groups_per_batch
    local_group = pid - batch * groups_per_batch
    kv_head = local_group // groups_per_kv
    head_base = (local_group - kv_head * groups_per_kv) * GROUP_H

    token_begin = tl.load(indptr_ptr + batch)
    token_end = tl.load(indptr_ptr + batch + 1)
    token_count = token_end - token_begin

    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, HEAD_DIM)
    q_head = kv_head * 8 + head_base + h
    q_offsets = (batch * 32 + q_head[:, None]) * HEAD_DIM + d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=h[:, None] < GROUP_H, other=0.0)

    running_max = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, HEAD_DIM), tl.float32)

    token_base = 0
    while token_base < token_count:
        n = token_base + tl.arange(0, BLOCK_N)
        valid = n < token_count
        page = tl.load(indices_ptr + token_begin + n, mask=valid, other=0)

        kv_offsets = page[:, None] * (4 * HEAD_DIM) + kv_head * HEAD_DIM + d[None, :]
        k = tl.load(k_ptr + kv_offsets, mask=valid[:, None], other=0.0)
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])

        v = tl.load(v_ptr + kv_offsets, mask=valid[:, None], other=0.0)
        if USE_BF16_P:
            value_update = tl.dot(
                probabilities.to(tl.bfloat16), v, out_dtype=tl.float32
            )
        else:
            value_update = tl.dot(
                probabilities,
                v.to(tl.float32),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        accumulator = accumulator * old_scale[:, None] + value_update
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = new_max
        token_base += BLOCK_N

    nonempty = token_count > 0
    result = tl.where(nonempty, accumulator / running_sum[:, None], 0.0)
    logsumexp2 = tl.where(
        nonempty,
        (running_max + tl.log(running_sum)) * 1.4426950408889634,
        -float("inf"),
    )
    tl.store(out_ptr + q_offsets, result, mask=h[:, None] < GROUP_H)
    tl.store(lse_ptr + batch * 32 + q_head, logsumexp2, mask=h < GROUP_H)


@triton.jit
def _paged_decode_split(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    partial_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    sm_scale,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_H: tl.constexpr,
    SPLITS: tl.constexpr,
):
    group_id = tl.program_id(0)
    split = tl.program_id(1)
    groups_per_kv = 8 // GROUP_H
    groups_per_batch = 4 * groups_per_kv
    batch = group_id // groups_per_batch
    local_group = group_id - batch * groups_per_batch
    kv_head = local_group // groups_per_kv
    head_base = (local_group - kv_head * groups_per_kv) * GROUP_H

    sequence_begin = tl.load(indptr_ptr + batch)
    sequence_end = tl.load(indptr_ptr + batch + 1)
    token_count = sequence_end - sequence_begin
    tile_count = (token_count + BLOCK_N - 1) // BLOCK_N
    tiles_per_split = (tile_count + SPLITS - 1) // SPLITS
    split_begin = split * tiles_per_split * BLOCK_N
    split_end = tl.minimum(split_begin + tiles_per_split * BLOCK_N, token_count)

    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, HEAD_DIM)
    q_head = kv_head * 8 + head_base + h
    q_offsets = (batch * 32 + q_head[:, None]) * HEAD_DIM + d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=h[:, None] < GROUP_H, other=0.0)

    running_max = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, HEAD_DIM), tl.float32)

    token_base = split_begin
    while token_base < split_end:
        n = token_base + tl.arange(0, BLOCK_N)
        valid = n < split_end
        page = tl.load(indices_ptr + sequence_begin + n, mask=valid, other=0)
        kv_offsets = page[:, None] * (4 * HEAD_DIM) + kv_head * HEAD_DIM + d[None, :]
        k = tl.load(k_ptr + kv_offsets, mask=valid[:, None], other=0.0)
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        v = tl.load(v_ptr + kv_offsets, mask=valid[:, None], other=0.0)
        value_update = tl.dot(
            probabilities.to(tl.bfloat16), v, out_dtype=tl.float32
        )
        accumulator = accumulator * old_scale[:, None] + value_update
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = new_max
        token_base += BLOCK_N

    partial_base = (group_id * SPLITS + split) * BLOCK_H
    partial_offsets = (partial_base + h[:, None]) * HEAD_DIM + d[None, :]
    tl.store(partial_ptr + partial_offsets, accumulator, mask=h[:, None] < GROUP_H)
    tl.store(partial_max_ptr + partial_base + h, running_max, mask=h < GROUP_H)
    tl.store(partial_sum_ptr + partial_base + h, running_sum, mask=h < GROUP_H)


@triton.jit
def _paged_decode_reduce(
    partial_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    out_ptr,
    lse_ptr,
    BLOCK_H: tl.constexpr,
    GROUP_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    group_id = tl.program_id(0)
    groups_per_batch = 32 // GROUP_H
    batch = group_id // groups_per_batch
    group_in_batch = group_id - batch * groups_per_batch

    s = tl.arange(0, SPLITS)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, HEAD_DIM)
    stat_offsets = (group_id * SPLITS + s[:, None]) * BLOCK_H + h[None, :]
    partial_max = tl.load(partial_max_ptr + stat_offsets, mask=h[None, :] < GROUP_H)
    partial_sum = tl.load(partial_sum_ptr + stat_offsets, mask=h[None, :] < GROUP_H)
    global_max = tl.max(partial_max, axis=0)
    merge_scale = tl.where(partial_sum > 0.0, tl.exp(partial_max - global_max[None, :]), 0.0)
    global_sum = tl.sum(partial_sum * merge_scale, axis=0)

    partial_offsets = (
        ((group_id * SPLITS + s[:, None, None]) * BLOCK_H + h[None, :, None])
        * HEAD_DIM
        + d[None, None, :]
    )
    partial = tl.load(
        partial_ptr + partial_offsets,
        mask=h[None, :, None] < GROUP_H,
        other=0.0,
    )
    numerator = tl.sum(partial * merge_scale[:, :, None], axis=0)
    nonempty = global_sum > 0.0
    result = tl.where(nonempty[:, None], numerator / global_sum[:, None], 0.0)

    q_head = group_in_batch * GROUP_H + h
    output_offsets = (batch * 32 + q_head[:, None]) * HEAD_DIM + d[None, :]
    tl.store(out_ptr + output_offsets, result, mask=h[:, None] < GROUP_H)
    tl.store(
        lse_ptr + batch * 32 + q_head,
        tl.where(
            nonempty,
            (global_max + tl.log(global_sum)) * 1.4426950408889634,
            -float("inf"),
        ),
        mask=h < GROUP_H,
    )


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size = q.shape[0]
    average_length = (kv_indices.shape[0] + batch_size - 1) // batch_size

    if batch_size >= 32:
        group_h = 8
        if average_length < 485:
            block_n, splits = 256, 2
        elif average_length < 550:
            block_n, splits = 128, 4
        elif average_length < 690:
            block_n, splits = 64, 4
        else:
            block_n, splits = 128, 2
    elif batch_size > 1:
        group_h = 4
        if average_length <= 80:
            block_n, splits = 64, 1
        elif average_length <= 160:
            block_n, splits = 256, 1
        elif average_length <= 230:
            block_n, splits = 64, 4
        elif average_length <= 350:
            block_n, splits = 64, 8
        elif average_length <= 500:
            group_h = 8
            block_n, splits = 128, 4
        else:
            group_h = 8
            block_n, splits = 128, 8
    else:
        group_h = 4
        if average_length <= 64:
            block_n, splits = 64, 1
        elif average_length <= 128:
            block_n, splits = 128, 1
        elif average_length <= 256:
            block_n, splits = 256, 1
        elif average_length <= 400:
            block_n, splits = 128, 1
        elif average_length <= 500:
            block_n, splits = 128, 4
        else:
            block_n, splits = 128, 8

    groups = batch_size * 32 // group_h
    output_elements = batch_size * 32 * 128
    output_words = output_elements // 2
    lse_elements = batch_size * 32
    if splits == 1:
        workspace = torch.empty(
            lse_elements + output_words, dtype=torch.float32, device=q.device
        )
        lse = workspace[:lse_elements].view(batch_size, 32)
        output = workspace[lse_elements:].view(torch.bfloat16).view_as(q)
        _paged_decode_grouped[(groups,)](
            q,
            k_cache,
            v_cache,
            kv_indptr,
            kv_indices,
            output,
            lse,
            sm_scale,
            BLOCK_N=block_n,
            HEAD_DIM=128,
            BLOCK_H=group_h,
            GROUP_H=group_h,
            USE_BF16_P=True,
            num_warps=2,
        )
    else:
        partial_elements = groups * splits * group_h * 128
        stats_elements = 2 * groups * splits * group_h
        workspace = torch.empty(
            partial_elements + stats_elements + lse_elements + output_words,
            dtype=torch.float32,
            device=q.device,
        )
        partial = workspace[:partial_elements].view(groups, splits, group_h, 128)
        stats = workspace[
            partial_elements : partial_elements + stats_elements
        ].view(2, groups, splits, group_h)
        lse_begin = partial_elements + stats_elements
        lse = workspace[lse_begin : lse_begin + lse_elements].view(batch_size, 32)
        output = (
            workspace[lse_begin + lse_elements :]
            .view(torch.bfloat16)
            .view_as(q)
        )
        _paged_decode_split[(groups, splits)](
            q,
            k_cache,
            v_cache,
            kv_indptr,
            kv_indices,
            partial,
            stats[0],
            stats[1],
            sm_scale,
            BLOCK_N=block_n,
            HEAD_DIM=128,
            BLOCK_H=group_h,
            GROUP_H=group_h,
            SPLITS=splits,
            num_warps=2 if group_h == 4 else 4,
        )
        _paged_decode_reduce[(groups,)](
            partial,
            stats[0],
            stats[1],
            output,
            lse,
            BLOCK_H=group_h,
            GROUP_H=group_h,
            HEAD_DIM=128,
            SPLITS=splits,
            num_warps=2 if group_h == 4 else 4,
        )
    return output, lse
