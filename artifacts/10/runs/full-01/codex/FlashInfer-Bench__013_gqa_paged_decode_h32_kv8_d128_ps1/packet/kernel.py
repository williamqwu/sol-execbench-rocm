import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    sm_scale,
    BLOCK_N: tl.constexpr,
):
    # Four useful rows (the GQA group) are padded to the native 16-row MFMA
    # tile.  K and V are still fetched just once for all four query heads.
    pid = tl.program_id(0)
    batch = pid // 8
    kv_head = pid % 8
    row = tl.arange(0, 16)
    d = tl.arange(0, 128)

    q_base = batch * (32 * 128) + kv_head * (4 * 128)
    q = tl.load(
        q_ptr + q_base + row[:, None] * 128 + d[None, :],
        mask=row[:, None] < 4,
        other=0.0,
    )

    acc = tl.zeros([16, 128], tl.float32)
    m = tl.full([16], -float("inf"), tl.float32)
    z = tl.zeros([16], tl.float32)

    begin = tl.load(indptr_ptr + batch)
    end = tl.load(indptr_ptr + batch + 1)
    length = end - begin
    n_lane = tl.arange(0, BLOCK_N)
    block_start = 0

    while block_start < length:
        n = block_start + n_lane
        valid = n < length
        page = tl.load(indices_ptr + begin + n, mask=valid, other=0)
        cache_off = page[:, None] * (8 * 128) + kv_head * 128 + d[None, :]
        k = tl.load(k_ptr + cache_off, mask=valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        block_m = tl.max(scores, axis=1)
        new_m = tl.maximum(m, block_m)
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])

        v = tl.load(v_ptr + cache_off, mask=valid[:, None], other=0.0)
        # Softmax probabilities benefit from FP16's three extra mantissa bits.
        # BF16 cache values in the workload's finite range convert exactly.
        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32
        )
        z = z * alpha + tl.sum(p, axis=1)
        m = new_m
        block_start += BLOCK_N

    nonempty = length > 0
    out = tl.where(nonempty, acc / z[:, None], 0.0)
    tl.store(
        out_ptr + q_base + row[:, None] * 128 + d[None, :],
        out,
        mask=row[:, None] < 4,
    )

    log2e: tl.constexpr = 1.4426950408889634
    lse = tl.where(nonempty, (m + tl.log(z)) * log2e, -float("inf"))
    tl.store(
        lse_ptr + batch * 32 + kv_head * 4 + row,
        lse,
        mask=row < 4,
    )


@triton.jit
def _gqa_decode_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indptr_ptr,
    indices_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_z_ptr,
    sm_scale,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // SPLITS
    split = pid % SPLITS
    batch = group // 8
    kv_head = group % 8
    row = tl.arange(0, 16)
    d = tl.arange(0, 128)

    q_base = batch * (32 * 128) + kv_head * (4 * 128)
    q = tl.load(
        q_ptr + q_base + row[:, None] * 128 + d[None, :],
        mask=row[:, None] < 4,
        other=0.0,
    )

    acc = tl.zeros([16, 128], tl.float32)
    m = tl.full([16], -float("inf"), tl.float32)
    z = tl.zeros([16], tl.float32)
    sequence_begin = tl.load(indptr_ptr + batch)
    sequence_end = tl.load(indptr_ptr + batch + 1)
    sequence_length = sequence_end - sequence_begin
    segment_begin = sequence_length * split // SPLITS
    segment_end = sequence_length * (split + 1) // SPLITS
    begin = sequence_begin + segment_begin
    length = segment_end - segment_begin

    n_lane = tl.arange(0, BLOCK_N)
    block_start = 0
    while block_start < length:
        n = block_start + n_lane
        valid = n < length
        page = tl.load(indices_ptr + begin + n, mask=valid, other=0)
        cache_off = page[:, None] * (8 * 128) + kv_head * 128 + d[None, :]
        k = tl.load(k_ptr + cache_off, mask=valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        block_m = tl.max(scores, axis=1)
        new_m = tl.maximum(m, block_m)
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        v = tl.load(v_ptr + cache_off, mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32
        )
        z = z * alpha + tl.sum(p, axis=1)
        m = new_m
        block_start += BLOCK_N

    partial = group * SPLITS + split
    useful = row < 4
    tl.store(
        partial_acc_ptr + partial * (4 * 128) + row[:, None] * 128 + d[None, :],
        acc,
        mask=useful[:, None],
    )
    tl.store(partial_m_ptr + partial * 4 + row, m, mask=useful)
    tl.store(partial_z_ptr + partial * 4 + row, z, mask=useful)


@triton.jit
def _merge_splits_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_z_ptr,
    indptr_ptr,
    out_ptr,
    lse_ptr,
    SPLITS: tl.constexpr,
):
    group = tl.program_id(0)
    batch = group // 8
    kv_head = group % 8
    split = tl.arange(0, SPLITS)
    row = tl.arange(0, 4)
    d = tl.arange(0, 128)
    partial = group * SPLITS + split

    pm = tl.load(partial_m_ptr + partial[:, None] * 4 + row[None, :])
    pz = tl.load(partial_z_ptr + partial[:, None] * 4 + row[None, :])
    global_m = tl.max(pm, axis=0)
    alpha = tl.exp(pm - global_m[None, :])
    denominator = tl.sum(pz * alpha, axis=0)
    pa = tl.load(
        partial_acc_ptr
        + partial[:, None, None] * (4 * 128)
        + row[None, :, None] * 128
        + d[None, None, :]
    )
    numerator = tl.sum(pa * alpha[:, :, None], axis=0)
    nonempty = tl.load(indptr_ptr + batch + 1) > tl.load(indptr_ptr + batch)
    out = tl.where(nonempty, numerator / denominator[:, None], 0.0)
    out_base = batch * (32 * 128) + kv_head * (4 * 128)
    tl.store(out_ptr + out_base + row[:, None] * 128 + d[None, :], out)

    log2e: tl.constexpr = 1.4426950408889634
    lse = tl.where(
        nonempty,
        (global_m + tl.log(denominator)) * log2e,
        -float("inf"),
    )
    tl.store(lse_ptr + batch * 32 + kv_head * 4 + row, lse)


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size = q.shape[0]
    output = torch.empty_like(q)
    lse = torch.empty((batch_size, 32), dtype=torch.float32, device=q.device)
    average_length = kv_indices.shape[0] // batch_size

    if batch_size == 64 or (batch_size == 16 and average_length > 600):
        splits = 4 if batch_size == 64 else 8
        partial_count = batch_size * 8 * splits
        acc_elements = partial_count * 4 * 128
        stat_elements = partial_count * 4
        workspace = torch.empty(
            acc_elements + 2 * stat_elements,
            dtype=torch.float32,
            device=q.device,
        )
        partial_acc = workspace[:acc_elements]
        partial_m = workspace[acc_elements : acc_elements + stat_elements]
        partial_z = workspace[acc_elements + stat_elements :]
        _gqa_decode_split_kernel[(partial_count,)](
            q,
            k_cache,
            v_cache,
            kv_indptr,
            kv_indices,
            partial_acc,
            partial_m,
            partial_z,
            sm_scale,
            SPLITS=splits,
            BLOCK_N=64,
            num_warps=1,
        )
        _merge_splits_kernel[(batch_size * 8,)](
            partial_acc,
            partial_m,
            partial_z,
            kv_indptr,
            output,
            lse,
            SPLITS=splits,
            num_warps=4,
        )
        return output, lse

    if average_length <= 64:
        block_n = 16
    elif average_length <= 256:
        block_n = 32
    elif average_length <= 600:
        block_n = 64
    else:
        block_n = 128
    _gqa_decode_kernel[(batch_size * 8,)](
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        output,
        lse,
        sm_scale,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return output, lse
