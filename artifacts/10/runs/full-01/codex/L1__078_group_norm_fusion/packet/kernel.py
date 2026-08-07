import torch
import triton
import triton.language as tl


@triton.jit
def _group_norm_kernel(
    x, weight, bias, output,
    n: tl.constexpr, spatial: tl.constexpr, channels_per_group: tl.constexpr,
    eps: tl.constexpr, BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + group * n + offsets, mask=mask, other=0.0)

    mean = tl.sum(values, axis=0) / n
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) * (1.0 / n)
    normalized = centered / tl.sqrt(variance + eps)

    channel = (group % 32) * channels_per_group + offsets // spatial
    scale = tl.load(weight + channel, mask=mask)
    shift = tl.load(bias + channel, mask=mask)
    result = normalized * scale + shift
    tl.store(output + group * n + offsets, result, mask=mask)


@triton.jit
def _stats_kernel(x, mean_output, variance_output,
                  n: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + group * n + offsets, mask=mask, other=0.0)
    mean = tl.sum(values, axis=0) / n
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) * (1.0 / n)
    tl.store(mean_output + group, mean)
    tl.store(variance_output + group, variance)


@triton.jit
def _variance_apply_kernel(
    x, weight, bias, mean, output,
    n: tl.constexpr, spatial: tl.constexpr, channels_per_group: tl.constexpr,
    eps: tl.constexpr, BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + group * n + offsets, mask=mask, other=0.0)
    avg = tl.load(mean + group)
    centered = values - avg
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) * (1.0 / n)
    normalized = centered / tl.sqrt(variance + eps)
    channel = (group % 32) * channels_per_group + offsets // spatial
    scale = tl.load(weight + channel, mask=mask)
    shift = tl.load(bias + channel, mask=mask)
    result = normalized * scale + shift
    tl.store(output + group * n + offsets, result, mask=mask)


@triton.jit
def _partial_variance_kernel(
    x, mean, partials,
    n: tl.constexpr, PARTS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    group = program // PARTS
    part = program % PARTS
    offsets = part * CHUNK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + group * n + offsets, mask=mask, other=0.0)
    avg = tl.load(mean + group)
    centered = values - avg
    partial = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0)
    tl.store(partials + program, partial)


@triton.jit
def _segmented_apply_kernel(
    x, weight, bias, mean, partials, output,
    n: tl.constexpr, spatial: tl.constexpr, channels_per_group: tl.constexpr,
    parts: tl.constexpr, chunks: tl.constexpr, eps: tl.constexpr,
    BLOCK: tl.constexpr, PARTIAL_BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    partial_offsets = tl.arange(0, PARTIAL_BLOCK)
    partial_values = tl.load(
        partials + group * parts + partial_offsets,
        mask=partial_offsets < parts, other=0.0,
    )
    variance = tl.sum(partial_values, axis=0) * (1.0 / n)
    avg = tl.load(mean + group)
    values = tl.load(x + group * n + offsets, mask=mask)
    normalized = (values - avg) / tl.sqrt(variance + eps)
    channel = (group % 32) * channels_per_group + offsets // spatial
    scale = tl.load(weight + channel, mask=mask)
    shift = tl.load(bias + channel, mask=mask)
    result = normalized * scale + shift
    tl.store(output + group * n + offsets, result, mask=mask)


@triton.jit
def _segmented_apply_channel_kernel(
    x, weight, bias, mean, partials, output,
    n: tl.constexpr, spatial: tl.constexpr, channels_per_group: tl.constexpr,
    parts: tl.constexpr, eps: tl.constexpr,
    BLOCK: tl.constexpr, PARTIAL_BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    channel_in_group = tl.program_id(1)
    chunk = tl.program_id(2)
    spatial_offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = spatial_offsets < spatial

    partial_offsets = tl.arange(0, PARTIAL_BLOCK)
    partial_values = tl.load(
        partials + group * parts + partial_offsets,
        mask=partial_offsets < parts, other=0.0,
    )
    variance = tl.sum(partial_values, axis=0) * (1.0 / n)
    avg = tl.load(mean + group)
    group_offsets = channel_in_group * spatial + spatial_offsets
    values = tl.load(x + group * n + group_offsets, mask=mask)
    normalized = (values - avg) / tl.sqrt(variance + eps)
    channel = (group % 32) * channels_per_group + channel_in_group
    scale = tl.load(weight + channel)
    shift = tl.load(bias + channel)
    result = normalized * scale + shift
    tl.store(output + group * n + group_offsets, result, mask=mask)


@triton.jit
def _mean_serial_kernel(x, mean_output, n: tl.constexpr,
                        THREADS: tl.constexpr, VEC: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, THREADS)
    acc = tl.zeros((THREADS,), tl.float32)
    for base in tl.static_range(0, n, THREADS * VEC):
        for v in tl.static_range(0, VEC):
            offset = base + lane * VEC + v
            value = tl.load(x + group * n + offset,
                            mask=offset < n, other=0.0)
            acc += value
    mean = tl.sum(acc, axis=0) / n
    tl.store(mean_output + group, mean)


@triton.jit
def _mean_loop_kernel(x, mean_output, n: tl.constexpr,
                      THREADS: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, THREADS)
    acc = tl.zeros((THREADS,), tl.float32)
    for base in tl.range(0, n, THREADS, loop_unroll_factor=1):
        offset = base + lane
        value = tl.load(x + group * n + offset,
                        mask=offset < n, other=0.0)
        acc += value
    mean = tl.sum(acc, axis=0) / n
    tl.store(mean_output + group, mean)


@triton.jit
def _mean_vt_kernel(x, mean_output, n: tl.constexpr,
                    THREADS: tl.constexpr, VT: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, THREADS)
    acc = tl.zeros((THREADS,), tl.float32)
    for base in tl.range(0, n, THREADS * VT, loop_unroll_factor=1):
        for v in tl.static_range(0, VT):
            offset = base + lane + v * THREADS
            value = tl.load(x + group * n + offset,
                            mask=offset < n, other=0.0)
            acc += value
    mean = tl.sum(acc, axis=0) / n
    tl.store(mean_output + group, mean)


@triton.jit
def _mean_vec_loop_kernel(x, mean_output, n: tl.constexpr,
                          THREADS: tl.constexpr, VEC: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, THREADS)
    acc = tl.zeros((THREADS,), tl.float32)
    for base in tl.range(0, n, THREADS * VEC, loop_unroll_factor=1):
        for v in tl.static_range(0, VEC):
            offset = base + lane * VEC + v
            value = tl.load(x + group * n + offset,
                            mask=offset < n, other=0.0)
            acc += value
    mean = tl.sum(acc, axis=0) / n
    tl.store(mean_output + group, mean)


@triton.jit
def _mean_manual_kernel(x, mean_output, n: tl.constexpr,
                        VEC: tl.constexpr, CONTIG: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, 512)
    acc = tl.zeros((512,), tl.float32)
    for base in tl.range(0, n, 512 * VEC, loop_unroll_factor=1):
        for v in tl.static_range(0, VEC):
            if CONTIG:
                offset = base + lane * VEC + v
            else:
                offset = base + lane + v * 512
            value = tl.load(x + group * n + offset,
                            mask=offset < n, other=0.0)
            acc += value
    rows = tl.reshape(acc, (8, 64))
    wave_sums = tl.sum(rows, axis=1)
    total = tl.sum(wave_sums, axis=0)
    tl.store(mean_output + group, total * (1.0 / n))


@triton.jit
def _mean_multiacc_kernel(x, mean_output, n: tl.constexpr,
                          CONTIG: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, 512)[None, :]
    item = tl.arange(0, 4)[:, None]
    acc = tl.zeros((4, 512), tl.float32)
    for base in tl.range(0, n, 2048, loop_unroll_factor=1):
        if CONTIG:
            offsets = base + lane * 4 + item
        else:
            offsets = base + lane + item * 512
        values = tl.load(x + group * n + offsets,
                         mask=offsets < n, other=0.0)
        acc += values
    thread_sums = tl.sum(acc, axis=0)
    rows = tl.reshape(thread_sums, (8, 64))
    wave_sums = tl.sum(rows, axis=1)
    total = tl.sum(wave_sums, axis=0)
    tl.store(mean_output + group, total / n)


@triton.jit
def _mean_ordered_kernel(x, mean_output, n: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, 512)
    acc = tl.zeros((512,), tl.float32)
    for base in tl.range(0, n, 512, loop_unroll_factor=1):
        offset = base + lane
        value = tl.load(x + group * n + offset,
                        mask=offset < n, other=0.0)
        acc += value

    level = tl.sum(tl.reshape(acc, (8, 32, 2)), axis=2)
    level = tl.sum(tl.reshape(level, (8, 16, 2)), axis=2)
    level = tl.sum(tl.reshape(level, (8, 8, 2)), axis=2)
    level = tl.sum(tl.reshape(level, (8, 4, 2)), axis=2)
    level = tl.sum(tl.reshape(level, (8, 2, 2)), axis=2)
    wave_sums = tl.sum(tl.reshape(level, (8, 1, 2)), axis=2)

    y_level = tl.sum(tl.reshape(wave_sums, (2, 4)), axis=0)
    y_level = tl.sum(tl.reshape(y_level, (2, 2)), axis=0)
    total = tl.sum(y_level, axis=0)
    tl.store(mean_output + group, total * (1.0 / n))


@triton.jit
def _debug_acc_kernel(x, acc_output, n: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, 512)
    acc = tl.zeros((512,), tl.float32)
    for base in tl.range(0, n, 512, loop_unroll_factor=1):
        offset = base + lane
        value = tl.load(x + group * n + offset,
                        mask=offset < n, other=0.0)
        acc += value
    tl.store(acc_output + group * 512 + lane, acc)


@triton.jit
def _debug_scale_kernel(sums, out_div, out_mul, n: tl.constexpr,
                        count: tl.constexpr):
    offsets = tl.arange(0, 1024)
    mask = offsets < count
    values = tl.load(sums + offsets, mask=mask)
    tl.store(out_div + offsets, values / n, mask=mask)
    tl.store(out_mul + offsets, values * (1.0 / n), mask=mask)


@triton.jit
def _mean_exact_kernel(x, mean_output, n: tl.constexpr):
    group = tl.program_id(0)
    lane = tl.arange(0, 512)
    acc0 = tl.zeros((512,), tl.float32)
    acc1 = tl.zeros((512,), tl.float32)
    acc2 = tl.zeros((512,), tl.float32)
    acc3 = tl.zeros((512,), tl.float32)
    for base in tl.range(0, n, 2048, loop_unroll_factor=1):
        offset = base + lane * 4
        mask0 = offset < n
        mask1 = offset + 1 < n
        mask2 = offset + 2 < n
        mask3 = offset + 3 < n
        acc0 += tl.load(x + group * n + offset, mask=mask0, other=0.0)
        acc1 += tl.load(x + group * n + offset + 1, mask=mask1, other=0.0)
        acc2 += tl.load(x + group * n + offset + 2, mask=mask2, other=0.0)
        acc3 += tl.load(x + group * n + offset + 3, mask=mask3, other=0.0)

    thread_sums = acc0 + acc1
    thread_sums = thread_sums + acc2
    thread_sums = thread_sums + acc3

    rows = tl.reshape(thread_sums, (8, 64))
    y_level = tl.sum(tl.reshape(rows, (2, 4, 64)), axis=0)
    y_level = tl.sum(tl.reshape(y_level, (2, 2, 64)), axis=0)
    x_level = tl.sum(tl.reshape(y_level, (2, 1, 64)), axis=0)
    x_level = tl.sum(tl.reshape(x_level, (1, 32, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 16, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 8, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 4, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 2, 2)), axis=2)
    total = tl.sum(tl.reshape(x_level, (1, 1, 2)), axis=2)
    total = tl.sum(total, axis=1)
    total = tl.sum(total, axis=0)
    tl.store(mean_output + group, total * (1.0 / n))


@triton.jit
def _exact_mean_variance_apply_kernel(
    x, weight, bias, output,
    n: tl.constexpr, spatial: tl.constexpr, channels_per_group: tl.constexpr,
    eps: tl.constexpr, BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    lane = tl.arange(0, 512)
    acc0 = tl.zeros((512,), tl.float32)
    acc1 = tl.zeros((512,), tl.float32)
    acc2 = tl.zeros((512,), tl.float32)
    acc3 = tl.zeros((512,), tl.float32)
    for base in tl.range(0, n, 2048, loop_unroll_factor=1):
        offset = base + lane * 4
        acc0 += tl.load(x + group * n + offset,
                        mask=offset < n, other=0.0)
        acc1 += tl.load(x + group * n + offset + 1,
                        mask=offset + 1 < n, other=0.0)
        acc2 += tl.load(x + group * n + offset + 2,
                        mask=offset + 2 < n, other=0.0)
        acc3 += tl.load(x + group * n + offset + 3,
                        mask=offset + 3 < n, other=0.0)

    thread_sums = acc0 + acc1
    thread_sums = thread_sums + acc2
    thread_sums = thread_sums + acc3
    rows = tl.reshape(thread_sums, (8, 64))
    y_level = tl.sum(tl.reshape(rows, (2, 4, 64)), axis=0)
    y_level = tl.sum(tl.reshape(y_level, (2, 2, 64)), axis=0)
    x_level = tl.sum(tl.reshape(y_level, (2, 1, 64)), axis=0)
    x_level = tl.sum(tl.reshape(x_level, (1, 32, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 16, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 8, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 4, 2)), axis=2)
    x_level = tl.sum(tl.reshape(x_level, (1, 2, 2)), axis=2)
    total = tl.sum(tl.reshape(x_level, (1, 1, 2)), axis=2)
    total = tl.sum(total, axis=1)
    total = tl.sum(total, axis=0)
    mean = total * (1.0 / n)

    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + group * n + offsets, mask=mask, other=0.0)
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) * (1.0 / n)
    normalized = centered / tl.sqrt(variance + eps)
    channel = (group % 32) * channels_per_group + offsets // spatial
    scale = tl.load(weight + channel, mask=mask)
    shift = tl.load(bias + channel, mask=mask)
    result = normalized * scale + shift
    tl.store(output + group * n + offsets, result, mask=mask)


@triton.jit
def _apply_kernel(
    x, weight, bias, mean, variance, output,
    total: tl.constexpr, n: tl.constexpr, spatial: tl.constexpr,
    channels_per_group: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    values = tl.load(x + offsets, mask=mask)
    group = offsets // n
    avg = tl.load(mean + group, mask=mask)
    var = tl.load(variance + group, mask=mask)
    channel = (group % 32) * channels_per_group + (offsets % n) // spatial
    scale = tl.load(weight + channel, mask=mask)
    shift = tl.load(bias + channel, mask=mask)
    centered = values - avg
    normalized = centered / tl.sqrt(var + eps)
    result = normalized * scale + shift
    tl.store(output + offsets, result, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    b, c, h, w = x.shape
    spatial = h * w
    channels_per_group = c // 32
    n = channels_per_group * spatial
    groups = b * 32
    exact_mean_layout = n >= 4096 and n % 4 == 0
    output = torch.empty_like(x)
    if exact_mean_layout and n <= 65536:
        _exact_mean_variance_apply_kernel[(groups,)](
            x, weight, bias, output,
            n=n, spatial=spatial, channels_per_group=channels_per_group,
            eps=eps, BLOCK=triton.next_power_of_2(n), num_warps=8,
        )
        return output

    if exact_mean_layout:
        mean = torch.empty((groups,), device=x.device, dtype=torch.float32)
        _mean_exact_kernel[(groups,)](x, mean, n=n, num_warps=8)
    else:
        mean = x.view(groups, n).mean(dim=1)
    if n > 65536:
        chunk = 16384
        parts = triton.cdiv(n, chunk)
        partials = torch.empty((groups, parts), device=x.device, dtype=torch.float32)
        _partial_variance_kernel[(groups * parts,)](
            x, mean, partials,
            n=n, PARTS=parts, CHUNK=chunk, BLOCK=chunk, num_warps=8,
        )
        out_block = 4096
        chunks = triton.cdiv(n, out_block)
        _segmented_apply_kernel[(groups, chunks)](
            x, weight, bias, mean, partials, output,
            n=n, spatial=spatial, channels_per_group=channels_per_group,
            parts=parts, chunks=chunks, eps=eps, BLOCK=out_block,
            PARTIAL_BLOCK=triton.next_power_of_2(parts), num_warps=4,
        )
    else:
        _variance_apply_kernel[(groups,)](
            x, weight, bias, mean, output,
            n=n, spatial=spatial, channels_per_group=channels_per_group,
            eps=eps, BLOCK=triton.next_power_of_2(n), num_warps=8,
        )
    return output
