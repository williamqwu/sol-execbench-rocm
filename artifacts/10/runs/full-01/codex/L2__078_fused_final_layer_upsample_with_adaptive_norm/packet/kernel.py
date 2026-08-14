import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True


@triton.jit
def _adaptive_silu_kernel(
    x,
    emb,
    out,
    n_elements: tl.constexpr,
    spatial: tl.constexpr,
    channels: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    channel = (offsets // spatial) % channels
    batch = offsets // (spatial * channels)

    x_val = tl.load(x + offsets, mask=mask).to(tl.float32)
    scale = tl.load(emb + batch * (2 * channels) + channel, mask=mask).to(
        tl.float32
    )
    shift = tl.load(
        emb + batch * (2 * channels) + channels + channel, mask=mask
    ).to(tl.float32)

    # Match the bf16 tensor boundaries in the eager expression before SiLU.
    scale = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    x_val = (x_val * scale).to(tl.bfloat16).to(tl.float32)
    x_val = (x_val + shift).to(tl.bfloat16).to(tl.float32)
    x_val = x_val * tl.sigmoid(x_val)
    tl.store(out + offsets, x_val, mask=mask)


def _adaptive_silu(x, emb):
    out = x
    n_elements = x.numel()
    spatial = x.shape[2] * x.shape[3]
    _adaptive_silu_kernel[(triton.cdiv(n_elements, 1024),)](
        x,
        emb,
        out,
        n_elements=n_elements,
        spatial=spatial,
        channels=x.shape[1],
        BLOCK=1024,
        num_warps=8,
    )
    return out


@triton.jit
def _add_transposed_kernel(
    x,
    y,
    spatial: tl.constexpr,
    channels: tl.constexpr,
    TILE_S: tl.constexpr,
    TILE_C: tl.constexpr,
):
    batch = tl.program_id(0)
    spatial_idx = tl.program_id(1) * TILE_S + tl.arange(0, TILE_S)
    channel = tl.program_id(2) * TILE_C + tl.arange(0, TILE_C)
    y_mask = (spatial_idx[:, None] < spatial) & (channel[None, :] < channels)
    y_offsets = (batch * spatial + spatial_idx[:, None]) * channels + channel[None, :]
    y_value = tl.load(y + y_offsets, mask=y_mask, other=0.0).to(tl.float32)
    x_mask = (channel[:, None] < channels) & (spatial_idx[None, :] < spatial)
    x_offsets = (batch * channels + channel[:, None]) * spatial + spatial_idx[None, :]
    x_value = tl.load(x + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
    tl.store(x + x_offsets, x_value + tl.trans(y_value), mask=x_mask)


def _add_transposed_(x, y):
    batch, channels, height, width = x.shape
    spatial = height * width
    _add_transposed_kernel[
        (batch, triton.cdiv(spatial, 32), triton.cdiv(channels, 32))
    ](
        x,
        y,
        spatial=spatial,
        channels=channels,
        TILE_S=32,
        TILE_C=32,
        num_warps=8,
    )


@triton.jit
def _groupnorm_partial_kernel(
    x,
    partial,
    spatial: tl.constexpr,
    channels: tl.constexpr,
    group_channels: tl.constexpr,
    parts: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    part = pid % parts
    batch_group = pid // parts
    batch = batch_group // 32
    group = batch_group % 32

    offsets = part * CHUNK + tl.arange(0, CHUNK)
    group_elements = spatial * group_channels
    mask = offsets < group_elements
    spatial_idx = offsets // group_channels
    channel = offsets % group_channels + group * group_channels
    x_offsets = (batch * spatial + spatial_idx) * channels + channel
    value = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
    count = tl.minimum(CHUNK, group_elements - part * CHUNK).to(tl.float32)
    mean = tl.sum(value, axis=0) / count
    centered = tl.where(mask, value - mean, 0.0)
    tl.store(partial + (batch_group * parts + part) * 2, mean)
    tl.store(
        partial + (batch_group * parts + part) * 2 + 1,
        tl.sum(centered * centered, axis=0),
    )


@triton.jit
def _groupnorm_finish_kernel(
    partial,
    stats,
    eps,
    group_elements: tl.constexpr,
    parts: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_PARTS: tl.constexpr,
):
    batch_group = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_PARTS)
    mask = offsets < parts
    means = tl.load(
        partial + (batch_group * parts + offsets) * 2, mask=mask, other=0.0
    )
    m2 = tl.load(
        partial + (batch_group * parts + offsets) * 2 + 1,
        mask=mask,
        other=0.0,
    )
    remaining = group_elements - offsets * CHUNK
    counts = tl.where(mask, tl.minimum(CHUNK, remaining), 0).to(tl.float32)
    mean = tl.sum(means * counts, axis=0) / group_elements
    delta = means - mean
    variance = tl.sum(m2 + counts * delta * delta, axis=0) / group_elements
    rstd = tl.rsqrt(variance + eps)
    tl.store(stats + batch_group * 2, mean)
    tl.store(stats + batch_group * 2 + 1, rstd)


@triton.jit
def _groupnorm_silu_transpose_kernel(
    x,
    weight,
    bias,
    stats,
    raw_out,
    norm_out,
    spatial: tl.constexpr,
    channels: tl.constexpr,
    group_channels: tl.constexpr,
    TILE_S: tl.constexpr,
    TILE_C: tl.constexpr,
):
    batch_group = tl.program_id(0)
    spatial_tile = tl.program_id(1)
    channel_tile = tl.program_id(2)
    batch = batch_group // 32
    group = batch_group % 32

    spatial_idx = spatial_tile * TILE_S + tl.arange(0, TILE_S)
    channel_in_group = channel_tile * TILE_C + tl.arange(0, TILE_C)
    channel = group * group_channels + channel_in_group
    mask = (spatial_idx[:, None] < spatial) & (
        channel_in_group[None, :] < group_channels
    )
    x_offsets = (
        (batch * spatial + spatial_idx[:, None]) * channels + channel[None, :]
    )
    value = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(stats + batch_group * 2)
    rstd = tl.load(stats + batch_group * 2 + 1)
    w = tl.load(
        weight + channel_in_group + group * group_channels,
        mask=channel_in_group < group_channels,
        other=0.0,
    ).to(tl.float32)
    b = tl.load(
        bias + channel_in_group + group * group_channels,
        mask=channel_in_group < group_channels,
        other=0.0,
    ).to(tl.float32)
    norm = (value - mean) * rstd
    norm = norm * w[None, :] + b[None, :]
    norm = norm.to(tl.bfloat16).to(tl.float32)
    norm = norm * tl.sigmoid(norm)

    out_offsets = (
        (batch * channels + channel[:, None]) * spatial + spatial_idx[None, :]
    )
    out_mask = (channel_in_group[:, None] < group_channels) & (
        spatial_idx[None, :] < spatial
    )
    tl.store(raw_out + out_offsets, tl.trans(value), mask=out_mask)
    tl.store(norm_out + out_offsets, tl.trans(norm), mask=out_mask)


def _groupnorm_silu_transpose(x, weight, bias, eps, height, width):
    batch, spatial, channels = x.shape
    group_channels = channels // 32
    group_elements = spatial * group_channels
    chunk = 4096
    parts = triton.cdiv(group_elements, chunk)
    partial = torch.empty(
        (batch * 32 * parts * 2,), device=x.device, dtype=torch.float32
    )
    stats = torch.empty((batch * 32 * 2,), device=x.device, dtype=torch.float32)
    _groupnorm_partial_kernel[(batch * 32 * parts,)](
        x,
        partial,
        spatial=spatial,
        channels=channels,
        group_channels=group_channels,
        parts=parts,
        CHUNK=chunk,
        num_warps=8,
    )
    _groupnorm_finish_kernel[(batch * 32,)](
        partial,
        stats,
        eps,
        group_elements=group_elements,
        parts=parts,
        CHUNK=chunk,
        BLOCK_PARTS=triton.next_power_of_2(parts),
        num_warps=4,
    )
    raw_out = torch.empty(
        (batch, channels, height, width), device=x.device, dtype=x.dtype
    )
    norm_out = torch.empty_like(raw_out)
    _groupnorm_silu_transpose_kernel[
        (batch * 32, triton.cdiv(spatial, 64), triton.cdiv(group_channels, 32))
    ](
        x,
        weight,
        bias,
        stats,
        raw_out,
        norm_out,
        spatial=spatial,
        channels=channels,
        group_channels=group_channels,
        TILE_S=64,
        TILE_C=32,
        num_warps=8,
    )
    return raw_out, norm_out


@triton.jit
def _add_groupnorm_partial_kernel(
    x,
    residual,
    partial,
    group_elements: tl.constexpr,
    parts: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    part = pid % parts
    batch_group = pid // parts
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    mask = offsets < group_elements
    global_offsets = batch_group * group_elements + offsets
    value = tl.load(x + global_offsets, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(residual + global_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    value = value.to(tl.bfloat16).to(tl.float32)
    count = tl.minimum(CHUNK, group_elements - part * CHUNK).to(tl.float32)
    mean = tl.sum(value, axis=0) / count
    centered = tl.where(mask, value - mean, 0.0)
    tl.store(partial + (batch_group * parts + part) * 2, mean)
    tl.store(
        partial + (batch_group * parts + part) * 2 + 1,
        tl.sum(centered * centered, axis=0),
    )


@triton.jit
def _add_groupnorm_silu_kernel(
    x,
    residual,
    weight,
    bias,
    stats,
    spatial: tl.constexpr,
    group_channels: tl.constexpr,
    group_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_group = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < group_elements
    global_offsets = batch_group * group_elements + offsets
    value = tl.load(x + global_offsets, mask=mask, other=0.0).to(tl.float32)
    value += tl.load(residual + global_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    value = value.to(tl.bfloat16).to(tl.float32)
    mean = tl.load(stats + batch_group * 2)
    rstd = tl.load(stats + batch_group * 2 + 1)
    group = batch_group % 32
    channel = group * group_channels + offsets // spatial
    w = tl.load(weight + channel, mask=mask).to(tl.float32)
    b = tl.load(bias + channel, mask=mask).to(tl.float32)
    value = (value - mean) * rstd
    value = value * w + b
    value = value.to(tl.bfloat16).to(tl.float32)
    value = value * tl.sigmoid(value)
    tl.store(x + global_offsets, value, mask=mask)


def _add_groupnorm_silu_(x, residual, weight, bias, eps):
    batch, channels, height, width = x.shape
    spatial = height * width
    group_channels = channels // 32
    group_elements = group_channels * spatial
    chunk = 4096
    parts = triton.cdiv(group_elements, chunk)
    partial = torch.empty(
        (batch * 32 * parts * 2,), device=x.device, dtype=torch.float32
    )
    stats = torch.empty((batch * 32 * 2,), device=x.device, dtype=torch.float32)
    _add_groupnorm_partial_kernel[(batch * 32 * parts,)](
        x,
        residual,
        partial,
        group_elements=group_elements,
        parts=parts,
        CHUNK=chunk,
        num_warps=8,
    )
    _groupnorm_finish_kernel[(batch * 32,)](
        partial,
        stats,
        eps,
        group_elements=group_elements,
        parts=parts,
        CHUNK=chunk,
        BLOCK_PARTS=triton.next_power_of_2(parts),
        num_warps=4,
    )
    _add_groupnorm_silu_kernel[
        (batch * 32, triton.cdiv(group_elements, 1024))
    ](
        x,
        residual,
        weight,
        bias,
        stats,
        spatial=spatial,
        group_channels=group_channels,
        group_elements=group_elements,
        BLOCK=1024,
        num_warps=8,
    )


@triton.jit
def _groupnorm_partial_nchw_kernel(
    x,
    partial,
    group_elements: tl.constexpr,
    parts: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    part = pid % parts
    batch_group = pid // parts
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    mask = offsets < group_elements
    value = tl.load(
        x + batch_group * group_elements + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    count = tl.minimum(CHUNK, group_elements - part * CHUNK).to(tl.float32)
    mean = tl.sum(value, axis=0) / count
    centered = tl.where(mask, value - mean, 0.0)
    tl.store(partial + (batch_group * parts + part) * 2, mean)
    tl.store(
        partial + (batch_group * parts + part) * 2 + 1,
        tl.sum(centered * centered, axis=0),
    )


@triton.jit
def _groupnorm_adaptive_silu_kernel(
    x,
    emb,
    weight,
    bias,
    stats,
    spatial: tl.constexpr,
    channels: tl.constexpr,
    group_channels: tl.constexpr,
    group_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_group = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < group_elements
    global_offsets = batch_group * group_elements + offsets
    value = tl.load(x + global_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(stats + batch_group * 2)
    rstd = tl.load(stats + batch_group * 2 + 1)
    batch = batch_group // 32
    group = batch_group % 32
    channel = group * group_channels + offsets // spatial
    w = tl.load(weight + channel, mask=mask).to(tl.float32)
    b = tl.load(bias + channel, mask=mask).to(tl.float32)
    value = (value - mean) * rstd
    value = value * w + b
    value = value.to(tl.bfloat16).to(tl.float32)

    scale = tl.load(emb + batch * (2 * channels) + channel, mask=mask).to(
        tl.float32
    )
    shift = tl.load(
        emb + batch * (2 * channels) + channels + channel, mask=mask
    ).to(tl.float32)
    scale = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    value = (value * scale).to(tl.bfloat16).to(tl.float32)
    value = (value + shift).to(tl.bfloat16).to(tl.float32)
    value = value * tl.sigmoid(value)
    tl.store(x + global_offsets, value, mask=mask)


def _groupnorm_adaptive_silu_(x, emb, weight, bias, eps):
    batch, channels, height, width = x.shape
    spatial = height * width
    group_channels = channels // 32
    group_elements = group_channels * spatial
    chunk = 4096
    parts = triton.cdiv(group_elements, chunk)
    partial = torch.empty(
        (batch * 32 * parts * 2,), device=x.device, dtype=torch.float32
    )
    stats = torch.empty((batch * 32 * 2,), device=x.device, dtype=torch.float32)
    _groupnorm_partial_nchw_kernel[(batch * 32 * parts,)](
        x,
        partial,
        group_elements=group_elements,
        parts=parts,
        CHUNK=chunk,
        num_warps=8,
    )
    _groupnorm_finish_kernel[(batch * 32,)](
        partial,
        stats,
        eps,
        group_elements=group_elements,
        parts=parts,
        CHUNK=chunk,
        BLOCK_PARTS=triton.next_power_of_2(parts),
        num_warps=4,
    )
    _groupnorm_adaptive_silu_kernel[
        (batch * 32, triton.cdiv(group_elements, 1024))
    ](
        x,
        emb,
        weight,
        bias,
        stats,
        spatial=spatial,
        channels=channels,
        group_channels=group_channels,
        group_elements=group_elements,
        BLOCK=1024,
        num_warps=8,
    )


@torch.no_grad()
def run(
    x,
    timestep_emb,
    time_emb_mlp_linear_weight,
    time_emb_mlp_linear_bias,
    resblock_in_norm_weight,
    resblock_in_norm_bias,
    resblock_in_conv_weight,
    resblock_in_conv_bias,
    resblock_emb_linear_weight,
    resblock_emb_linear_bias,
    resblock_out_norm_weight,
    resblock_out_norm_bias,
    resblock_out_conv_weight,
    resblock_out_conv_bias,
    resblock_skip_conv_weight,
    resblock_skip_conv_bias,
    final_norm_weight,
    final_norm_bias,
    final_conv_weight,
    final_conv_bias,
    eps,
):
    batch_size, seq_len, hidden_size = x.shape
    token_h = int(seq_len ** 0.5)
    token_w = seq_len // token_h
    while token_h * token_w != seq_len:
        token_h -= 1
        token_w = seq_len // token_h

    x_spatial = x.reshape(batch_size, token_h, token_w, hidden_size)
    x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()
    h = F.group_norm(
        x_spatial, 32, resblock_in_norm_weight, resblock_in_norm_bias, eps
    )
    h = F.silu(h, inplace=True)
    h = F.conv2d(h, resblock_in_conv_weight, resblock_in_conv_bias, padding=1)

    emb_out = F.linear(
        F.silu(timestep_emb),
        resblock_emb_linear_weight,
        resblock_emb_linear_bias,
    )
    h = F.group_norm(
        h, 32, resblock_out_norm_weight, resblock_out_norm_bias, eps
    )
    h = _adaptive_silu(h, emb_out)
    h = F.conv2d(h, resblock_out_conv_weight, resblock_out_conv_bias, padding=1)

    skip = F.linear(
        x, resblock_skip_conv_weight[:, :, 0, 0], resblock_skip_conv_bias
    )
    _add_transposed_(h, skip)
    h = F.group_norm(h, 32, final_norm_weight, final_norm_bias, eps)
    h = F.silu(h, inplace=True)
    return F.conv2d(h, final_conv_weight, final_conv_bias, padding=1)
