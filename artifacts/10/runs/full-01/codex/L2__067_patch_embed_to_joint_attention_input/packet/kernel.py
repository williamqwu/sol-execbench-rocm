import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sincos_kernel(timestep_ptr, freqs_ptr, out_ptr, count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    batch = offsets // 128
    freq = offsets % 128
    args = tl.load(timestep_ptr + batch, mask=mask) * tl.load(
        freqs_ptr + freq, mask=mask
    )
    base = batch * 256 + freq
    tl.store(out_ptr + base, tl.cos(args), mask=mask)
    tl.store(out_ptr + base + 128, tl.sin(args), mask=mask)


def _sincos(timestep, freqs):
    count = timestep.shape[0] * 128
    out = torch.empty((timestep.shape[0], 256), device=timestep.device, dtype=timestep.dtype)
    _sincos_kernel[(triton.cdiv(count, 256),)](
        timestep, freqs, out, count=count, BLOCK=256, num_warps=4
    )
    return out


@triton.jit
def _transpose_add_kernel(
    conv_ptr,
    pos_ptr,
    out_ptr,
    PATCHES: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_p = tl.program_id(0)
    tile_n = tl.program_id(1)
    batch = tl.program_id(2)
    p = tile_p * BLOCK_P + tl.arange(0, BLOCK_P)
    n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (p[None, :] < PATCHES) & (n[:, None] < WIDTH)

    # conv is physically [B, WIDTH, PATCHES].  The returned tensor is
    # contiguous [B, PATCHES, WIDTH], so this is a tiled transpose while the
    # positional value is added.
    conv_offs = batch * WIDTH * PATCHES + n[:, None] * PATCHES + p[None, :]
    pos_offs = p[None, :] * WIDTH + n[:, None]
    out_offs = batch * PATCHES * WIDTH + pos_offs
    values = tl.load(conv_ptr + conv_offs, mask=mask)
    positions = tl.load(pos_ptr + pos_offs, mask=mask)
    tl.store(out_ptr + out_offs, values + positions, mask=mask)


def _transpose_add(conv, pos):
    batch = conv.shape[0]
    out = torch.empty(
        (batch, 4096, 2432), device=conv.device, dtype=conv.dtype
    )
    grid = (triton.cdiv(4096, 128), triton.cdiv(2432, 32), batch)
    _transpose_add_kernel[grid](
        conv,
        pos,
        out,
        PATCHES=4096,
        WIDTH=2432,
        BLOCK_P=128,
        BLOCK_N=32,
        num_warps=8,
    )
    return out


@triton.jit
def _patch_embed_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    pos_ptr,
    out_ptr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, 64)

    batch = rows // 4096
    patch = rows % 4096
    patch_y = patch // 64
    patch_x = patch % 64
    channel = k // 4
    kernel_y = (k % 4) // 2
    kernel_x = k % 2
    hidden_offs = (
        batch[:, None] * (16 * 128 * 128)
        + channel[None, :] * (128 * 128)
        + (patch_y[:, None] * 2 + kernel_y[None, :]) * 128
        + patch_x[:, None] * 2
        + kernel_x[None, :]
    )
    x = tl.load(hidden_ptr + hidden_offs, mask=rows[:, None] < TOTAL_ROWS, other=0.0)
    weight_offs = cols[None, :] * 64 + k[:, None]
    w = tl.load(weight_ptr + weight_offs, mask=cols[None, :] < 2432, other=0.0)
    accum = tl.dot(x, w, input_precision="ieee")
    mask = (rows[:, None] < TOTAL_ROWS) & (cols[None, :] < 2432)
    accum += tl.load(bias_ptr + cols[None, :], mask=cols[None, :] < 2432)
    pos_offs = patch[:, None] * 2432 + cols[None, :]
    accum += tl.load(pos_ptr + pos_offs, mask=mask)
    out_offs = rows[:, None] * 2432 + cols[None, :]
    tl.store(out_ptr + out_offs, accum, mask=mask)


def _patch_embed(hidden, weight, bias, pos, block_m=64, block_n=64, warps=4):
    total_rows = hidden.shape[0] * 4096
    out = torch.empty(
        (hidden.shape[0], 4096, 2432), device=hidden.device, dtype=hidden.dtype
    )
    grid = (triton.cdiv(total_rows, block_m), triton.cdiv(2432, block_n))
    _patch_embed_kernel[grid](
        hidden,
        weight,
        bias,
        pos,
        out,
        TOTAL_ROWS=total_rows,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
    )
    return out


@triton.jit
def _patch_embed_sequential_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    pos_ptr,
    out_ptr,
    TOTAL_ROWS: tl.constexpr,
    BIAS_FIRST: tl.constexpr,
    ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = rows // 4096
    patch = rows % 4096
    patch_y = patch // 64
    patch_x = patch % 64
    if BIAS_FIRST:
        accum = tl.broadcast_to(
            tl.load(bias_ptr + cols[None, :], mask=cols[None, :] < 2432, other=0.0),
            (BLOCK_M, BLOCK_N),
        )
    else:
        accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ii in range(64):
        if ORDER == 0:
            kk = ii
        else:
            kk = (ii % 16) * 4 + ii // 16
        channel = kk // 4
        kernel_y = (kk % 4) // 2
        kernel_x = kk % 2
        hidden_offs = (
            batch * (16 * 128 * 128)
            + channel * (128 * 128)
            + (patch_y * 2 + kernel_y) * 128
            + patch_x * 2
            + kernel_x
        )
        xv = tl.load(hidden_ptr + hidden_offs, mask=rows < TOTAL_ROWS, other=0.0)
        wv = tl.load(weight_ptr + cols * 64 + kk, mask=cols < 2432, other=0.0)
        accum += xv[:, None] * wv[None, :]
    if not BIAS_FIRST:
        accum += tl.load(
            bias_ptr + cols[None, :], mask=cols[None, :] < 2432, other=0.0
        )
    mask = (rows[:, None] < TOTAL_ROWS) & (cols[None, :] < 2432)
    accum += tl.load(pos_ptr + patch[:, None] * 2432 + cols[None, :], mask=mask)
    tl.store(out_ptr + rows[:, None] * 2432 + cols[None, :], accum, mask=mask)


def _patch_embed_sequential(
    hidden, weight, bias, pos, bias_first=False, bm=16, bn=16, warps=4, order=0
):
    total_rows = hidden.shape[0] * 4096
    out = torch.empty(
        (hidden.shape[0], 4096, 2432), device=hidden.device, dtype=hidden.dtype
    )
    grid = (triton.cdiv(total_rows, bm), triton.cdiv(2432, bn))
    _patch_embed_sequential_kernel[grid](
        hidden,
        weight,
        bias,
        pos,
        out,
        TOTAL_ROWS=total_rows,
        BIAS_FIRST=bias_first,
        ORDER=order,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=warps,
    )
    return out


@triton.jit
def _patch_sample_order_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ORDER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    patch_y = rows // 64
    patch_x = rows % 64
    accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ii in range(64):
        if ORDER == 0:
            kk = ii
        elif ORDER == 1:
            kk = (ii % 16) * 4 + ii // 16
        elif ORDER == 2:
            kk = (ii % 4) * 16 + ii // 4
        elif ORDER == 3:
            kk = (ii % 32) * 2 + ii // 32
        elif ORDER == 4:
            kk = (ii % 8) * 8 + ii // 8
        elif ORDER == 5:
            kk = 63 - ii
        elif ORDER == 6:
            kk = ii ^ 1
        else:
            kk = (ii & ~3) + (3 - (ii & 3))
        channel = kk // 4
        kernel_y = (kk % 4) // 2
        kernel_x = kk % 2
        hidden_offs = (
            channel * (128 * 128)
            + (patch_y * 2 + kernel_y) * 128
            + patch_x * 2
            + kernel_x
        )
        xv = tl.load(hidden_ptr + hidden_offs, mask=rows < 256, other=0.0)
        wv = tl.load(weight_ptr + cols * 64 + kk, mask=cols < 512, other=0.0)
        accum += xv[:, None] * wv[None, :]
    accum += tl.load(bias_ptr + cols[None, :], mask=cols[None, :] < 512)
    mask = (rows[:, None] < 256) & (cols[None, :] < 512)
    tl.store(out_ptr + rows[:, None] * 512 + cols[None, :], accum, mask=mask)


def _patch_sample_order(hidden, weight, bias, order):
    out = torch.empty((256, 512), device=hidden.device, dtype=hidden.dtype)
    _patch_sample_order_kernel[(2, 16)](
        hidden,
        weight,
        bias,
        out,
        ORDER=order,
        BLOCK_M=128,
        BLOCK_N=32,
        num_warps=1,
    )
    return out


_condition_stream = None
_context_stream = None


def _get_condition_stream():
    global _condition_stream
    if _condition_stream is None:
        _condition_stream = torch.cuda.Stream()
    return _condition_stream


def _get_context_stream():
    global _context_stream
    if _context_stream is None:
        _context_stream = torch.cuda.Stream()
    return _context_stream


def _conditioning(
    pooled_projections,
    timestep,
    timestep_linear1_weight,
    timestep_linear1_bias,
    timestep_linear2_weight,
    timestep_linear2_bias,
    pooled_linear1_weight,
    pooled_linear1_bias,
    pooled_linear2_weight,
    pooled_linear2_bias,
    freqs,
):
    timestep_sinusoidal = _sincos(timestep, freqs)
    timestep_embed = F.linear(
        timestep_sinusoidal, timestep_linear1_weight, timestep_linear1_bias
    )
    timestep_embed = F.silu(timestep_embed)
    timestep_embed = F.linear(
        timestep_embed, timestep_linear2_weight, timestep_linear2_bias
    )
    pooled_embed = F.linear(
        pooled_projections, pooled_linear1_weight, pooled_linear1_bias
    )
    pooled_embed = F.silu(pooled_embed)
    pooled_embed = F.linear(pooled_embed, pooled_linear2_weight, pooled_linear2_bias)
    return timestep_embed + pooled_embed


@torch.no_grad()
def run(
    hidden_states,
    encoder_hidden_states,
    pooled_projections,
    timestep,
    proj_weight,
    proj_bias,
    pos_embed,
    timestep_linear1_weight,
    timestep_linear1_bias,
    timestep_linear2_weight,
    timestep_linear2_bias,
    pooled_linear1_weight,
    pooled_linear1_bias,
    pooled_linear2_weight,
    pooled_linear2_bias,
    context_embedder_weight,
    context_embedder_bias,
    freqs,
):
    batch = hidden_states.shape[0]
    async_conditioning = 2 <= batch <= 8
    async_context = batch * encoder_hidden_states.shape[1] >= 50000
    if async_conditioning:
        current_stream = torch.cuda.current_stream()
        condition_stream = _get_condition_stream()
        condition_stream.wait_stream(current_stream)
        with torch.cuda.stream(condition_stream):
            temb = _conditioning(
                pooled_projections,
                timestep,
                timestep_linear1_weight,
                timestep_linear1_bias,
                timestep_linear2_weight,
                timestep_linear2_bias,
                pooled_linear1_weight,
                pooled_linear1_bias,
                pooled_linear2_weight,
                pooled_linear2_bias,
                freqs,
            )
    if async_context:
        current_stream = torch.cuda.current_stream()
        context_stream = _get_context_stream()
        context_stream.wait_stream(current_stream)
        with torch.cuda.stream(context_stream):
            output_encoder_hidden_states = F.linear(
                encoder_hidden_states,
                context_embedder_weight,
                context_embedder_bias,
            )

    if batch == 1:
        output_hidden_states = _patch_embed_sequential(
            hidden_states,
            proj_weight,
            proj_bias,
            pos_embed,
            bias_first=False,
            bm=128,
            bn=32,
            warps=1,
        )
    elif batch <= 8:
        output_hidden_states = _patch_embed_sequential(
            hidden_states,
            proj_weight,
            proj_bias,
            pos_embed,
            bias_first=False,
            bm=256,
            bn=32,
            warps=2,
        )
    else:
        patch_embedded = F.conv2d(hidden_states, proj_weight, proj_bias, stride=2)
        output_hidden_states = _transpose_add(patch_embedded, pos_embed)

    if not async_conditioning:
        temb = _conditioning(
            pooled_projections,
            timestep,
            timestep_linear1_weight,
            timestep_linear1_bias,
            timestep_linear2_weight,
            timestep_linear2_bias,
            pooled_linear1_weight,
            pooled_linear1_bias,
            pooled_linear2_weight,
            pooled_linear2_bias,
            freqs,
        )
    if not async_context:
        output_encoder_hidden_states = F.linear(
            encoder_hidden_states, context_embedder_weight, context_embedder_bias
        )
    if async_conditioning:
        current_stream.wait_stream(condition_stream)
    if async_context:
        current_stream.wait_stream(context_stream)
    return output_hidden_states, temb, output_encoder_hidden_states
