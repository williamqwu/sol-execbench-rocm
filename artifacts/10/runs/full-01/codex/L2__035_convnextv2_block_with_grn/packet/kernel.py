import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _dwconv_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BIAS_FIRST: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    ow = offsets % width
    oh = (offsets // width) % height
    channel = (offsets // (height * width)) % channels
    batch = offsets // (channels * height * width)
    if BIAS_FIRST:
        accumulator = tl.load(bias_ptr + channel, mask=mask)
    else:
        accumulator = tl.zeros((BLOCK,), tl.float32)
    for index in tl.static_range(0, 49):
        tap = 48 - index if REVERSE else index
        ky = tap // 7
        kx = tap % 7
        ih = oh + ky - 3
        iw = ow + kx - 3
        valid = mask & (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
        input_offsets = ((batch * channels + channel) * height + ih) * width + iw
        value = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)
        weight = tl.load(weight_ptr + channel * 49 + tap, mask=mask)
        accumulator = tl.inline_asm_elementwise(
            "v_fma_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            [value, weight, accumulator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    if not BIAS_FIRST:
        bias = tl.load(bias_ptr + channel, mask=mask)
        accumulator = tl.inline_asm_elementwise(
            "v_add_f32 $0, $1, $2", "=v,v,v", [accumulator, bias],
            dtype=tl.float32, is_pure=True, pack=1,
        )
    tl.store(output_ptr + offsets, accumulator, mask=mask)


@triton.jit
def _grn_kernel(
    input_ptr,
    norm_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    expanded_dim: tl.constexpr,
    spatial: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    channels = offsets % expanded_dim
    batches = offsets // (spatial * expanded_dim)

    x = tl.load(input_ptr + offsets, mask=mask)
    norm = tl.load(norm_ptr + batches * expanded_dim + channels, mask=mask)
    weight = tl.load(weight_ptr + channels, mask=mask)
    bias = tl.load(bias_ptr + channels, mask=mask)

    value = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [x, norm],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [weight, value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [value, bias],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [value, x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def _grn_2d_kernel(
    input_ptr,
    norm_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    expanded_dim: tl.constexpr,
    spatial: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_block = tl.program_id(0)
    batch_spatial_block = tl.program_id(1)
    spatial_blocks = tl.cdiv(spatial, BLOCK_S)
    batch = batch_spatial_block // spatial_blocks
    spatial_block = batch_spatial_block % spatial_blocks
    s = spatial_block * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    c = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (s < spatial) & (c < expanded_dim)
    offsets = (batch * spatial + s) * expanded_dim + c

    x = tl.load(input_ptr + offsets, mask=mask)
    norm = tl.load(norm_ptr + batch * expanded_dim + c, mask=c < expanded_dim)
    weight = tl.load(weight_ptr + c, mask=c < expanded_dim)
    bias = tl.load(bias_ptr + c, mask=c < expanded_dim)

    value = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [x, norm],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [weight, value],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [value, bias],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [value, x],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def _grn_fused_norm_kernel(
    input_ptr,
    global_ptr,
    mean_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    expanded_dim: tl.constexpr,
    spatial: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_block = tl.program_id(0)
    batch_spatial_block = tl.program_id(1)
    spatial_blocks = tl.cdiv(spatial, BLOCK_S)
    batch = batch_spatial_block // spatial_blocks
    spatial_block = batch_spatial_block % spatial_blocks
    s = spatial_block * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    c = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    channel_mask = c < expanded_dim
    mask = (s < spatial) & channel_mask
    offsets = (batch * spatial + s) * expanded_dim + c

    x = tl.load(input_ptr + offsets, mask=mask)
    global_value = tl.load(
        global_ptr + batch * expanded_dim + c, mask=channel_mask
    )
    mean = tl.load(mean_ptr + batch)
    denominator = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [mean, eps],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    norm = global_value / denominator
    weight = tl.load(weight_ptr + c, mask=channel_mask)
    bias = tl.load(bias_ptr + c, mask=channel_mask)

    value = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [x, norm],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [weight, value],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [value, bias],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [value, x],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def _normalize_kernel(
    global_ptr,
    mean_ptr,
    output_ptr,
    eps,
    n_elements: tl.constexpr,
    expanded_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(global_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + offsets // expanded_dim, mask=mask)
    denominator = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [mean, eps],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    value = value / denominator
    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def _residual_kernel(
    projection_ptr,
    residual_ptr,
    output_ptr,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    block_s = tl.program_id(0)
    block_c = tl.program_id(1)
    batch = tl.program_id(2)
    s = block_s * BLOCK_S + tl.arange(0, BLOCK_S)
    c = block_c * BLOCK_C + tl.arange(0, BLOCK_C)

    projection_offsets = (
        batch * spatial * channels + s[:, None] * channels + c[None, :]
    )
    projection_mask = (s[:, None] < spatial) & (c[None, :] < channels)
    projection = tl.load(projection_ptr + projection_offsets, mask=projection_mask)
    projection = tl.trans(projection)

    output_offsets = batch * channels * spatial + c[:, None] * spatial + s[None, :]
    output_mask = (c[:, None] < channels) & (s[None, :] < spatial)
    residual = tl.load(residual_ptr + output_offsets, mask=output_mask)
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2", "=v,v,v", [residual, projection],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(output_ptr + output_offsets, value, mask=output_mask)


@triton.jit
def _nchw_to_nhwc_kernel(
    input_ptr,
    output_ptr,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    block_s = tl.program_id(0)
    block_c = tl.program_id(1)
    batch = tl.program_id(2)
    s = block_s * BLOCK_S + tl.arange(0, BLOCK_S)
    c = block_c * BLOCK_C + tl.arange(0, BLOCK_C)
    input_offsets = batch * channels * spatial + c[:, None] * spatial + s[None, :]
    mask = (c[:, None] < channels) & (s[None, :] < spatial)
    value = tl.load(input_ptr + input_offsets, mask=mask)
    value = tl.trans(value)
    output_offsets = batch * spatial * channels + s[:, None] * channels + c[None, :]
    tl.store(output_ptr + output_offsets, value, mask=tl.trans(mask))


@torch.no_grad()
def run(
    x,
    dwconv_weight,
    dwconv_bias,
    layernorm_weight,
    layernorm_bias,
    pwconv1_weight,
    pwconv1_bias,
    grn_weight,
    grn_bias,
    pwconv2_weight,
    pwconv2_bias,
    eps,
    layer_norm_eps,
):
    residual = x
    channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3]
    out = F.conv2d(x, dwconv_weight, dwconv_bias, padding=3, groups=channels)
    if x.numel() >= 300_000 and (x.shape[0] > 1 or x.numel() >= 400_000):
        transposed = torch.empty(
            (x.shape[0], x.shape[2], x.shape[3], channels),
            device=x.device,
            dtype=x.dtype,
        )
        if channels >= 256:
            transpose_block, transpose_warps = 32, 4
        else:
            transpose_block, transpose_warps = 64, 8
        _nchw_to_nhwc_kernel[
            (
                triton.cdiv(spatial, transpose_block),
                triton.cdiv(channels, transpose_block),
                x.shape[0],
            )
        ](
            out,
            transposed,
            channels,
            spatial,
            BLOCK_S=transpose_block,
            BLOCK_C=transpose_block,
            num_warps=transpose_warps,
        )
        out = transposed
    else:
        out = out.permute(0, 2, 3, 1)
    out = F.layer_norm(
        out, (channels,), layernorm_weight, layernorm_bias, eps=layer_norm_eps
    )
    out = F.linear(out, pwconv1_weight, pwconv1_bias)
    out = F.gelu(out)
    global_features = torch.linalg.vector_norm(out, ord=2, dim=(1, 2), keepdim=True)
    feature_means = global_features.mean(dim=-1, keepdim=True)
    expanded_dim = out.shape[-1]
    n_elements = out.numel()
    grn_out = torch.empty_like(out)
    if n_elements > 100_000_000:
        norm_elements = global_features.numel()
        _normalize_kernel[(triton.cdiv(norm_elements, 1024),)](
            global_features,
            feature_means,
            global_features,
            eps,
            norm_elements,
            expanded_dim,
            BLOCK=1024,
            num_warps=4,
        )
        _grn_kernel[(triton.cdiv(n_elements, 2048),)](
            out,
            global_features,
            grn_weight,
            grn_bias,
            grn_out,
            n_elements,
            expanded_dim,
            spatial,
            BLOCK=2048,
            num_warps=4,
        )
    else:
        if n_elements > 20_000_000:
            block_s, block_c, grn_warps = 32, 128, 8
        elif spatial <= 64:
            if x.shape[0] >= 32:
                block_s, block_c, grn_warps = 32, 256, 8
            else:
                block_s, block_c, grn_warps = 32, 128, 4
        elif expanded_dim == 384 and x.shape[0] >= 4:
            block_s, block_c, grn_warps = 32, 128, 8
        elif expanded_dim == 1536 and x.shape[0] < 16:
            block_s, block_c, grn_warps = 16, 128, 4
        elif expanded_dim == 768 and x.shape[0] < 8:
            block_s, block_c, grn_warps = 16, 128, 4
        elif expanded_dim % 256 == 0 and spatial >= 1000:
            block_s, block_c, grn_warps = 32, 256, 8
        elif expanded_dim % 256 == 0:
            block_s, block_c, grn_warps = 16, 256, 8
        else:
            block_s, block_c, grn_warps = 16, 128, 4
        _grn_fused_norm_kernel[
            (
                triton.cdiv(expanded_dim, block_c),
                x.shape[0] * triton.cdiv(spatial, block_s),
            )
        ](
            out,
            global_features,
            feature_means,
            grn_weight,
            grn_bias,
            grn_out,
            eps,
            expanded_dim,
            spatial,
            BLOCK_S=block_s,
            BLOCK_C=block_c,
            num_warps=grn_warps,
        )
    out = grn_out
    out = F.linear(out, pwconv2_weight, pwconv2_bias)
    output_elements = x.numel()
    if x.shape[0] == 1 or output_elements < 200_000:
        return residual + out.permute(0, 3, 1, 2)
    result = torch.empty_like(x)
    if output_elements <= 2_000_000:
        block = 32
        num_warps = 4
    elif channels <= 128:
        block = 64
        num_warps = 8
    else:
        block = 64
        num_warps = 4
    _residual_kernel[
        (triton.cdiv(spatial, block), triton.cdiv(channels, block), x.shape[0])
    ](
        out,
        residual,
        result,
        channels,
        spatial,
        BLOCK_S=block,
        BLOCK_C=block,
        num_warps=num_warps,
    )
    return result
