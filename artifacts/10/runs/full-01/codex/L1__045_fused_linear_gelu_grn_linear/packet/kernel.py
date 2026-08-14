import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _add_f32(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_f32(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _post_grn_kernel(
    x_ptr,
    global_ptr,
    mean_ptr,
    weight_ptr,
    bias_ptr,
    n_spatial: tl.constexpr,
    batch_stride: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    num_spatial_blocks: tl.constexpr = tl.cdiv(n_spatial, BLOCK_M)
    num_channel_blocks: tl.constexpr = tl.cdiv(512, BLOCK_C)
    pid = tl.program_id(0)
    channel_block = pid % num_channel_blocks
    spatial_block = (pid // num_channel_blocks) % num_spatial_blocks
    batch = pid // (num_channel_blocks * num_spatial_blocks)

    spatial = spatial_block * BLOCK_M + tl.arange(0, BLOCK_M)
    channel = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    offsets = (
        batch * batch_stride
        + spatial[:, None] * 512
        + channel[None, :]
    )
    mask = spatial[:, None] < n_spatial

    x = tl.load(x_ptr + offsets, mask=mask)
    g = tl.load(global_ptr + batch * 512 + channel)
    mean = tl.load(mean_ptr + batch)
    weight = tl.load(weight_ptr + channel)
    bias = tl.load(bias_ptr + channel)

    denominator = _add_f32(mean, eps)
    norm = g / denominator
    value = _mul_f32(x, norm)
    value = _mul_f32(weight, value)
    value = _add_f32(value, bias)
    value = _add_f32(value, x)
    tl.store(x_ptr + offsets, value, mask=mask)


@torch.no_grad()
def run(
    hidden_states,
    pwconv1_weight,
    pwconv1_bias,
    grn_weight,
    grn_bias,
    pwconv2_weight,
    pwconv2_bias,
    eps,
):
    x = F.linear(hidden_states, pwconv1_weight, pwconv1_bias)
    x = F.gelu(x)
    global_features = torch.linalg.vector_norm(
        x, ord=2, dim=(1, 2), keepdim=True
    )
    mean_features = global_features.mean(dim=-1, keepdim=True)

    batch_size = x.shape[0]
    n_spatial = x.shape[1] * x.shape[2]
    batch_stride = x.shape[1] * x.shape[2] * 512
    block_m = 4
    block_c = 256
    grid = batch_size * triton.cdiv(n_spatial, block_m) * triton.cdiv(512, block_c)
    _post_grn_kernel[(grid,)](
        x,
        global_features,
        mean_features,
        grn_weight,
        grn_bias,
        n_spatial=n_spatial,
        batch_stride=batch_stride,
        eps=eps,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return F.linear(x, pwconv2_weight, pwconv2_bias)
