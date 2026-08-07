import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fp32_mul(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32_add(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32_sub(a, b):
    return tl.inline_asm_elementwise(
        "v_sub_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _make_trig_kernel(
    temporal_freqs, spatial_freqs, cosines, sines,
    N: tl.constexpr, P: tl.constexpr, SIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    axis = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offs // 10
    freq_idx = offs % 10
    patch = token % P
    frame_pos = token // P
    height_pos = patch // SIDE
    width_pos = patch % SIDE
    pos = tl.where(axis == 0, frame_pos, tl.where(axis == 1, height_pos, width_pos))
    tf = tl.load(temporal_freqs + freq_idx, mask=offs < N * 10)
    sf = tl.load(spatial_freqs + freq_idx, mask=offs < N * 10)
    freq = tl.where(axis == 0, tf, sf)
    angle = pos.to(tl.float32) * freq
    out_off = axis * N * 10 + offs
    tl.store(cosines + out_off, tl.cos(angle), mask=offs < N * 10)
    tl.store(sines + out_off, tl.sin(angle), mask=offs < N * 10)


@triton.jit
def _rope_kernel(
    qkv, cosines, sines, qk_out,
    N: tl.constexpr, B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = 2 * B * 16 * N * 64
    valid = offs < total
    d = offs % 64
    t = offs // 64
    token = t % N
    t = t // N
    head = t % 16
    t = t // 16
    batch = t % B
    q_or_k = t // B

    axis = d // 21
    local_d = d - axis * 21
    rotated = (d < 63) & (local_d < 20)
    pair = local_d // 2
    even = (local_d & 1) == 0
    other_d = tl.where(even, d + 1, d - 1)

    base = batch * N * 3072 + token * 3072 + q_or_k * 1024 + head * 64
    x = tl.load(qkv + base + d, mask=valid)
    other = tl.load(qkv + base + other_d, mask=valid & rotated, other=0.0)
    trig_off = axis * N * 10 + token * 10 + pair
    c = tl.load(cosines + trig_off, mask=valid & rotated, other=1.0)
    s = tl.load(sines + trig_off, mask=valid & rotated, other=0.0)
    xc = _fp32_mul(x, c)
    other_s = _fp32_mul(other, s)
    y = tl.where(even, _fp32_sub(xc, other_s), _fp32_add(other_s, xc))
    tl.store(qk_out + offs, tl.where(rotated, y, x), mask=valid)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    temporal_freqs: torch.Tensor,
    spatial_freqs: torch.Tensor,
    scale: float,
):
    batch_size, num_frames, num_patches, hidden_size = hidden_states.shape
    seq_len = num_frames * num_patches
    num_heads = 16
    head_dim = 64

    patches_per_side = math.isqrt(num_patches)
    if patches_per_side * patches_per_side != num_patches:
        patches_per_side += 1

    qkv = F.linear(
        hidden_states.reshape(batch_size, seq_len, hidden_size),
        qkv_weight,
        qkv_bias,
    )
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    cosines = torch.empty(
        (3, seq_len, 10), device=hidden_states.device, dtype=torch.float32
    )
    sines = torch.empty_like(cosines)
    _make_trig_kernel[(triton.cdiv(seq_len * 10, 1024), 3)](
        temporal_freqs,
        spatial_freqs,
        cosines,
        sines,
        N=seq_len,
        P=num_patches,
        SIDE=patches_per_side,
        BLOCK=1024,
    )
    qk = torch.empty(
        (2, batch_size, num_heads, seq_len, head_dim),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    _rope_kernel[(triton.cdiv(qk.numel(), 256),)](
        qkv,
        cosines,
        sines,
        qk,
        N=seq_len,
        B=batch_size,
        BLOCK=256,
    )
    q, k = qk[0], qk[1]
    v = qkv[:, :, 2].permute(0, 2, 1, 3)
    # baddbmm exposes the BLAS alpha parameter.  For the specified scale of
    # 1/8 this is bit-identical to the reference's rounded GEMM followed by a
    # multiply, and avoids another complete pass over the quadratic tensor.
    scores = torch.baddbmm(
        qkv_bias[:1].reshape(1, 1, 1),
        q.reshape(batch_size * num_heads, seq_len, head_dim),
        k.transpose(-2, -1).reshape(batch_size * num_heads, head_dim, seq_len),
        beta=0.0,
        alpha=scale,
    ).reshape(batch_size, num_heads, seq_len, seq_len)
    torch.softmax(scores, dim=-1, out=scores)

    # Reinterpret the dead Q storage as token-major output.  hipBLAS can write
    # its head-batched result through this strided view, avoiding the explicit
    # transpose/contiguous copy made by reshape in the reference.
    attended = qk[0].reshape(batch_size, seq_len, num_heads, head_dim)
    attended_heads = attended.permute(0, 2, 1, 3)
    torch.matmul(scores, v, out=attended_heads)

    # K is also dead after the score GEMM, so its storage is the final result.
    output = qk[1].reshape(batch_size * seq_len, hidden_size)
    torch.addmm(
        out_bias,
        attended.reshape(batch_size * seq_len, hidden_size),
        out_weight.t(),
        out=output,
    )
    return output.reshape(batch_size, num_frames, num_patches, hidden_size)
