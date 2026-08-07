import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    position_ids,
    inv_freq,
    output,
    n_tokens: tl.constexpr,
    attention_scaling,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 64)
    row_mask = rows < n_tokens

    positions = tl.load(
        position_ids + rows[:, None], mask=row_mask[:, None], other=0
    ).to(tl.float32)
    frequencies = tl.load(inv_freq + dims[None, :])
    angles = positions * frequencies

    cosine = tl.cos(angles) * attention_scaling
    sine = tl.sin(angles) * attention_scaling

    # output is [token, head_dim=128, cos/sin=2].  The reference
    # concatenates the 64 frequencies with themselves before stacking.
    offsets = rows[:, None] * 256 + dims[None, :] * 2
    mask = row_mask[:, None]
    tl.store(output + offsets, cosine, mask=mask)
    tl.store(output + offsets + 1, sine, mask=mask)
    tl.store(output + offsets + 128, cosine, mask=mask)
    tl.store(output + offsets + 129, sine, mask=mask)


@triton.jit
def _rope_packed_kernel(
    position_ids,
    inv_freq,
    output,
    n_tokens: tl.constexpr,
    attention_scaling: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 64)
    row_mask = rows < n_tokens

    positions = tl.load(
        position_ids + rows[:, None], mask=row_mask[:, None], other=0
    ).to(tl.float32)
    frequencies = tl.load(inv_freq + dims[None, :])
    angles = positions * frequencies
    cosine = (tl.cos(angles) * attention_scaling).to(tl.bfloat16)
    sine = (tl.sin(angles) * attention_scaling).to(tl.bfloat16)

    cos_bits = cosine.to(tl.uint16, bitcast=True).to(tl.uint32)
    sin_bits = sine.to(tl.uint16, bitcast=True).to(tl.uint32)
    packed = cos_bits | (sin_bits << 16)

    offsets = rows[:, None] * 128 + dims[None, :]
    mask = row_mask[:, None]
    tl.store(output + offsets, packed, mask=mask)
    tl.store(output + offsets + 64, packed, mask=mask)


@triton.jit
def _rope_native_kernel(
    position_ids,
    inv_freq,
    output,
    n_tokens: tl.constexpr,
    attention_scaling: tl.constexpr,
    BLOCK_M: tl.constexpr,
    REDUCE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 64)
    row_mask = rows < n_tokens

    if EVEN_N:
        positions = tl.load(position_ids + rows[:, None]).to(tl.float32)
    else:
        positions = tl.load(
            position_ids + rows[:, None], mask=row_mask[:, None], other=0
        ).to(tl.float32)
    frequencies = tl.load(inv_freq + dims[None, :])
    angles = positions * frequencies

    # CDNA's native trig instructions take turns rather than radians and are
    # most accurate after reduction to [-0.5, 0.5].
    turns = angles * 0.15915494309189535
    if REDUCE:
        turns = turns - tl.floor(turns + 0.5)
    cosine = tl.inline_asm_elementwise(
        "v_cos_f32 $0, $1",
        "=v,v",
        [turns],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    sine = tl.inline_asm_elementwise(
        "v_sin_f32 $0, $1",
        "=v,v",
        [turns],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    cosine = (cosine * attention_scaling).to(tl.bfloat16)
    sine = (sine * attention_scaling).to(tl.bfloat16)

    cos_bits = cosine.to(tl.uint16, bitcast=True).to(tl.uint32)
    sin_bits = sine.to(tl.uint16, bitcast=True).to(tl.uint32)
    packed = cos_bits | (sin_bits << 16)
    offsets = rows[:, None] * 128 + dims[None, :]
    mask = row_mask[:, None]
    if EVEN_N:
        tl.store(output + offsets, packed)
        tl.store(output + offsets + 64, packed)
    else:
        tl.store(output + offsets, packed, mask=mask)
        tl.store(output + offsets + 64, packed, mask=mask)


@triton.jit
def _rope_poly_kernel(
    position_ids,
    inv_freq,
    output,
    n_tokens: tl.constexpr,
    attention_scaling: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 64)
    row_mask = rows < n_tokens
    positions = tl.load(
        position_ids + rows[:, None], mask=row_mask[:, None], other=0
    ).to(tl.float32)
    frequencies = tl.load(inv_freq + dims[None, :])
    angles = positions * frequencies

    # Reduce to [-pi, pi], then reflect into [0, pi/2].  Minimax degree-5
    # sine and degree-4 cosine are comfortably inside the BF16 tolerance.
    periods = tl.floor(angles * 0.15915494309189535 + 0.5)
    reduced = angles - periods * 6.283185307179586
    magnitude = tl.abs(reduced)
    flip_cos = magnitude > 1.5707963267948966
    x = tl.where(flip_cos, 3.141592653589793 - magnitude, magnitude)
    x2 = x * x

    sin_p = 0.00751438
    sin_p = sin_p * x2 - 0.16567307
    sine = x * (sin_p * x2 + 0.99969677)

    cos_p = 0.03679167
    cos_p = cos_p * x2 - 0.49558082
    cosine = cos_p * x2 + 0.99940324

    sine = tl.where(reduced < 0.0, -sine, sine)
    cosine = tl.where(flip_cos, -cosine, cosine)
    cosine = (cosine * attention_scaling).to(tl.bfloat16)
    sine = (sine * attention_scaling).to(tl.bfloat16)

    cos_bits = cosine.to(tl.uint16, bitcast=True).to(tl.uint32)
    sin_bits = sine.to(tl.uint16, bitcast=True).to(tl.uint32)
    packed = cos_bits | (sin_bits << 16)
    offsets = rows[:, None] * 128 + dims[None, :]
    mask = row_mask[:, None]
    tl.store(output + offsets, packed, mask=mask)
    tl.store(output + offsets + 64, packed, mask=mask)


def run(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    batch_size, seq_len = position_ids.shape
    n_tokens = batch_size * seq_len
    output = torch.empty(
        (batch_size, seq_len, 128),
        device=position_ids.device,
        dtype=torch.int32,
    )
    if n_tokens >= 32768:
        block_m = 8
        num_warps = 2
    else:
        block_m = 16
        num_warps = 4
    _rope_native_kernel[(triton.cdiv(n_tokens, block_m),)](
        position_ids,
        inv_freq,
        output,
        n_tokens=n_tokens,
        attention_scaling=attention_scaling,
        BLOCK_M=block_m,
        REDUCE=False,
        EVEN_N=(n_tokens % block_m == 0),
        num_warps=num_warps,
    )
    return output.view(torch.bfloat16).reshape(batch_size, seq_len, 128, 2)
