import torch
import triton
import triton.language as tl


@triton.jit
def _cos_sin_packed_asm_kernel(
    freqs_ptr,
    packed_ptr,
    n_elements,
    attention_scaling: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN_N:
        x = tl.load(freqs_ptr + offsets)
    else:
        mask = offsets < n_elements
        x = tl.load(freqs_ptr + offsets, mask=mask)
    cos_x = tl.cos(x) * attention_scaling
    sin_x = tl.sin(x) * attention_scaling
    packed = tl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        [cos_x, sin_x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    row = offsets // 64
    col = offsets % 64
    out_offsets = row * 128 + col
    if EVEN_N:
        tl.store(packed_ptr + out_offsets, packed)
        tl.store(packed_ptr + out_offsets + 64, packed)
    else:
        tl.store(packed_ptr + out_offsets, packed, mask=mask)
        tl.store(packed_ptr + out_offsets + 64, packed, mask=mask)


@triton.jit
def _cos_sin_packed_asm_persistent_kernel(
    freqs_ptr,
    packed_ptr,
    n_elements,
    attention_scaling: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    grid_stride = tl.num_programs(0) * BLOCK_SIZE
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        if EVEN_N:
            x = tl.load(freqs_ptr + offsets)
        else:
            mask = offsets < n_elements
            x = tl.load(freqs_ptr + offsets, mask=mask)
        cos_x = tl.cos(x) * attention_scaling
        sin_x = tl.sin(x) * attention_scaling
        packed = tl.inline_asm_elementwise(
            "v_cvt_pk_bf16_f32 $0, $1, $2",
            "=v,v,v",
            [cos_x, sin_x],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        row = offsets // 64
        col = offsets % 64
        out_offsets = row * 128 + col
        if EVEN_N:
            tl.store(packed_ptr + out_offsets, packed)
            tl.store(packed_ptr + out_offsets + 64, packed)
        else:
            tl.store(packed_ptr + out_offsets, packed, mask=mask)
            tl.store(packed_ptr + out_offsets + 64, packed, mask=mask)
        block_start += grid_stride


def run(freqs: torch.Tensor, attention_scaling: float):
    n_elements = freqs.numel()
    out_shape = (*freqs.shape[:-1], 128)
    packed = torch.empty(out_shape, device=freqs.device, dtype=torch.int32)

    if n_elements > 1_000_000:
        block_size = 1024
        grid = min(triton.cdiv(n_elements, block_size), 1408)
        _cos_sin_packed_asm_persistent_kernel[(grid,)](
            freqs,
            packed,
            n_elements,
            attention_scaling,
            BLOCK_SIZE=block_size,
            EVEN_N=(n_elements % block_size == 0),
            num_warps=4,
        )
    else:
        if n_elements <= 65_536:
            block_size, num_warps = 128, 2
        elif n_elements < 100_000:
            block_size, num_warps = 512, 2
        elif n_elements <= 150_000:
            block_size, num_warps = 1024, 4
        else:
            block_size, num_warps = 512, 4
        _cos_sin_packed_asm_kernel[(triton.cdiv(n_elements, block_size),)](
            freqs,
            packed,
            n_elements,
            attention_scaling,
            BLOCK_SIZE=block_size,
            EVEN_N=(n_elements % block_size == 0),
            num_warps=num_warps,
        )

    pairs = packed.view(torch.bfloat16).view(*out_shape, 2)
    cos = pairs[..., 0]
    sin = pairs[..., 1]
    return cos, sin
