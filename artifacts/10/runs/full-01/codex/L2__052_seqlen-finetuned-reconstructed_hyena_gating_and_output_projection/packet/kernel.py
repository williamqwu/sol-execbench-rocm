import torch
import triton
import triton.language as tl


@triton.jit
def _pack_fft_inputs(
    v_ptr,
    x1_ptr,
    k_ptr,
    fft_inputs_ptr,
    N_GATED: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gated_mask = offsets < N_GATED
    total_mask = offsets < N_TOTAL

    v = tl.load(v_ptr + offsets, mask=gated_mask, other=0.0)
    x1 = tl.load(x1_ptr + offsets, mask=gated_mask, other=0.0)
    gated = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [v, x1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    filter_value = tl.load(
        k_ptr + offsets - N_GATED, mask=total_mask & ~gated_mask, other=0.0
    )
    value = tl.where(gated_mask, gated, filter_value)
    tl.store(fft_inputs_ptr + offsets, value, mask=total_mask)


@triton.jit
def _gate_transpose(
    conv_ptr,
    gated_ptr,
    x0_ptr,
    skip_bias_ptr,
    projected_input_ptr,
    L: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_ELEMENTS

    batch = offsets // (256 * L)
    within_batch = offsets - batch * (256 * L)
    channel = within_batch // L
    position = within_batch - channel * L
    source_offsets = offsets
    conv_offsets = batch * (512 * L) + channel * (2 * L) + position

    conv = tl.load(conv_ptr + conv_offsets, mask=mask)
    gated = tl.load(gated_ptr + source_offsets, mask=mask)
    gate0 = tl.load(x0_ptr + source_offsets, mask=mask)
    skip_bias = tl.load(skip_bias_ptr + channel, mask=mask)
    skip = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [gated, skip_bias],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value = (conv + skip) * gate0
    tl.store(projected_input_ptr + offsets, value, mask=mask)


@torch.no_grad()
def run(v, x0, x1, k, bias, out_proj_weight, out_proj_bias):
    batch, _, seqlen = v.shape
    fft_size = 2 * seqlen

    if batch > 1 or seqlen >= 1000:
        gated_elements = batch * 256 * seqlen
        total_elements = (batch + 1) * 256 * seqlen
        fft_inputs = torch.empty(
            (batch + 1, 256, seqlen), device=v.device, dtype=torch.float32
        )
        _pack_fft_inputs[(triton.cdiv(total_elements, 256),)](
            v,
            x1,
            k,
            fft_inputs,
            N_GATED=gated_elements,
            N_TOTAL=total_elements,
            BLOCK=256,
            num_warps=4,
        )
        gated = fft_inputs[:batch]
        spectra = torch.fft.rfft(fft_inputs, n=fft_size)
        u_f = spectra[:batch]
        k_f = spectra[batch] / fft_size
    else:
        gated = v * x1
        k_f = torch.fft.rfft(k[0], n=fft_size) / fft_size
        u_f = torch.fft.rfft(gated, n=fft_size)
    conv = torch.fft.irfft(
        u_f * k_f.unsqueeze(0), n=fft_size, norm="forward"
    )[..., :seqlen]

    projected_input = torch.empty_like(v)
    n_elements = batch * seqlen * 256
    _gate_transpose[(triton.cdiv(n_elements, 256),)](
        conv,
        gated,
        x0,
        bias,
        projected_input,
        L=seqlen,
        N_ELEMENTS=n_elements,
        BLOCK=256,
        num_warps=4,
    )
    return (
        torch.matmul(projected_input.transpose(1, 2), out_proj_weight.t())
        + out_proj_bias
    )
