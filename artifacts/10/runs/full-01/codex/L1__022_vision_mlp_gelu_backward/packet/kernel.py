import torch
import triton
import triton.language as tl


_AUX_STREAM = torch.cuda.Stream()


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gelu_pre_kernel(x_ptr, tanh_arg_ptr, n_elements: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Use explicitly-rounded instructions to reproduce the
    # separate float32 TensorIterator kernels in the eager reference.
    x_cubed = _mul_rn(_mul_rn(x, x), x)
    inner = _add_rn(x, _mul_rn(0.044715, x_cubed))
    tanh_arg = _mul_rn(0.7978845608028654, inner)
    tl.store(tanh_arg_ptr + offsets, tanh_arg, mask=mask)


@triton.jit
def _gelu_post_kernel(grad_gelu_ptr, x_ptr, tanh_ptr, out_ptr,
                      n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad_gelu = tl.load(grad_gelu_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    tanh_out = tl.load(tanh_ptr + offsets, mask=mask)
    sech_sq = _add_rn(1.0, -_mul_rn(tanh_out, tanh_out))

    x_term = _mul_rn(_mul_rn(0.134145, x), x)
    d_tanh_arg = _mul_rn(0.7978845608028654, _add_rn(1.0, x_term))
    left = _mul_rn(0.5, _add_rn(1.0, tanh_out))
    right = _mul_rn(_mul_rn(_mul_rn(0.5, x), sech_sq), d_tanh_arg)
    gelu_grad = _add_rn(left, right)
    result = _mul_rn(grad_gelu, gelu_grad)
    tl.store(out_ptr + offsets, result, mask=mask)


def _gelu_backward(grad_gelu, x):
    tanh_arg = torch.empty_like(grad_gelu)
    out = grad_gelu
    n_elements = out.numel()
    seq_len = x.shape[0]
    if seq_len <= 131:
        block = 128
    elif seq_len < 768:
        block = 256
    elif seq_len < 1163:
        block = 512
    else:
        block = 1024
    grid = (triton.cdiv(n_elements, block),)
    _gelu_pre_kernel[grid](
        x, tanh_arg, n_elements=n_elements, BLOCK=block, num_warps=4
    )
    tanh_arg.tanh_()
    _gelu_post_kernel[grid](
        grad_gelu, x, tanh_arg, out,
        n_elements=n_elements, BLOCK=block, num_warps=4
    )
    return out


@torch.no_grad()
def _run(
    grad_output,
    hidden_state,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
    fc1_output,
    gelu_output,
):
    current_stream = torch.cuda.current_stream()
    seq_len = grad_output.shape[0]
    parallel = (
        256 <= seq_len < 769
        or 900 <= seq_len < 1100
        or seq_len >= 1900
    )

    if parallel:
        _AUX_STREAM.wait_stream(current_stream)
        with torch.cuda.stream(_AUX_STREAM):
            grad_fc2_weight = grad_output.t().mm(gelu_output)
            grad_fc2_bias = grad_output.sum(dim=0)
    else:
        grad_fc2_bias = grad_output.sum(dim=0)
        grad_fc2_weight = grad_output.t().mm(gelu_output)

    grad_gelu_output = grad_output.mm(fc2_weight)

    grad_fc1_output = _gelu_backward(grad_gelu_output, fc1_output)

    if parallel:
        _AUX_STREAM.wait_stream(current_stream)
        with torch.cuda.stream(_AUX_STREAM):
            grad_fc1_weight = grad_fc1_output.t().mm(hidden_state)
        grad_hidden_state = grad_fc1_output.mm(fc1_weight)
        grad_fc1_bias = grad_fc1_output.sum(dim=0)
        current_stream.wait_stream(_AUX_STREAM)
    else:
        grad_fc1_bias = grad_fc1_output.sum(dim=0)
        grad_fc1_weight = grad_fc1_output.t().mm(hidden_state)
        grad_hidden_state = grad_fc1_output.mm(fc1_weight)
    return (
        grad_hidden_state,
        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias,
    )


run = _run
