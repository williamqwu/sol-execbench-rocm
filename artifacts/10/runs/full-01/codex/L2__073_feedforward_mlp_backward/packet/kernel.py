import torch
import triton
import triton.language as tl


_aux_stream = torch.cuda.Stream()


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
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_sub_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gelu_backward_kernel(gia, x, gi, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    xv = tl.load(x + offsets, mask=mask)
    gv = tl.load(gia + offsets, mask=mask)

    x2 = _mul_rn(xv, xv)
    x3 = _mul_rn(x2, xv)
    coeff_x3 = _mul_rn(0.044715, x3)
    inner_term = _add_rn(xv, coeff_x3)
    inner = _mul_rn(0.7978845608028654, inner_term)
    t = tl.extra.hip.libdevice.tanh(inner)
    d_coeff_x = _mul_rn(0.134145, xv)
    d_coeff_x2 = _mul_rn(d_coeff_x, xv)
    d_inner_term = _add_rn(1.0, d_coeff_x2)
    d_inner = _mul_rn(0.7978845608028654, d_inner_term)
    t2 = _mul_rn(t, t)
    sech_squared = _sub_rn(1.0, t2)
    left0 = _add_rn(1.0, t)
    left = _mul_rn(0.5, left0)
    right0 = _mul_rn(0.5, xv)
    right1 = _mul_rn(right0, sech_squared)
    right = _mul_rn(right1, d_inner)
    gelu_grad = _add_rn(left, right)
    tl.store(gi + offsets, _mul_rn(gv, gelu_grad), mask=mask)


@triton.jit
def _gelu_pre_kernel(x, tanh_inner, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    xv = tl.load(x + offsets, mask=mask)
    x2 = _mul_rn(xv, xv)
    x3 = _mul_rn(x2, xv)
    coeff_x3 = _mul_rn(0.044715, x3)
    inner_term = _add_rn(xv, coeff_x3)
    inner = _mul_rn(0.7978845608028654, inner_term)
    tl.store(tanh_inner + offsets, inner, mask=mask)


@triton.jit
def _gelu_post_kernel(gia, x, tanh_inner, gi,
                      n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    xv = tl.load(x + offsets, mask=mask)
    gv = tl.load(gia + offsets, mask=mask)
    t = tl.load(tanh_inner + offsets, mask=mask)
    d_coeff_x = _mul_rn(0.134145, xv)
    d_coeff_x2 = _mul_rn(d_coeff_x, xv)
    d_inner_term = _add_rn(1.0, d_coeff_x2)
    d_inner = _mul_rn(0.7978845608028654, d_inner_term)
    t2 = _mul_rn(t, t)
    sech_squared = _sub_rn(1.0, t2)
    left0 = _add_rn(1.0, t)
    left = _mul_rn(0.5, left0)
    right0 = _mul_rn(0.5, xv)
    right1 = _mul_rn(right0, sech_squared)
    right = _mul_rn(right1, d_inner)
    gelu_grad = _add_rn(left, right)
    tl.store(gi + offsets, _mul_rn(gv, gelu_grad), mask=mask)


@triton.jit
def _gelu_diag_kernel(x, o_x3, o_inner, o_t, o_tfast, o_tdouble, o_di, o_ss, o_gg,
                      n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    xv = tl.load(x + offsets, mask=mask)
    x2 = _mul_rn(xv, xv)
    x3 = _mul_rn(x2, xv)
    coeff_x3 = _mul_rn(0.044715, x3)
    inner_term = _add_rn(xv, coeff_x3)
    inner = _mul_rn(0.7978845608028654, inner_term)
    t = tl.extra.hip.libdevice.tanh(inner)
    tfast = tl.extra.hip.libdevice.fast_tanhf(inner)
    tdouble = tl.extra.hip.libdevice.tanh(inner.to(tl.float64)).to(tl.float32)
    d_coeff_x = _mul_rn(0.134145, xv)
    d_coeff_x2 = _mul_rn(d_coeff_x, xv)
    d_inner_term = _add_rn(1.0, d_coeff_x2)
    d_inner = _mul_rn(0.7978845608028654, d_inner_term)
    t2 = _mul_rn(t, t)
    sech_squared = _sub_rn(1.0, t2)
    left0 = _add_rn(1.0, t)
    left = _mul_rn(0.5, left0)
    right0 = _mul_rn(0.5, xv)
    right1 = _mul_rn(right0, sech_squared)
    right = _mul_rn(right1, d_inner)
    gelu_grad = _add_rn(left, right)
    tl.store(o_x3 + offsets, x3, mask=mask)
    tl.store(o_inner + offsets, inner, mask=mask)
    tl.store(o_t + offsets, t, mask=mask)
    tl.store(o_tfast + offsets, tfast, mask=mask)
    tl.store(o_tdouble + offsets, tdouble, mask=mask)
    tl.store(o_di + offsets, d_inner, mask=mask)
    tl.store(o_ss + offsets, sech_squared, mask=mask)
    tl.store(o_gg + offsets, gelu_grad, mask=mask)


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    weight1,
    bias1,
    weight2,
    bias2,
    intermediate,
    intermediate_activated,
):
    hidden_size = grad_output.shape[-1]
    intermediate_size = intermediate.shape[-1]
    current_stream = torch.cuda.current_stream()
    _aux_stream.wait_stream(current_stream)
    grad_intermediate_activated = torch.matmul(grad_output, weight2)
    with torch.cuda.stream(_aux_stream):
        grad_weight2 = torch.matmul(
            grad_output.reshape(-1, hidden_size).t(),
            intermediate_activated.reshape(-1, intermediate_size),
        )

    grad_intermediate = torch.empty_like(intermediate)
    n_elements = intermediate.numel()
    _gelu_pre_kernel[(triton.cdiv(n_elements, 512),)](
        intermediate,
        grad_intermediate,
        n_elements,
        BLOCK=512,
        num_warps=4,
    )
    grad_intermediate.tanh_()
    _gelu_post_kernel[(triton.cdiv(n_elements, 512),)](
        grad_intermediate_activated,
        intermediate,
        grad_intermediate,
        grad_intermediate,
        n_elements,
        BLOCK=512,
        num_warps=4,
    )

    _aux_stream.wait_stream(current_stream)
    with torch.cuda.stream(_aux_stream):
        grad_weight1 = torch.matmul(
            grad_intermediate.reshape(-1, intermediate_size).t(),
            hidden_states.reshape(-1, hidden_size),
        )
    grad_hidden_states = torch.matmul(grad_intermediate, weight1)
    grad_bias1 = grad_intermediate.sum(dim=(0, 1))
    grad_bias2 = grad_output.sum(dim=(0, 1))
    current_stream.wait_stream(_aux_stream)

    return grad_hidden_states, grad_weight1, grad_bias1, grad_weight2, grad_bias2
