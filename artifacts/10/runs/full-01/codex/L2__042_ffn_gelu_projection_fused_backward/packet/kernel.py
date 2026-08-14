import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fmul(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fadd(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fsub(a, b):
    return tl.inline_asm_elementwise(
        "v_sub_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _gelu_backward_kernel(grad_ptr, x_ptr, out_ptr, n_elements: tl.constexpr,
                          BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    x2 = _fmul(x, x)
    x3 = _fmul(x2, x)
    inner = _fmul(x3, 0.044715)
    inner = _fadd(x, inner)
    inner = _fmul(inner, 0.7978845608028654)
    tanh_out = libdevice.tanh(inner)

    dtanh = _fmul(x, 0.134145)
    dtanh = _fmul(dtanh, x)
    dtanh = _fadd(1.0, dtanh)
    dtanh = _fmul(dtanh, 0.7978845608028654)
    tanh_sq = _fmul(tanh_out, tanh_out)
    sech_sq = _fsub(1.0, tanh_sq)
    first = _fadd(1.0, tanh_out)
    first = _fmul(0.5, first)
    second = _fmul(0.5, x)
    second = _fmul(second, sech_sq)
    second = _fmul(second, dtanh)
    gelu_grad = _fadd(first, second)
    result = _fmul(grad, gelu_grad)
    tl.store(out_ptr + offsets, result, mask=mask)


def _gelu_backward(grad, x):
    out = torch.empty_like(grad)
    n = grad.numel()
    _gelu_backward_kernel[(triton.cdiv(n, 256),)](
        grad, x, out, n_elements=n, BLOCK=256, num_warps=4
    )
    return out


@triton.jit
def _layernorm_products_kernel(
    grad_ptr,
    norm_ptr,
    weight_ptr,
    grad_norm_ptr,
    grad_norm_norm_ptr,
    grad_out_norm_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask)
    norm = tl.load(norm_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + (offsets % 512), mask=mask)
    grad_norm = _fmul(grad, weight)
    grad_norm_norm = _fmul(grad_norm, norm)
    grad_out_norm = _fmul(grad, norm)
    tl.store(grad_norm_ptr + offsets, grad_norm, mask=mask)
    tl.store(grad_norm_norm_ptr + offsets, grad_norm_norm, mask=mask)
    tl.store(grad_out_norm_ptr + offsets, grad_out_norm, mask=mask)


def _layernorm_products(grad, norm, weight):
    grad_norm = torch.empty_like(grad)
    grad_norm_norm = torch.empty_like(grad)
    grad_out_norm = torch.empty_like(grad)
    n = grad.numel()
    _layernorm_products_kernel[(triton.cdiv(n, 256),)](
        grad,
        norm,
        weight,
        grad_norm,
        grad_norm_norm,
        grad_out_norm,
        n_elements=n,
        BLOCK=256,
        num_warps=4,
    )
    return grad_norm, grad_norm_norm, grad_out_norm


@triton.jit
def _layernorm_finish_kernel(
    grad_norm_ptr,
    norm_ptr,
    mean_ptr,
    norm_mean_ptr,
    inv_std_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    rows = offsets // 512
    grad_norm = tl.load(grad_norm_ptr + offsets, mask=mask)
    norm = tl.load(norm_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + rows, mask=mask)
    norm_mean = tl.load(norm_mean_ptr + rows, mask=mask)
    inv_std = tl.load(inv_std_ptr + rows, mask=mask)
    centered = _fsub(grad_norm, mean)
    norm_term = _fmul(norm, norm_mean)
    result = _fsub(centered, norm_term)
    result = _fmul(inv_std, result)
    tl.store(out_ptr + offsets, result, mask=mask)


def _layernorm_finish(grad_norm, norm, mean, norm_mean, inv_std):
    out = torch.empty_like(grad_norm)
    n = grad_norm.numel()
    _layernorm_finish_kernel[(triton.cdiv(n, 256),)](
        grad_norm,
        norm,
        mean,
        norm_mean,
        inv_std,
        out,
        n_elements=n,
        BLOCK=256,
        num_warps=4,
    )
    return out


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    fc1_weight,
    fc1_output,
    gelu_output,
    fc2_weight,
    residual_output,
    normalized,
    var,
    ln_weight,
    eps,
):
    B, S, H = grad_output.shape
    I = fc1_weight.shape[0]

    grad_normalized, grad_normalized_normalized, grad_output_normalized = (
        _layernorm_products(grad_output, normalized, ln_weight)
    )
    grad_ln_weight = grad_output_normalized.sum(dim=(0, 1))
    grad_ln_bias = grad_output.sum(dim=(0, 1))
    std = torch.sqrt(var + eps)
    grad_normalized_mean = grad_normalized.mean(dim=-1, keepdim=True)
    grad_normalized_normalized_mean = grad_normalized_normalized.mean(
        dim=-1, keepdim=True
    )
    grad_residual = _layernorm_finish(
        grad_normalized,
        normalized,
        grad_normalized_mean,
        grad_normalized_normalized_mean,
        1.0 / std,
    )

    grad_fc2_bias = grad_residual.sum(dim=(0, 1))
    grad_fc2_weight = grad_residual.view(-1, H).t() @ gelu_output.view(-1, I)
    grad_gelu_output = grad_residual @ fc2_weight

    grad_fc1_output = _gelu_backward(grad_gelu_output, fc1_output)

    grad_fc1_bias = grad_fc1_output.sum(dim=(0, 1))
    grad_fc1_weight = grad_fc1_output.view(-1, I).t() @ hidden_states.view(-1, H)
    grad_hidden_states = torch.addmm(
        grad_residual.view(-1, H),
        grad_fc1_output.view(-1, I),
        fc1_weight,
    ).view(B, S, H)

    return (
        grad_hidden_states,
        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
