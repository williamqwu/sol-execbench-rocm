import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@triton.jit
def _swiglu_bwd_kernel(
    grad_swiglu_ptr,
    silu_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad_swiglu = tl.load(grad_swiglu_ptr + offsets, mask=mask)
    silu = tl.load(silu_ptr + offsets, mask=mask)
    gate_value = tl.load(gate_ptr + offsets, mask=mask)
    up_value = tl.load(up_ptr + offsets, mask=mask)

    # The explicit bf16 conversions reproduce the intermediate tensor
    # roundings in the eager reference.
    grad_gate = (grad_swiglu * silu).to(tl.bfloat16)
    up_fp32 = up_value.to(tl.float32)
    sigmoid = tl.sigmoid(up_fp32)
    grad_silu = (sigmoid * (1.0 + up_fp32 * (1.0 - sigmoid))).to(tl.bfloat16)
    grad_times_gate = (grad_swiglu * gate_value).to(tl.bfloat16)
    grad_up = (grad_times_gate * grad_silu).to(tl.bfloat16)
    token = offsets // 14336
    feature = offsets - token * 14336
    tl.store(grad_gate_ptr + token * 28672 + feature, grad_gate, mask=mask)
    tl.store(grad_up_ptr + token * 28672 + 14336 + feature, grad_up, mask=mask)


@triton.jit
def _softmax_bwd_kernel(
    grad_weights_ptr,
    weights_ptr,
    grad_logits_ptr,
    n_cols,
    INV_SQRT_HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    grad = tl.load(grad_weights_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weights_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dot = tl.sum(grad * weight, axis=0)
    out = weight * (grad - dot) * INV_SQRT_HEAD_DIM
    tl.store(grad_logits_ptr + offsets, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _rope_q_bwd_kernel(
    grad_rotated_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_elements,
    seq_len,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    feature = offsets % 5120
    token = offsets // 5120
    batch = token // seq_len
    seq = token - batch * seq_len
    head = feature // 160
    dim = feature - head * 160
    other_dim = tl.where(dim < 80, dim + 80, dim - 80)
    input_base = (batch * 32 + head) * seq_len * 160 + seq * 160
    trig_offset = (batch * seq_len + seq) * 160 + dim
    grad = tl.load(grad_rotated_ptr + input_base + dim, mask=mask)
    grad_other = tl.load(grad_rotated_ptr + input_base + other_dim, mask=mask)
    cos_value = tl.load(cos_ptr + trig_offset, mask=mask)
    sin_value = tl.load(sin_ptr + trig_offset, mask=mask)
    rotated = tl.where(dim < 80, -grad_other, grad_other)
    first = (grad * cos_value).to(tl.bfloat16)
    second = (rotated * (-sin_value)).to(tl.bfloat16)
    out = (first + second).to(tl.bfloat16)
    tl.store(out_ptr + token * 7680 + feature, out, mask=mask)


@triton.jit
def _rope_kv_bwd_kernel(
    grad_key_repeated_ptr,
    grad_value_repeated_ptr,
    cos_ptr,
    sin_ptr,
    grad_key_ptr,
    grad_value_ptr,
    n_elements,
    seq_len,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    feature = offsets % 1280
    token = offsets // 1280
    batch = token // seq_len
    seq = token - batch * seq_len
    kv_head = feature // 160
    dim = feature - kv_head * 160
    other_dim = tl.where(dim < 80, dim + 80, dim - 80)
    head0 = kv_head * 4
    base0 = (batch * 32 + head0) * seq_len * 160 + seq * 160
    head_stride = seq_len * 160

    key = (
        tl.load(grad_key_repeated_ptr + base0 + dim, mask=mask).to(tl.float32)
        + tl.load(grad_key_repeated_ptr + base0 + head_stride + dim, mask=mask).to(tl.float32)
        + tl.load(grad_key_repeated_ptr + base0 + 2 * head_stride + dim, mask=mask).to(tl.float32)
        + tl.load(grad_key_repeated_ptr + base0 + 3 * head_stride + dim, mask=mask).to(tl.float32)
    ).to(tl.bfloat16)
    key_other = (
        tl.load(grad_key_repeated_ptr + base0 + other_dim, mask=mask).to(tl.float32)
        + tl.load(grad_key_repeated_ptr + base0 + head_stride + other_dim, mask=mask).to(tl.float32)
        + tl.load(grad_key_repeated_ptr + base0 + 2 * head_stride + other_dim, mask=mask).to(tl.float32)
        + tl.load(grad_key_repeated_ptr + base0 + 3 * head_stride + other_dim, mask=mask).to(tl.float32)
    ).to(tl.bfloat16)
    trig_offset = (batch * seq_len + seq) * 160 + dim
    cos_value = tl.load(cos_ptr + trig_offset, mask=mask)
    sin_value = tl.load(sin_ptr + trig_offset, mask=mask)
    rotated = tl.where(dim < 80, -key_other, key_other)
    first = (key * cos_value).to(tl.bfloat16)
    second = (rotated * (-sin_value)).to(tl.bfloat16)
    key_out = (first + second).to(tl.bfloat16)
    tl.store(grad_key_ptr + token * 7680 + 5120 + feature, key_out, mask=mask)

    value = (
        tl.load(grad_value_repeated_ptr + base0 + dim, mask=mask).to(tl.float32)
        + tl.load(grad_value_repeated_ptr + base0 + head_stride + dim, mask=mask).to(tl.float32)
        + tl.load(grad_value_repeated_ptr + base0 + 2 * head_stride + dim, mask=mask).to(tl.float32)
        + tl.load(grad_value_repeated_ptr + base0 + 3 * head_stride + dim, mask=mask).to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(grad_value_ptr + token * 7680 + 6400 + feature, value, mask=mask)


def _swiglu_backward(grad_swiglu, silu, gate, up):
    grad_gate_up = torch.empty(
        (*grad_swiglu.shape[:-1], 28672), device=grad_swiglu.device, dtype=grad_swiglu.dtype
    )
    grad_gate = grad_gate_up[..., :14336]
    grad_up = grad_gate_up[..., 14336:]
    n_elements = grad_swiglu.numel()
    _swiglu_bwd_kernel[(triton.cdiv(n_elements, 256),)](
        grad_swiglu, silu, gate, up, grad_gate_up, grad_gate_up, n_elements, BLOCK=256
    )
    return grad_gate, grad_up, grad_gate_up


def _softmax_backward(grad_weights, weights, seq_len):
    out = torch.empty_like(grad_weights)
    rows = grad_weights.numel() // seq_len
    block = triton.next_power_of_2(seq_len)
    warps = 4 if block <= 256 else 8
    _softmax_bwd_kernel[(rows,)](
        grad_weights,
        weights,
        out,
        seq_len,
        INV_SQRT_HEAD_DIM=1.0 / math.sqrt(160.0),
        BLOCK=block,
        num_warps=warps,
    )
    return out


def _rope_gqa_backward(grad_q_rotated, grad_k_repeated, grad_v_repeated, cos, sin, batch, seq_len):
    grad_qkv = torch.empty((batch, seq_len, 7680), device=grad_q_rotated.device, dtype=grad_q_rotated.dtype)
    grad_q = grad_qkv[..., :5120]
    grad_k = grad_qkv[..., 5120:6400]
    grad_v = grad_qkv[..., 6400:]
    q_elements = batch * seq_len * 5120
    kv_elements = batch * seq_len * 1280
    _rope_q_bwd_kernel[(triton.cdiv(q_elements, 256),)](
        grad_q_rotated, cos, sin, grad_qkv, q_elements, seq_len, BLOCK=256
    )
    _rope_kv_bwd_kernel[(triton.cdiv(kv_elements, 256),)](
        grad_k_repeated,
        grad_v_repeated,
        cos,
        sin,
        grad_qkv,
        grad_qkv,
        kv_elements,
        seq_len,
        BLOCK=256,
    )
    return grad_q, grad_k, grad_v, grad_qkv


@torch.compile(fullgraph=True, dynamic=True)
def _rmsnorm_backward(grad_input, residual, weight, variance, normalized, residual_grad, eps):
    hidden_size = grad_input.shape[-1]
    grad_fp32 = grad_input.float()
    grad_weight = (grad_fp32 * normalized).sum(dim=(0, 1))
    rsqrt_var = torch.rsqrt(variance + eps)
    grad_normalized = grad_fp32 * weight.float()
    grad_hidden = grad_normalized * rsqrt_var
    grad_var = (
        -0.5
        * (grad_normalized * residual.float()).sum(dim=-1, keepdim=True)
        * rsqrt_var.pow(3)
    )
    grad_hidden = grad_hidden + (2.0 / hidden_size) * residual.float() * grad_var
    return residual_grad + grad_hidden.to(residual.dtype), grad_weight


@torch.compile(fullgraph=True, dynamic=True)
def _norm_weight_gradient(grad_fp32, normalized):
    return (grad_fp32 * normalized).sum(dim=(0, 1))


@torch.compile(fullgraph=True, dynamic=True)
def _normalized_gradient(grad_fp32, weight):
    return grad_fp32 * weight.float()


@torch.compile(fullgraph=True, dynamic=True)
def _variance_gradient(grad_normalized, residual, rsqrt_var):
    return (
        -0.5
        * (grad_normalized * residual.float()).sum(dim=-1, keepdim=True)
        * rsqrt_var.pow(3)
    )


@torch.no_grad()
def run(
    grad_output,
    residual,
    attn_input,
    query_states,
    key_states,
    value_states,
    query_states_rotated,
    key_states_rotated,
    key_states_repeated,
    value_states_repeated,
    cos,
    sin,
    attn_weights,
    attn_output,
    residual2,
    ffn_input,
    gate,
    up,
    silu_up,
    swiglu_output,
    input_ln_weight,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    post_attn_ln_weight,
    gate_weight,
    up_weight,
    down_weight,
    variance1,
    variance2,
    hidden_states_normalized1,
    hidden_states_normalized2,
    eps,
):
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    num_kv_heads = 8
    head_dim = 160
    intermediate_size = 14336
    groups = num_heads // num_kv_heads

    # FFN backward.
    grad_swiglu_output = F.linear(grad_output, down_weight.t())
    grad_down_weight = grad_output.reshape(-1, hidden_size).t() @ swiglu_output.reshape(-1, intermediate_size)

    grad_gate, grad_up, grad_gate_up = _swiglu_backward(grad_swiglu_output, silu_up, gate, up)

    grad_ffn_input_gate = F.linear(grad_gate, gate_weight.t())
    grad_ffn_input_up = F.linear(grad_up, up_weight.t())
    grad_ffn_input = grad_ffn_input_gate + grad_ffn_input_up
    grad_gate_up_weight = grad_gate_up.reshape(-1, 2 * intermediate_size).t() @ ffn_input.reshape(-1, hidden_size)
    grad_gate_weight = grad_gate_up_weight[:intermediate_size]
    grad_up_weight = grad_gate_up_weight[intermediate_size:]

    grad_ffn_input_fp32 = grad_ffn_input.float()
    grad_post_attn_ln_weight = _norm_weight_gradient(grad_ffn_input_fp32, hidden_states_normalized2)
    rsqrt_var2 = torch.rsqrt(variance2 + eps)
    grad_normalized2 = _normalized_gradient(grad_ffn_input_fp32, post_attn_ln_weight)
    grad_hidden_states2 = grad_normalized2 * rsqrt_var2
    # The very small reductions are both cheap and numerically sensitive once
    # their result is propagated through the complete attention backward.
    if batch_size * seq_len <= 256:
        grad_var2 = (
            -0.5
            * (grad_normalized2 * residual2.float()).sum(dim=-1, keepdim=True)
            * rsqrt_var2.pow(3)
        )
    else:
        grad_var2 = _variance_gradient(grad_normalized2, residual2, rsqrt_var2)
    grad_hidden_states2 = grad_hidden_states2 + (2.0 / hidden_size) * residual2.float() * grad_var2
    grad_hidden_states_attn = grad_output + grad_hidden_states2.to(residual2.dtype)

    # Attention backward.
    grad_attn_output = F.linear(grad_hidden_states_attn, o_weight.t())
    grad_o_weight = grad_hidden_states_attn.reshape(-1, hidden_size).t() @ attn_output.reshape(-1, num_heads * head_dim)
    grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    grad_attn_weights = grad_attn_output @ value_states_repeated.transpose(2, 3)
    grad_value_states_repeated = attn_weights.transpose(2, 3) @ grad_attn_output
    grad_attn_logits = _softmax_backward(grad_attn_weights, attn_weights, seq_len)

    grad_query_states_rotated = grad_attn_logits @ key_states_repeated
    grad_key_states_repeated = grad_attn_logits.transpose(2, 3) @ query_states_rotated
    grad_query_states, grad_key_states, grad_value_states, grad_qkv = _rope_gqa_backward(
        grad_query_states_rotated,
        grad_key_states_repeated,
        grad_value_states_repeated,
        cos,
        sin,
        batch_size,
        seq_len,
    )

    grad_attn_input_q = F.linear(grad_query_states, q_weight.t())
    grad_attn_input_k = F.linear(grad_key_states, k_weight.t())
    grad_attn_input_v = F.linear(grad_value_states, v_weight.t())
    grad_attn_input = grad_attn_input_q + grad_attn_input_k + grad_attn_input_v
    grad_qkv_weight = grad_qkv.reshape(-1, 7680).t() @ attn_input.reshape(-1, hidden_size)
    grad_q_weight = grad_qkv_weight[:5120]
    grad_k_weight = grad_qkv_weight[5120:6400]
    grad_v_weight = grad_qkv_weight[6400:]

    grad_input, grad_input_ln_weight = _rmsnorm_backward(
        grad_attn_input,
        residual,
        input_ln_weight,
        variance1,
        hidden_states_normalized1,
        grad_hidden_states_attn,
        eps,
    )

    return (
        grad_input,
        grad_input_ln_weight,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_o_weight,
        grad_post_attn_ln_weight,
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
    )
