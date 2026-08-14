import torch
import torch.nn.functional as F
import triton
import triton.language as tl


HIDDEN = 1280
NORM_BLOCK = 2048


@triton.jit
def _layer_norm_affine(
    x_ptr, weight_ptr, bias_ptr, out_ptr, n_rows: tl.constexpr, eps,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(x_ptr + row * N + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    norm = ((x - mean) * tl.rsqrt(var + eps)).to(tl.bfloat16)
    weight = tl.load(weight_ptr + col, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col, mask=mask, other=0.0)
    product = (norm * weight).to(tl.bfloat16)
    y = (product + bias).to(tl.bfloat16)
    tl.store(out_ptr + row * N + col, y, mask=mask)


@triton.jit
def _gated_residual_norm_affine(
    residual_ptr, projected_ptr, gate_ptr, weight_ptr, bias_ptr,
    saved_residual_ptr, normed_ptr, eps,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    residual = tl.load(residual_ptr + row * N + col, mask=mask, other=0.0)
    projected = tl.load(projected_ptr + row * N + col, mask=mask, other=0.0)
    gate = tl.load(gate_ptr)
    scaled = (projected * gate).to(tl.bfloat16)
    x_bf16 = (residual + scaled).to(tl.bfloat16)
    tl.store(saved_residual_ptr + row * N + col, x_bf16, mask=mask)

    x = x_bf16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    norm = ((x - mean) * tl.rsqrt(var + eps)).to(tl.bfloat16)
    weight = tl.load(weight_ptr + col, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col, mask=mask, other=0.0)
    product = (norm * weight).to(tl.bfloat16)
    y = (product + bias).to(tl.bfloat16)
    tl.store(normed_ptr + row * N + col, y, mask=mask)


@triton.jit
def _bias_gelu(x_ptr, bias_ptr, out_ptr, n_elements: tl.constexpr,
               WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets % WIDTH, mask=mask, other=0.0)
    z = (x + bias).to(tl.bfloat16).to(tl.float32)
    y = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))
    tl.store(out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _final_gated_residual(
    projected_ptr, bias_ptr, residual_ptr, gate_ptr, out_ptr,
    n_elements: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    projected = tl.load(projected_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets % WIDTH, mask=mask, other=0.0)
    hidden = (projected + bias).to(tl.bfloat16)
    gate = tl.load(gate_ptr)
    scaled = (hidden * gate).to(tl.bfloat16)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    out = (residual + scaled).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.no_grad()
def run(
    hidden_state,
    input_layernorm_weight,
    input_layernorm_bias,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    post_attention_layernorm_weight,
    post_attention_layernorm_bias,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
    gate_attn,
    gate_ffn,
    norm_eps,
):
    batch_size, seq_len, _ = hidden_state.shape
    rows = batch_size * seq_len
    residual0 = hidden_state.view(rows, HIDDEN)
    # Compute these with PyTorch to exactly retain the reference's bf16 tanh.
    gate_attn_activated = torch.tanh(gate_attn)
    gate_ffn_activated = torch.tanh(gate_ffn)

    normed = torch.empty_like(residual0)
    _layer_norm_affine[(rows,)](
        residual0, input_layernorm_weight, input_layernorm_bias, normed,
        rows, norm_eps, N=HIDDEN, BLOCK=NORM_BLOCK, num_warps=8,
    )

    query = torch.mm(normed, q_proj_weight.t())
    key = torch.mm(normed, k_proj_weight.t())
    value = torch.mm(normed, v_proj_weight.t())
    query = query.view(batch_size, seq_len, 16, 80).transpose(1, 2)
    key = key.view(batch_size, seq_len, 16, 80).transpose(1, 2)
    value = value.view(batch_size, seq_len, 16, 80).transpose(1, 2)
    attention = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, scale=(80 ** -0.5)
    )
    attention = attention.transpose(1, 2).contiguous().view(rows, HIDDEN)
    attention = torch.mm(attention, o_proj_weight.t())

    residual1 = torch.empty_like(residual0)
    post_normed = torch.empty_like(residual0)
    _gated_residual_norm_affine[(rows,)](
        residual0, attention, gate_attn_activated,
        post_attention_layernorm_weight, post_attention_layernorm_bias,
        residual1, post_normed, norm_eps,
        N=HIDDEN, BLOCK=NORM_BLOCK, num_warps=8,
    )

    mlp = torch.mm(post_normed, fc1_weight.t())
    activated = torch.empty_like(mlp)
    n_mlp = rows * 5120
    _bias_gelu[(triton.cdiv(n_mlp, 256),)](
        mlp, fc1_bias, activated, n_mlp,
        WIDTH=5120, BLOCK=256, num_warps=4,
    )
    mlp = torch.mm(activated, fc2_weight.t())

    output = torch.empty_like(residual1)
    n_out = rows * HIDDEN
    _final_gated_residual[(triton.cdiv(n_out, 256),)](
        mlp, fc2_bias, residual1, gate_ffn_activated, output, n_out,
        WIDTH=HIDDEN, BLOCK=256, num_warps=4,
    )
    return output.view(batch_size, seq_len, HIDDEN)
