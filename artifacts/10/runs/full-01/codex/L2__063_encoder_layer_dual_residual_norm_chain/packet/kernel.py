import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _norm_mul_kernel(x, inv_rms, weight, out, n_tokens: tl.constexpr):
    token = tl.program_id(0)
    d = tl.arange(0, 1024)
    xval = tl.load(x + token * 1024 + d)
    scale = tl.load(inv_rms + token)
    w = tl.load(weight + d)
    # Keep the same two rounded multiplies as the eager expression.
    y = xval * scale
    y = w * y
    tl.store(out + token * 1024 + d, y)


@triton.jit
def _residual_square_kernel(a, b, residual, squared, n_elements: tl.constexpr,
                            BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    value = tl.load(a + offs, mask=mask) + tl.load(b + offs, mask=mask)
    tl.store(residual + offs, value, mask=mask)
    tl.store(squared + offs, value * value, mask=mask)


@triton.jit
def _qkv_rope_repeat_kernel(
    q_proj, k_proj, v_proj, cos, sin, q_out, k_out, v_out,
    seq_len: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // 16
    head = row - token * 16
    kv_head = head // 4
    d = tl.arange(0, 64)
    rd = tl.where(d < 32, d + 32, d - 32)

    q_base = token * 1024 + head * 64
    kv_base = token * 256 + kv_head * 64
    pos_base = token * 64

    q = tl.load(q_proj + q_base + d)
    qr = tl.load(q_proj + q_base + rd)
    k = tl.load(k_proj + kv_base + d)
    kr = tl.load(k_proj + kv_base + rd)
    c = tl.load(cos + pos_base + d)
    s = tl.load(sin + pos_base + d)
    qrot = tl.where(d < 32, -qr, qr)
    krot = tl.where(d < 32, -kr, kr)

    # Materialize standard contiguous [B, H, S, D] for attention.
    batch = token // seq_len
    pos = token - batch * seq_len
    kv_out_base = ((batch * 16 + head) * seq_len + pos) * 64
    tl.store(q_out + kv_out_base + d, q * c + qrot * s)
    tl.store(k_out + kv_out_base + d, k * c + krot * s)
    tl.store(v_out + kv_out_base + d, tl.load(v_proj + kv_base + d))


@triton.jit
def _softmax_inplace_kernel(x, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    z = tl.load(x + row * n_cols + cols, mask=mask, other=-float("inf"))
    z = z * 0.125
    z = z - tl.max(z, axis=0)
    numerator = tl.exp(z)
    denominator = tl.sum(numerator, axis=0)
    tl.store(x + row * n_cols + cols, numerator / denominator, mask=mask)


@triton.jit
def _silu_mul_kernel(gate_up, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    token = offs // 4096
    d = offs - token * 4096
    gate = tl.load(gate_up + token * 8192 + d, mask=mask)
    up = tl.load(gate_up + token * 8192 + 4096 + d, mask=mask)
    silu = gate * tl.sigmoid(gate)
    tl.store(out + offs, up * silu, mask=mask)


def _rms_norm(x, weight, eps):
    shape = x.shape
    flat = x.reshape(-1, 1024)
    variance = flat.pow(2).mean(-1, keepdim=True)
    variance.add_(eps)
    inv_rms = variance.rsqrt_()
    out = torch.empty_like(flat)
    _norm_mul_kernel[(flat.shape[0],)](
        flat, inv_rms, weight, out, flat.shape[0], num_warps=4
    )
    return out.view(shape)


def _rms_norm_from_square(x, squared, weight, eps):
    shape = x.shape
    flat = x.reshape(-1, 1024)
    variance = squared.view(-1, 1024).mean(-1, keepdim=True)
    variance.add_(eps)
    inv_rms = variance.rsqrt_()
    out = torch.empty_like(flat)
    _norm_mul_kernel[(flat.shape[0],)](
        flat, inv_rms, weight, out, flat.shape[0], num_warps=4
    )
    return out.view(shape)


@torch.no_grad()
def run(
    hidden_states,
    cos,
    sin,
    pre_sa_norm_weight,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    post_sa_norm_weight,
    gate_up_proj_weight,
    down_proj_weight,
    norm_eps,
):
    batch_size, seq_len, _ = hidden_states.shape
    n_tokens = batch_size * seq_len

    normed_1 = _rms_norm(hidden_states, pre_sa_norm_weight, norm_eps)

    normed_1_2d = normed_1.view(n_tokens, 1024)
    q_proj = torch.matmul(normed_1_2d, q_proj_weight.t())
    k_proj = torch.matmul(normed_1_2d, k_proj_weight.t())
    v_proj = torch.matmul(normed_1_2d, v_proj_weight.t())

    q = torch.empty(
        (batch_size, 16, seq_len, 64), device=hidden_states.device, dtype=hidden_states.dtype
    )
    k = torch.empty_like(q)
    v = torch.empty_like(k)
    _qkv_rope_repeat_kernel[(n_tokens * 16,)](
        q_proj, k_proj, v_proj, cos, sin, q, k, v,
        seq_len=seq_len, num_warps=1
    )

    attn_weights = torch.matmul(q, k.transpose(2, 3))
    attn_weights.div_(8.0)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = (
        attn_output.transpose(1, 2).contiguous().view(n_tokens, 1024)
    )

    residual_1_2d = hidden_states.view(n_tokens, 1024)
    attn_out = torch.matmul(attn_output, o_proj_weight.t())
    residual_2 = torch.empty_like(residual_1_2d)
    residual_2_squared = torch.empty_like(residual_1_2d)
    n_hidden = n_tokens * 1024
    _residual_square_kernel[(triton.cdiv(n_hidden, 256),)](
        residual_1_2d, attn_out, residual_2, residual_2_squared,
        n_elements=n_hidden, BLOCK=256, num_warps=4,
    )
    residual_2_3d = residual_2.view(batch_size, seq_len, 1024)

    normed_2 = _rms_norm_from_square(
        residual_2_3d, residual_2_squared, post_sa_norm_weight, norm_eps
    )
    gate_up = torch.matmul(
        normed_2.view(n_tokens, 1024), gate_up_proj_weight.t()
    )
    gate, up = gate_up.chunk(2, dim=-1)
    gate = F.silu(gate, inplace=True)
    activated = up.mul_(gate)
    mlp_out = torch.matmul(activated, down_proj_weight.t())
    output = residual_2 + mlp_out
    return output.view(batch_size, seq_len, 1024)
