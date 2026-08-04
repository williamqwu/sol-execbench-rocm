import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(
    x,
    cos,
    sin,
    out,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    nheads: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_s: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    pair = offs & 63
    tmp = offs >> 6
    pos = tmp % seq_len
    head = (tmp // seq_len) % nheads
    batch = tmp // (seq_len * nheads)
    ptr = batch * stride_b + head * stride_h + pos * stride_s + pair * 2
    cs_ptr = pos * 64 + pair

    x0 = tl.load(x + ptr, mask=mask).to(tl.float32)
    x1 = tl.load(x + ptr + 1, mask=mask).to(tl.float32)
    c = tl.load(cos + cs_ptr, mask=mask)
    s = tl.load(sin + cs_ptr, mask=mask)

    y0 = x0 * c + (-x1) * s
    y1 = x1 * c + x0 * s
    tl.store(out + ptr, y0, mask=mask)
    tl.store(out + ptr + 1, y1, mask=mask)


@triton.jit
def _rope_bwd_kernel(
    g,
    cos,
    sin,
    out,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    nheads: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_s: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    pair = offs & 63
    tmp = offs >> 6
    pos = tmp % seq_len
    head = (tmp // seq_len) % nheads
    batch = tmp // (seq_len * nheads)
    ptr = batch * stride_b + head * stride_h + pos * stride_s + pair * 2
    cs_ptr = pos * 64 + pair

    g0 = tl.load(g + ptr, mask=mask).to(tl.float32)
    g1 = tl.load(g + ptr + 1, mask=mask).to(tl.float32)
    c = tl.load(cos + cs_ptr, mask=mask)
    s = tl.load(sin + cs_ptr, mask=mask)

    y0 = g0 * c + g1 * s
    y1 = g1 * c - g0 * s
    tl.store(out + ptr, y0, mask=mask)
    tl.store(out + ptr + 1, y1, mask=mask)


@torch.no_grad()
def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    total = x.shape[0] * x.shape[1] * x.shape[2] * 64
    _rope_fwd_kernel[(triton.cdiv(total, 256),)](
        x,
        cos,
        sin,
        out,
        total,
        x.shape[2],
        x.shape[1],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK=256,
    )
    return out


@torch.no_grad()
def _rope_backward(g: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out = torch.empty_strided(g.shape, g.stride(), device=g.device, dtype=g.dtype)
    total = g.shape[0] * g.shape[1] * g.shape[2] * 64
    _rope_bwd_kernel[(triton.cdiv(total, 256),)](
        g,
        cos,
        sin,
        out,
        total,
        g.shape[2],
        g.shape[1],
        g.stride(0),
        g.stride(1),
        g.stride(2),
        BLOCK=256,
    )
    return out


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    scaling: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    dtype = hidden_states.dtype
    nh = 64
    nkv = 8
    groups = 8
    hd = 128

    query_states = F.linear(hidden_states, q_weight)
    key_states = F.linear(hidden_states, k_weight)
    value_states = F.linear(hidden_states, v_weight)

    q = query_states.view(batch_size, seq_len, nh, hd).transpose(1, 2)
    k = key_states.view(batch_size, seq_len, nkv, hd).transpose(1, 2)
    v = value_states.view(batch_size, seq_len, nkv, hd).transpose(1, 2)

    pos = torch.arange(seq_len, device=hidden_states.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq.float())
    cos_i = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin_i = freqs.sin().unsqueeze(0).unsqueeze(0)

    q_rope = _apply_rope(q, cos_i, sin_i)
    k_rope = _apply_rope(k, cos_i, sin_i)

    qg = q_rope.view(batch_size, nkv, groups, seq_len, hd)
    gog = F.linear(grad_output, o_weight.t())
    goh = gog.view(batch_size, seq_len, nh, hd).transpose(1, 2)
    gog_g = goh.view(batch_size, nkv, groups, seq_len, hd)

    attn = torch.matmul(qg, k_rope[:, :, None].transpose(-2, -1)) * scaling
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
        diagonal=1,
    )
    attn = attn.masked_fill(mask, float("-inf"))
    attn_float = F.softmax(attn.float(), dim=-1)
    attn_bf16 = attn_float.to(dtype)

    attn_out_g = torch.matmul(attn_bf16, v[:, :, None])
    attn_output = attn_out_g.view(batch_size, nh, seq_len, hd).transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, hidden_size)
    grad_o_weight = grad_output.reshape(-1, hidden_size).t() @ attn_output.reshape(-1, hidden_size)

    grad_attn_weights = torch.matmul(gog_g, v[:, :, None].transpose(-2, -1))
    grad_value_states = torch.matmul(attn_bf16.transpose(-2, -1), gog_g).sum(dim=2)

    grad_attn_weights_f = grad_attn_weights.float()
    grad_attn_scores = attn_float * (
        grad_attn_weights_f - (grad_attn_weights_f * attn_float).sum(dim=-1, keepdim=True)
    )
    grad_attn_scores = grad_attn_scores * scaling
    grad_attn_scores_bf16 = grad_attn_scores.to(dtype)

    grad_query_rope = torch.matmul(grad_attn_scores_bf16, k_rope[:, :, None])
    grad_key_rope = torch.matmul(qg.transpose(-2, -1), grad_attn_scores_bf16)
    grad_key_rope = grad_key_rope.transpose(-2, -1).sum(dim=2)

    grad_query_states = _rope_backward(grad_query_rope.view(batch_size, nh, seq_len, hd), cos_i, sin_i)
    grad_key_states = _rope_backward(grad_key_rope, cos_i, sin_i)

    grad_query_states = grad_query_states.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    grad_key_states = grad_key_states.transpose(1, 2).contiguous().view(batch_size, seq_len, nkv * hd)
    grad_value_states = grad_value_states.transpose(1, 2).contiguous().view(batch_size, seq_len, nkv * hd)

    grad_hidden_q = F.linear(grad_query_states, q_weight.t())
    grad_hidden_k = F.linear(grad_key_states, k_weight.t())
    grad_hidden_v = F.linear(grad_value_states, v_weight.t())
    grad_hidden_states = grad_hidden_q + grad_hidden_k + grad_hidden_v

    hs2 = hidden_states.reshape(-1, hidden_size)
    grad_q_weight = grad_query_states.reshape(-1, hidden_size).t() @ hs2
    grad_k_weight = grad_key_states.reshape(-1, nkv * hd).t() @ hs2
    grad_v_weight = grad_value_states.reshape(-1, nkv * hd).t() @ hs2

    return (
        grad_hidden_states.to(dtype),
        grad_q_weight.to(dtype),
        grad_k_weight.to(dtype),
        grad_v_weight.to(dtype),
        grad_o_weight.to(dtype),
    )
