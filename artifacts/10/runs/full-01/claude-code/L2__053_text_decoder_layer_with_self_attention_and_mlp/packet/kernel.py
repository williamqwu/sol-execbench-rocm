import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Numerics note
# ---------------------------------------------------------------------------
# With random weights the attention logits reach |s| ~ 2e4, where one bf16 ulp
# is ~85.  The softmax is therefore effectively an argmax (mean max-prob .986),
# so a single ulp of difference in the pre-softmax path flips which value row is
# selected, and the o_proj + RMSNorm that follow smear that one bad element
# across an entire token row.  Everything up to and including the P@V matmul is
# consequently kept bit-identical to the reference (same torch ops, same
# layouts, same accumulation order).  Everything after the attention is smooth,
# so that half is fused into Triton.

_NH = 32
_NKV = 8
_HD = 128
_G = 4


# ---------------------------------------------------------------------------
# RMSNorm  (bit-exact vs the reference: fp32 accumulate, round to bf16, then
# multiply by the weight in bf16)
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_kernel(X, W, Y, N: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    t = (x * rstd).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + cols, (t * w).to(tl.bfloat16), mask=mask)


def rms_norm(x2d, weight, eps):
    n_rows, N = x2d.shape
    out = torch.empty_like(x2d)
    _rmsnorm_kernel[(n_rows,)](
        x2d, weight, out, N=N, eps=eps,
        BLOCK=triton.next_power_of_2(N), num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# residual + o_proj-style add:  out = a + b, elementwise bf16
# fused silu(gate) * up
# ---------------------------------------------------------------------------
@triton.jit
def _silu_mul_kernel(G, U, O, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    s = g / (1.0 + tl.exp(-g))
    s = s.to(tl.bfloat16).to(tl.float32)
    tl.store(O + offs, (s * u).to(tl.bfloat16), mask=mask)


def silu_mul_(gate, up):
    n = gate.numel()
    _silu_mul_kernel[(triton.cdiv(n, 8192),)](
        gate, up, gate, n, BLOCK=8192, num_warps=8,
    )
    return gate


# ---------------------------------------------------------------------------
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    rms_norm_eps: float,
    rope_theta: float,
) -> torch.Tensor:
    NH, NKV, HD, G = _NH, _NKV, _HD, _G
    scaling = HD ** -0.5

    bsz, seq_len, hidden = hidden_states.shape
    ntok = bsz * seq_len
    device = hidden_states.device

    x2d = hidden_states.reshape(ntok, hidden)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()

    # ---- input RMSNorm (fused) ----
    xn = rms_norm(x2d, input_layernorm_weight, rms_norm_eps)

    # ---- Q/K/V projections (same op the reference uses) ----
    query_states = F.linear(xn, q_proj_weight)
    key_states = F.linear(xn, k_proj_weight)
    value_states = F.linear(xn, v_proj_weight)

    query_states = query_states.view(bsz, seq_len, NH, HD).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, NKV, HD).transpose(1, 2)
    value_states = value_states.view(bsz, seq_len, NKV, HD).transpose(1, 2)

    # ---- RoPE (verbatim: feeds the argmax-sensitive score matmul) ----
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, HD, 2, dtype=torch.float32, device=device) / HD))
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(bsz, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)

    half = HD // 2
    q1, q2 = query_states[..., :half], query_states[..., half:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = ((query_states * cos) + (q_rotated * sin)).to(query_states.dtype)

    k1, k2 = key_states[..., :half], key_states[..., half:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = ((key_states * cos) + (k_rotated * sin)).to(key_states.dtype)

    # ---- GQA repeat ----
    key_states = key_states[:, :, None, :, :].expand(
        bsz, NKV, G, seq_len, HD).reshape(bsz, NKV * G, seq_len, HD)
    value_states = value_states[:, :, None, :, :].expand(
        bsz, NKV, G, seq_len, HD).reshape(bsz, NKV * G, seq_len, HD)

    # ---- attention (verbatim) ----
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask[:, :, :seq_len, :seq_len]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(ntok, hidden)

    # ---- o_proj + residual.  NB: mm-then-add, not addmm: the reference rounds
    # the projection to bf16 before the residual add. ----
    h1 = x2d + F.linear(attn_output, o_proj_weight)

    # ---- MLP ----
    hn = rms_norm(h1, post_attention_layernorm_weight, rms_norm_eps)
    gate = F.linear(hn, gate_proj_weight)
    up = F.linear(hn, up_proj_weight)
    gate = silu_mul_(gate, up)
    out = h1 + F.linear(gate, down_proj_weight)

    return out.view(bsz, seq_len, hidden)
