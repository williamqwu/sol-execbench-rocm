"""
Fused backward pass for Gemma3 QKV projection with Q/K RMSNorm.

Strategy
--------
The reference performs, per call:
  * elementwise Q/K norm backward   (bf16 rounding in the middle -- reproduced)
  * a reduction over (b, h, s) for the two norm-weight gradients
  * three transposes + contiguous copies
  * three "grad_input" GEMMs summed together
  * three "grad_weight" GEMMs

Everything before the GEMMs is one memory-bound pass, so it is fused into a
single Triton kernel that *directly emits the transposed / concatenated*
layout  gproj = [dQ_proj | dK_proj | dV_proj]  of shape (B*S, 1536).  That
layout is exactly what both GEMM groups want:

    grad_hidden_states = gproj        @ cat(Wq, Wk, Wv)      (N,1536)x(1536,640)
    grad_W_cat         = gproj.T      @ hidden               (1536,N)x(N,640)

so the six reference GEMMs collapse into two, and the three explicit
`.contiguous()` transposes disappear entirely.

The norm-weight gradients are accumulated per-program into a partials buffer
and reduced afterwards (cheaper and more deterministic than global atomics on
just 512 float addresses).
"""

import torch
import triton
import triton.language as tl

_D: int = 256          # head_dim
_NH: int = 4           # num_attention_heads
_HS: int = 640         # hidden_size
_QCOLS: int = 1024     # num_attention_heads * head_dim
_OCOLS: int = 1536     # q_proj_size + 2 * kv_proj_size


@triton.jit
def _qkv_norm_bwd(
    GQ, GK, GV,          # grad_query (B,4,S,D), grad_key (B,1,S,D), grad_value
    QN, KN,              # q_normed, k_normed
    QR, KR,              # q_rstd (B,4,S,1) f32, k_rstd (B,1,S,1) f32
    QW, KW,              # q_norm_weight (D,), k_norm_weight (D,)
    OUT,                 # (B*S, 1536) bf16
    PART,                # (P, 2*D) f32 partials
    S,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    NH: tl.constexpr,
    OCOLS: tl.constexpr,
    QCOLS: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    s = pid_s * ROWS + tl.arange(0, ROWS)
    smask = s < S
    d = tl.arange(0, D)

    q_scale = 1.0 + tl.load(QW + d).to(tl.float32)
    k_scale = 1.0 + tl.load(KW + d).to(tl.float32)

    row = pid_b * S + s
    obase = row[:, None] * OCOLS

    acc_q = tl.zeros((D,), dtype=tl.float32)

    # ---------------- Q: NH heads ----------------
    for h in tl.static_range(NH):
        off = pid_b * (NH * S * D) + h * (S * D) + s[:, None] * D + d[None, :]
        gq = tl.load(GQ + off, mask=smask[:, None], other=0.0).to(tl.float32)
        qn = tl.load(QN + off, mask=smask[:, None], other=0.0).to(tl.float32)

        acc_q += tl.sum(gq * qn, axis=0)

        gqn = (gq * q_scale[None, :]).to(tl.bfloat16).to(tl.float32)
        m = tl.sum(gqn * qn, axis=1) * (1.0 / D)
        rstd = tl.load(QR + pid_b * (NH * S) + h * S + s, mask=smask, other=0.0)
        o = (rstd[:, None] * (gqn - m[:, None] * qn)).to(tl.bfloat16)
        tl.store(OUT + obase + h * D + d[None, :], o, mask=smask[:, None])

    # ---------------- K ----------------
    off = pid_b * (S * D) + s[:, None] * D + d[None, :]
    gk = tl.load(GK + off, mask=smask[:, None], other=0.0).to(tl.float32)
    kn = tl.load(KN + off, mask=smask[:, None], other=0.0).to(tl.float32)

    acc_k = tl.sum(gk * kn, axis=0)

    gkn = (gk * k_scale[None, :]).to(tl.bfloat16).to(tl.float32)
    m = tl.sum(gkn * kn, axis=1) * (1.0 / D)
    rstd = tl.load(KR + pid_b * S + s, mask=smask, other=0.0)
    o = (rstd[:, None] * (gkn - m[:, None] * kn)).to(tl.bfloat16)
    tl.store(OUT + obase + QCOLS + d[None, :], o, mask=smask[:, None])

    # ---------------- V: straight copy ----------------
    gv = tl.load(GV + off, mask=smask[:, None], other=0.0)
    tl.store(OUT + obase + QCOLS + D + d[None, :], gv, mask=smask[:, None])

    # ---------------- norm-weight partials ----------------
    pid = pid_s + tl.program_id(1) * tl.num_programs(0)
    tl.store(PART + pid * (2 * D) + d, acc_q)
    tl.store(PART + pid * (2 * D) + D + d, acc_k)


def _pick_rows(n: int) -> int:
    # aim for >= ~1024 programs while keeping enough work per program
    for r in (8, 4, 2):
        if n >= 1024 * r:
            return r
    return 1


def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_transposed: torch.Tensor,
    key_transposed: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    q_normed: torch.Tensor,
    k_normed: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, seq_len, _ = hidden_states.shape
    n = batch_size * seq_len
    dev = grad_query.device

    rows = _pick_rows(n)
    grid_s = triton.cdiv(seq_len, rows)
    nprog = grid_s * batch_size

    gproj = torch.empty((n, _OCOLS), dtype=torch.bfloat16, device=dev)
    part = torch.empty((nprog, 2 * _D), dtype=torch.float32, device=dev)

    _qkv_norm_bwd[(grid_s, batch_size)](
        grad_query, grad_key, grad_value,
        q_normed, k_normed,
        q_rstd, k_rstd,
        q_norm_weight, k_norm_weight,
        gproj, part,
        seq_len,
        ROWS=rows, D=_D, NH=_NH, OCOLS=_OCOLS, QCOLS=_QCOLS,
        num_warps=4, num_stages=1,
    )

    gnw = part.sum(0)

    w_cat = torch.cat((q_weight, k_weight, v_weight), 0)
    grad_hidden_states = torch.mm(gproj, w_cat).view(batch_size, seq_len, _HS)

    gw = torch.mm(gproj.t(), hidden_states.reshape(n, _HS))

    return (
        grad_hidden_states,
        gw[:_QCOLS],
        gw[_QCOLS:_QCOLS + _D],
        gw[_QCOLS + _D:],
        gnw[:_D],
        gnw[_D:],
    )
