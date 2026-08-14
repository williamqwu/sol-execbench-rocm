import torch
import triton
import triton.language as tl

# Optional AMD aiter asm GEMM. Falls back to torch.mm cleanly if unavailable.
try:
    import aiter as _aiter
    _HAS_AITER = hasattr(_aiter, "gemm_a16w16_asm")
except Exception:
    _aiter = None
    _HAS_AITER = False

_AITER_OK = _HAS_AITER


# ---------------------------------------------------------------------------
# Fused: residual add (bf16-rounded) + RMSNorm (fp32 accum) + weight scale
#
# Reproduces reference op-for-op:
#     t = (residual + hidden_states)                  -> bf16 rounding
#     v = t.float().pow(2).mean(-1)
#     y = (norm_weight * (t.float() * rsqrt(v+eps))).to(bf16)
# ---------------------------------------------------------------------------
@triton.jit
def _add_rmsnorm_kernel(
    H_ptr, R_ptr, W_ptr, Y_ptr,
    eps,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols

    # ---- pass 1: add + sum of squares (accumulated in fp32) ----
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(NUM_SPLIT):
        cols = i * BLOCK + tl.arange(0, BLOCK)
        h = tl.load(H_ptr + base + cols).to(tl.float32)
        r = tl.load(R_ptr + base + cols).to(tl.float32)
        t = (r + h).to(tl.bfloat16).to(tl.float32)
        acc += t * t
    var = tl.sum(acc, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # ---- pass 2: normalize + scale (re-reads from L2) ----
    for i in tl.static_range(NUM_SPLIT):
        cols = i * BLOCK + tl.arange(0, BLOCK)
        h = tl.load(H_ptr + base + cols).to(tl.float32)
        r = tl.load(R_ptr + base + cols).to(tl.float32)
        t = (r + h).to(tl.bfloat16).to(tl.float32)
        w = tl.load(W_ptr + cols).to(tl.float32)
        tl.store(Y_ptr + base + cols, (w * (t * rstd)).to(tl.bfloat16))


@triton.jit
def _add_rmsnorm_kernel_masked(
    H_ptr, R_ptr, W_ptr, Y_ptr,
    eps,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    off = row * n_cols + cols
    h = tl.load(H_ptr + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + off, mask=mask, other=0.0).to(tl.float32)
    t = (r + h).to(tl.bfloat16).to(tl.float32)
    var = tl.sum(t * t, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y_ptr + off, (w * (t * rstd)).to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Fused SwiGLU:  out = bf16( bf16(silu(g)) * u )
# reference: F.silu(gate) -> bf16 tensor, then * up -> bf16
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_kernel(
    G_ptr, U_ptr, O_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * (BLOCK * UNROLL)
    for i in tl.static_range(UNROLL):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        tl.store(O_ptr + offs, (s * u).to(tl.bfloat16), mask=mask)


def _add_rmsnorm(x, r, norm_weight, eps):
    M, N = x.shape
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if N % 4096 == 0 and N >= 4096:
        BLOCK = 4096
        NUM_SPLIT = N // BLOCK
        _add_rmsnorm_kernel[(M,)](
            x, r, norm_weight, y, eps, N,
            BLOCK=BLOCK, NUM_SPLIT=NUM_SPLIT,
            num_warps=8, num_stages=1,
        )
    else:
        BLOCK = triton.next_power_of_2(N)
        _add_rmsnorm_kernel_masked[(M,)](
            x, r, norm_weight, y, eps, N,
            BLOCK=BLOCK, num_warps=8, num_stages=1,
        )
    return y


def _swiglu(gate, up):
    out = torch.empty_like(gate)
    n = gate.numel()
    BLOCK, UNROLL = 8192, 2
    grid = (triton.cdiv(n, BLOCK * UNROLL),)
    _swiglu_kernel[grid](gate, up, out, n, BLOCK=BLOCK, UNROLL=UNROLL,
                         num_warps=8, num_stages=1)
    return out


def _gemm_nt(a, w, out, split_k=None):
    """out = a @ w.T   with a:[M,K] w:[N,K] both row-major contiguous."""
    if _AITER_OK:
        try:
            if split_k is None:
                _aiter.gemm_a16w16_asm(a, w, out)
            else:
                _aiter.gemm_a16w16_asm(a, w, out, None, split_k)
            return out
        except Exception:
            pass
    return torch.mm(a, w.t(), out=out)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float,
):
    out_shape = hidden_states.shape
    Hn = out_shape[-1]

    x2 = hidden_states.reshape(-1, Hn)
    r2 = residual.reshape(-1, Hn)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not r2.is_contiguous():
        r2 = r2.contiguous()
    M = x2.shape[0]

    x = _add_rmsnorm(x2, r2, norm_weight, eps)

    I = gate_proj_weight.shape[0]
    gate = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
    up = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)

    # gate/up: N=intermediate is large -> aiter asm wins at every measured M.
    # small M benefits from split-K (more waves over a short N-tile).
    sk = 4 if M <= 512 else None
    gw = gate_proj_weight if gate_proj_weight.is_contiguous() else gate_proj_weight.contiguous()
    uw = up_proj_weight if up_proj_weight.is_contiguous() else up_proj_weight.contiguous()
    _gemm_nt(x, gw, gate, sk)
    _gemm_nt(x, uw, up, sk)

    inter = _swiglu(gate, up)
    del gate, up

    # down: K=intermediate is huge, N=hidden small. hipBLASLt is at/above the
    # asm kernel here at every measured M and is 2.2x better at small M.
    out = torch.mm(inter, down_proj_weight.t())
    return out.view(out_shape)
