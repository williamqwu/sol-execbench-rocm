import torch
import triton
import triton.language as tl

S2P = tl.constexpr(0.7978845608028654)
CF = tl.constexpr(0.044715)
C3 = tl.constexpr(3.0 * 0.044715)


# ---------------------------------------------------------------------------
# GELU (tanh approximation) backward, fused elementwise.
#
# Transcribes the reference expression tree exactly and compiles with
# enable_fp_fusion=False so the compiler does not contract a*b+c into an FMA.
# Both are required: the workload tolerances are ~1 ulp of the summed
# outputs, so this must be bit-identical to torch, not merely as accurate.
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_bwd_kernel(DG, X, O, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    dg = tl.load(DG + o, mask=m)
    x = tl.load(X + o, mask=m)

    x_cubed = x * x * x
    tanh_arg = S2P * (x + CF * x_cubed)
    t = tl.extra.libdevice.tanh(tanh_arg)
    dtanh_arg_dx = S2P * (1.0 + C3 * x * x)
    sech_sq = 1.0 - t * t
    gelu_grad = 0.5 * (1.0 + t) + 0.5 * x * sech_sq * dtanh_arg_dx

    tl.store(O + o, dg * gelu_grad, mask=m)


def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_output: torch.Tensor,
    gelu_output: torch.Tensor,
    fc2_weight: torch.Tensor,
    residual_output: torch.Tensor,
    normalized: torch.Tensor,
    var: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
):
    B, S, H = grad_output.shape
    I = fc1_weight.shape[0]
    N = B * S

    go2 = grad_output.reshape(N, H)
    nm2 = normalized.reshape(N, H)
    hs2 = hidden_states.reshape(N, H)

    # ---- layer norm backward -------------------------------------------
    grad_ln_weight = (go2 * nm2).sum(0)
    grad_ln_bias = go2.sum(0)

    gn = go2 * ln_weight
    std = torch.sqrt(var + eps)
    m1 = gn.mean(-1, keepdim=True).reshape(N, 1)
    m2 = (gn * nm2).mean(-1, keepdim=True).reshape(N, 1)
    g2 = ((1.0 / std).reshape(N, 1)) * (gn - m1 - nm2 * m2)

    grad_fc2_bias = g2.sum(0)

    # ---- FC2 backward ---------------------------------------------------
    grad_fc2_weight = g2.t() @ gelu_output.reshape(N, I)
    dgelu = g2 @ fc2_weight

    # ---- GELU backward --------------------------------------------------
    gf1 = torch.empty_like(dgelu)
    nn = N * I
    BLOCK = 1024
    _gelu_bwd_kernel[(triton.cdiv(nn, BLOCK),)](
        dgelu, fc1_output.reshape(N, I), gf1, nn,
        BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False,
    )

    # ---- FC1 backward ---------------------------------------------------
    grad_fc1_bias = gf1.sum(0)
    grad_fc1_weight = gf1.t() @ hs2
    grad_hidden_states = torch.addmm(g2, gf1, fc1_weight).reshape(B, S, H)

    return (
        grad_hidden_states,
        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
