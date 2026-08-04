import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

SP = tl.constexpr(0.7978845608028654)
C = tl.constexpr(0.044715)
C3 = tl.constexpr(3.0 * 0.044715)


@triton.jit
def _gelu_bwd(X, GG, O, n, BLK: tl.constexpr):
    p = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = p < n
    x = tl.load(X + p, mask=m, other=0.0)
    gg = tl.load(GG + p, mask=m, other=0.0)
    t = libdevice.tanh(SP * (x + C * (x * x * x)))
    o = gg * (0.5 * (1.0 + t) + ((0.5 * x) * (1.0 - t * t)) * (SP * (1.0 + (C3 * x) * x)))
    tl.store(O + p, o, mask=m)


@torch.no_grad()
def run(
    grad_output, hidden_states, fc1_weight, fc1_output, gelu_output,
    fc2_weight, residual_output, normalized, var, ln_weight, eps,
):
    B, S, H = grad_output.shape
    I = fc1_weight.shape[0]
    N = B * S

    grad_ln_weight = (grad_output * normalized).sum(dim=(0, 1))
    grad_ln_bias = grad_output.sum(dim=(0, 1))

    gn = grad_output * ln_weight
    std = torch.sqrt(var + eps)
    m1 = gn.mean(-1, keepdim=True)
    m2 = (gn * normalized).mean(-1, keepdim=True)
    gro = (1.0 / std) * (gn - m1 - normalized * m2)

    grad_fc2_bias = gro.sum(dim=(0, 1))
    gro2 = gro.view(N, H)
    grad_fc2_weight = gro2.t() @ gelu_output.view(N, I)
    ggel = gro @ fc2_weight

    gfo = torch.empty_like(ggel)
    n = gfo.numel()
    _gelu_bwd[(triton.cdiv(n, 1024),)](
        fc1_output, ggel, gfo, n, 1024, enable_fp_fusion=False, num_warps=4
    )

    grad_fc1_bias = gfo.sum(dim=(0, 1))
    grad_fc1_weight = gfo.view(N, I).t() @ hidden_states.view(N, H)
    grad_hidden_states = gfo @ fc1_weight + gro

    return (
        grad_hidden_states, grad_fc1_weight, grad_fc1_bias,
        grad_fc2_weight, grad_fc2_bias, grad_ln_weight, grad_ln_bias,
    )
