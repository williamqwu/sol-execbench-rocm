import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


# ---------------------------------------------------------------------------
# Numerics note (this is the whole difficulty of this problem).
#
# The tolerance here is ~1 ulp of float32 with required_matched_ratio 0.99.
# It is tight enough that an *exact* float64 evaluation of the whole pipeline
# fails it (measured: matched ratio 0.07-0.43). In other words the tolerance
# does not admit "a more accurate answer" -- it only admits reproducing the
# reference's own float32 rounding. Two consequences:
#
#   1. The four GEMMs must stay torch `mm` calls. Any Triton/split-k/bf16x3
#      reimplementation changes the accumulation order and fails, even when it
#      is measurably closer to the true result.
#   2. The fused GELU-backward elementwise pass must be *bit-exact* against
#      torch. Three separate things were needed to get there:
#        - enable_fp_fusion=False: otherwise the compiler contracts a*b+c into
#          an FMA (and perturbs libdevice.tanh), which torch does not do.
#        - the reference's exact expression tree/associativity, e.g.
#          `x*x*x` and `... * x * x`, not `x2*x` / `... * x2`.
#        - C3 below: torch folds `3.0 * 0.044715` in Python *double* before it
#          ever touches a tensor. Letting Triton fold it in float32 lands one
#          ulp off and loses ~10% of elements.
#
# With all three, every output is bit-identical to the reference (verified:
# max_abs error exactly 0.0 on all 16 workloads).
# ---------------------------------------------------------------------------

SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
GELU_CONST = tl.constexpr(0.044715)
# round_to_f32(double(3.0 * 0.044715)) -- see note above.
GELU_CONST_3 = tl.constexpr(0.13414500653743744)


@triton.jit
def _gelu_bwd_inplace(
    G,  # in: grad_gelu_output, out: grad_fc1_output (S*I, fp32)
    X,  # fc1_output (S*I, fp32)
    n_elements,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(G + offs, mask=mask)
    x = tl.load(X + offs, mask=mask)

    SQ = SQRT_2_OVER_PI
    C = GELU_CONST
    C3 = GELU_CONST_3

    # Mirrors reference.py term for term, including associativity.
    x_cubed = x * x * x
    tanh_out = libdevice.tanh(SQ * (x + C * x_cubed))
    sech_sq = 1.0 - tanh_out * tanh_out
    d_tanh_arg = SQ * (1.0 + C3 * x * x)
    gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech_sq * d_tanh_arg

    tl.store(G + offs, g * gelu_grad, mask=mask)


def _gelu_backward_(grad_gelu_output, fc1_output):
    """Fuse the 12-op GELU-derivative chain into one pass, in place.

    The reference materialises x_cubed, tanh_arg, tanh_out, sech_sq,
    d_tanh_arg, gelu_grad and grad_fc1_output as separate (seq_len,
    intermediate_size) tensors -- ~7 extra round trips through HBM. This does
    it in a single read-read-write. Writing back into grad_gelu_output is safe:
    it is a fresh temporary produced by the mm above and is never reused.
    """
    n = grad_gelu_output.numel()
    BLOCK = 2048
    _gelu_bwd_inplace[(triton.cdiv(n, BLOCK),)](
        grad_gelu_output,
        fc1_output,
        n,
        BLOCK=BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return grad_gelu_output


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_state: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    fc1_output: torch.Tensor,
    gelu_output: torch.Tensor,
):
    # --- fc2 backward -----------------------------------------------------
    grad_fc2_bias = grad_output.sum(dim=0)
    grad_fc2_weight = grad_output.t().mm(gelu_output)
    grad_gelu_output = grad_output.mm(fc2_weight)

    # --- GELU backward (fused, bit-exact, in place) -----------------------
    grad_fc1_output = _gelu_backward_(grad_gelu_output, fc1_output)

    # --- fc1 backward -----------------------------------------------------
    grad_fc1_bias = grad_fc1_output.sum(dim=0)
    grad_fc1_weight = grad_fc1_output.t().mm(hidden_state)
    grad_hidden_state = grad_fc1_output.mm(fc1_weight)

    return (
        grad_hidden_state,
        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias,
    )
