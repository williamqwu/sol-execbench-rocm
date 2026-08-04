import torch
import triton
import triton.language as tl


# Semantics (F.conv1d, kernel=3, padding=2, groups=C, then truncate to seq_len)
# collapses to a causal 3-tap depthwise conv:
#
#   uc[b, c, t] = w[c,0]*u[b,c,t-2] + w[c,1]*u[b,c,t-1] + w[c,2]*u[b,c,t] + bias[c]
#
#   x0 = uc[:,   0:256, :]
#   x1 = uc[:, 256:512, :]
#   v  = uc[:, 512:768, :]
#   v_gated = v * x0
#
# One fused pass over the data: 3 input rows in, 3 output rows out. That is the
# minimum traffic the problem allows (2 * B*768*S*4 bytes).
#
# Accumulation is done in float64. The reference's fp32 conv is not bit-exactly
# reproducible by any fp32 operation order (an exhaustive search over orderings
# and fma placements tops out at 80% bit-exact), so the correctly-rounded value
# is the estimator closest to it: it matches x0/x1 at ~100% and v_gated at
# ~99.7%, against a required 99%. Gating is likewise done before rounding,
# which measures better than rounding to fp32 first (99.7% vs 99.2%).


@triton.jit
def _hyena_fused(
    u_ptr, w_ptr, b_ptr, o_ptr,
    S, DS, TDS, BS3, BSD,
    BLOCK_S: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    u_row = u_ptr + pid_b * BS3 + pid_c * S
    o_off = pid_b * DS + pid_c * S

    if EVEN:
        m0 = offs >= 0
    else:
        m0 = offs < S
    m1 = (offs >= 1) & m0
    m2 = (offs >= 2) & m0

    # ---- group 0 -> x0 ----
    wo0 = pid_c * 3
    w00 = tl.load(w_ptr + wo0 + 0).to(tl.float64)
    w01 = tl.load(w_ptr + wo0 + 1).to(tl.float64)
    w02 = tl.load(w_ptr + wo0 + 2).to(tl.float64)
    bb0 = tl.load(b_ptr + pid_c).to(tl.float64)
    a00 = tl.load(u_row + offs, mask=m0, other=0.0).to(tl.float64)
    a01 = tl.load(u_row + offs - 1, mask=m1, other=0.0).to(tl.float64)
    a02 = tl.load(u_row + offs - 2, mask=m2, other=0.0).to(tl.float64)
    acc0 = w00 * a02 + w01 * a01 + w02 * a00 + bb0

    # ---- group 1 -> x1 ----
    wo1 = wo0 + 768
    p1 = u_row + DS
    w10 = tl.load(w_ptr + wo1 + 0).to(tl.float64)
    w11 = tl.load(w_ptr + wo1 + 1).to(tl.float64)
    w12 = tl.load(w_ptr + wo1 + 2).to(tl.float64)
    bb1 = tl.load(b_ptr + pid_c + 256).to(tl.float64)
    a10 = tl.load(p1 + offs, mask=m0, other=0.0).to(tl.float64)
    a11 = tl.load(p1 + offs - 1, mask=m1, other=0.0).to(tl.float64)
    a12 = tl.load(p1 + offs - 2, mask=m2, other=0.0).to(tl.float64)
    acc1 = w10 * a12 + w11 * a11 + w12 * a10 + bb1

    # ---- group 2 -> v ----
    wo2 = wo0 + 1536
    p2 = u_row + 2 * DS
    w20 = tl.load(w_ptr + wo2 + 0).to(tl.float64)
    w21 = tl.load(w_ptr + wo2 + 1).to(tl.float64)
    w22 = tl.load(w_ptr + wo2 + 2).to(tl.float64)
    bb2 = tl.load(b_ptr + pid_c + 512).to(tl.float64)
    a20 = tl.load(p2 + offs, mask=m0, other=0.0).to(tl.float64)
    a21 = tl.load(p2 + offs - 1, mask=m1, other=0.0).to(tl.float64)
    a22 = tl.load(p2 + offs - 2, mask=m2, other=0.0).to(tl.float64)
    acc2 = w20 * a22 + w21 * a21 + w22 * a20 + bb2

    # out[0] = v_gated, out[1] = x0, out[2] = x1
    tl.store(o_ptr + o_off + offs, (acc2 * acc0).to(tl.float32), mask=m0)
    tl.store(o_ptr + TDS + o_off + offs, acc0.to(tl.float32), mask=m0)
    tl.store(o_ptr + 2 * TDS + o_off + offs, acc1.to(tl.float32), mask=m0)


_CFG = {}


def _cfg(S):
    c = _CFG.get(S)
    if c is None:
        if S >= 2048:
            blk, nw = 1024, 4
        elif S >= 512:
            blk, nw = 512, 4
        elif S >= 256:
            blk, nw = 256, 2
        else:
            blk, nw = 128, 1
        c = (blk, nw, -(-S // blk), (S % blk) == 0)
        _CFG[S] = c
    return c


def run(u: torch.Tensor, short_filter_weight: torch.Tensor, short_filter_bias: torch.Tensor):
    B, _, S = u.shape
    D = 256
    DS = D * S

    out = torch.empty((3, B, D, S), device=u.device, dtype=torch.float32)

    blk, nw, nsb, even = _cfg(S)

    _hyena_fused[(nsb, D, B)](
        u, short_filter_weight, short_filter_bias, out,
        S, DS, B * DS, 3 * DS, 0,
        BLOCK_S=blk,
        EVEN=even,
        num_warps=nw,
        num_stages=1,
    )
    return out[0], out[1], out[2]
