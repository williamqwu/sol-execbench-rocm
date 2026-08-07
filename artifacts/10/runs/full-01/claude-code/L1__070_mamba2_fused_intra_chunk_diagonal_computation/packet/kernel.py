import torch
import triton
import triton.language as tl


@triton.jit
def _mamba2_diag_kernel(
    X, A, Bp, Cp, Y,
    sxb, sxc, sxp, sxh,
    sab, sah, sac,
    sbb, sbc, sbp, sbg,
    NC,
    REP: tl.constexpr,
    CS: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
):
    h = tl.program_id(0)
    bc = tl.program_id(1)
    bidx = bc // NC
    cidx = bc % NC
    g = h // REP

    om = tl.arange(0, CS)
    ok = tl.arange(0, S)
    od = tl.arange(0, D)

    # A_cumsum row for this (batch, head, chunk); P = segment cumulative sum.
    a = tl.load(A + bidx * sab + h * sah + cidx * sac + om).to(tl.float32)
    P = tl.cumsum(a, axis=0)

    # G = C @ B^T  (shared across the REP heads of this group)
    gb = bidx * sbb + cidx * sbc + g * sbg
    ct = tl.load(Cp + gb + om[:, None] * sbp + ok[None, :])
    bt = tl.load(Bp + gb + ok[:, None] + om[None, :] * sbp)
    G = tl.dot(ct, bt)

    # L = exp(segment_sum(A)) with strict causal mask; M = G * L
    E = tl.exp2((P[:, None] - P[None, :]) * 1.4426950408889634)
    M = G * tl.where(om[:, None] >= om[None, :], E, 0.0)

    xt = tl.load(X + bidx * sxb + cidx * sxc + h * sxh + om[:, None] * sxp + od[None, :])

    # Split M into high/low bfloat16 limbs so the bf16 MFMA pair reproduces the
    # reference's float32 accumulation of M @ hidden_states.
    Mh = M.to(tl.bfloat16)
    Ml = (M - Mh.to(tl.float32)).to(tl.bfloat16)
    acc = tl.dot(Mh, xt)
    acc += tl.dot(Ml, xt)

    tl.store(
        Y + bidx * sxb + cidx * sxc + h * sxh + om[:, None] * sxp + od[None, :],
        acc.to(tl.bfloat16),
    )


def run(hidden_states: torch.Tensor,
        A_cumsum: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor) -> torch.Tensor:
    b, nc, cs, h, d = hidden_states.shape
    s = B.shape[-1]
    rep = h // B.shape[3]

    hidden_states = hidden_states.contiguous()
    A_cumsum = A_cumsum.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    Y = torch.empty_like(hidden_states)

    _mamba2_diag_kernel[(h, b * nc)](
        hidden_states, A_cumsum, B, C, Y,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states.stride(2), hidden_states.stride(3),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        nc,
        REP=rep, CS=cs, D=d, S=s,
        num_warps=4, num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return Y
