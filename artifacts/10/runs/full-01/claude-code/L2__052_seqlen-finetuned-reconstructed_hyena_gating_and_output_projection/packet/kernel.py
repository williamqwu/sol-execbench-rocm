import torch
import triton
import triton.language as tl


@triton.jit
def _pad_mul(V, X1, K, OUT, L, N, BC, BLOCK: tl.constexpr):
    """Rows [0,BC) = (v*x1) zero-padded to N; rows [BC,BC+C) = k0 zero-padded.

    Packing the filter into the same buffer lets a single batched rFFT cover
    both, which is bit-identical to two separate rFFTs (verified) but one
    launch instead of two.
    """
    row = tl.program_id(0)
    blk = tl.program_id(1)
    o = blk * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    ml = o < L
    if row < BC:
        val = tl.load(V + row * L + o, mask=ml, other=0.0) * \
              tl.load(X1 + row * L + o, mask=ml, other=0.0)
    else:
        val = tl.load(K + (row - BC) * L + o, mask=ml, other=0.0)
    tl.store(OUT + row * N + o, tl.where(ml, val, 0.0), mask=m)


@triton.jit
def _epilogue(Y, VG, X0, BIAS, OUT, L, N, C, BL: tl.constexpr):
    """out = (y + vg*bias0) * x0, reading y/vg from the N-padded buffers."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    o = blk * BL + tl.arange(0, BL)
    m = o < L
    y = tl.load(Y + row * N + o, mask=m, other=0.0)
    vg = tl.load(VG + row * N + o, mask=m, other=0.0)
    x0 = tl.load(X0 + row * L + o, mask=m, other=0.0)
    b = tl.load(BIAS + (row % C))
    p = vg * b
    # The reference rounds vg*bias0 to fp32 before the add. Triton would
    # otherwise contract this into an FMA and be *more* accurate than the
    # reference, which fails the ~1-ulp tolerance. This blocks contraction.
    p = tl.inline_asm_elementwise("v_mov_b32 $0, $1", "=v,v", [p],
                                  dtype=tl.float32, is_pure=True, pack=1)
    tl.store(OUT + row * L + o, (y + p) * x0, mask=m)


def run(v, x0, x1, k, bias, out_proj_weight, out_proj_bias):
    B, C, L = v.shape
    N = 2 * L
    BC = B * C

    v = v.contiguous()
    x0 = x0.contiguous()
    x1 = x1.contiguous()
    k0 = k[0].contiguous()

    # Fused gate + zero-pad for both signal and filter, into one buffer.
    buf = torch.empty((BC + C, N), device=v.device, dtype=torch.float32)
    BLOCK = 1024
    _pad_mul[(BC + C, triton.cdiv(N, BLOCK))](
        v, x1, k0, buf, L, N, BC, BLOCK=BLOCK, num_warps=4)

    # One batched real FFT covers signal and filter.
    Fh = torch.fft.rfft(buf)
    u_f = Fh[:BC].view(B, C, -1)
    k_f = Fh[BC:] / N
    u_f = u_f * k_f.unsqueeze(0)
    y = torch.fft.irfft(u_f, n=N, norm='forward')

    # Gate with x0 while still in (B, C, L); the transposed view fed to matmul
    # matches the reference's GEMM layout exactly, so hipBLASLt picks the same
    # kernel and the result is bit-identical.
    g = torch.empty((B, C, L), device=v.device, dtype=torch.float32)
    _epilogue[(BC, triton.cdiv(L, 1024))](
        y, buf, x0, bias[0], g, L, N, C, BL=1024, num_warps=4)

    return torch.matmul(g.transpose(1, 2), out_proj_weight.t()) + out_proj_bias
