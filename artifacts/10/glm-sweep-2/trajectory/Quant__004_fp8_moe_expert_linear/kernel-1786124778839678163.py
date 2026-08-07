import torch
import torch.nn.functional as F
import triton
import triton.language as tl

E4M3_MAX = 448.0


def _scale_1x128(t):
    M, K = t.shape
    return (t.reshape(M, K // 128, 128).abs().amax(-1) / E4M3_MAX).clamp(min=1e-12)


def _quant_1x128(t, scale):
    M, K = t.shape
    return (t.reshape(M, K // 128, 128) / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).reshape(M, K).to(torch.float8_e4m3fn)


def _scale_128x128(t):
    M, K = t.shape
    return (t.reshape(M // 128, 128, K // 128, 128).abs().amax(3).amax(1) / E4M3_MAX).clamp(min=1e-12)


def _quant_128x128(t, scale):
    M, K = t.shape
    return (t.reshape(M // 128, 128, K // 128, 128) / scale.unsqueeze(1).unsqueeze(3)).clamp(-E4M3_MAX, E4M3_MAX).reshape(M, K).to(torch.float8_e4m3fn)


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128}, num_stages=3, num_warps=4),
        triton.Config({"BM": 128, "BN": 256}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 128}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 128}, num_stages=4, num_warps=8),
        triton.Config({"BM": 64, "BN": 128}, num_stages=3, num_warps=4),
        triton.Config({"BM": 128, "BN": 64}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_sam, stride_sak, stride_sbk, stride_sbn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    pid_m = pid // num_n
    pid_n = pid % num_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sa_base = sa_ptr + offs_m * stride_sam
    sb_base = sb_ptr + (offs_n // 128) * stride_sbk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        sa_v = tl.load(sa_base + (kb // BK) * stride_sak, mask=offs_m < M, other=0.0)
        sb_v = tl.load(sb_base + (kb // BK) * stride_sbn, mask=offs_n < N, other=0.0)
        partial = tl.dot(a, b)
        scale = sa_v[:, None] * sb_v[None, :]
        acc += partial * scale
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _blockwise_fp8_gemm(a_fp8, b_fp8, sa, sb):
    """Blockwise-scaled FP8 GEMM: C[M,N] = sum_k A_fp8[M,K] @ B_fp8[N,K]^T with
    1x128 activation scales (sa[M, K//128]) and 128x128 weight scales (sb[N//128, K//128])."""
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    c = torch.empty(M, N, device=a_fp8.device, dtype=torch.bfloat16)
    BK = 128
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    _gemm_kernel[grid](
        a_fp8, b_fp8, c, sa, sb, M, N, K,
        a_fp8.stride(0), a_fp8.stride(1), b_fp8.stride(1), b_fp8.stride(0),
        c.stride(0), c.stride(1),
        sa.stride(0), sa.stride(1), sb.stride(0), sb.stride(1),
        BK=BK,
    )
    return c


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    """FP8-quantized MoE expert computation with blockwise scaling."""
    hidden_fp32 = hidden_states.to(torch.float32)
    scale_hidden = _scale_1x128(hidden_fp32)
    hidden_fp8 = _quant_1x128(hidden_fp32, scale_hidden)

    gate_up_fp32 = gate_up_weight.to(torch.float32)
    scale_gate_up = _scale_128x128(gate_up_fp32)
    gate_up_fp8 = _quant_128x128(gate_up_fp32, scale_gate_up)

    gate_up_output = _blockwise_fp8_gemm(hidden_fp8, gate_up_fp8, scale_hidden, scale_gate_up)

    gate, up = gate_up_output.chunk(2, dim=-1)
    gated_output = F.silu(gate) * up

    gated_fp32 = gated_output.to(torch.float32)
    scale_gated = _scale_1x128(gated_fp32)
    gated_fp8 = _quant_1x128(gated_fp32, scale_gated)

    down_fp32 = down_weight.to(torch.float32)
    scale_down = _scale_128x128(down_fp32)
    down_fp8 = _quant_128x128(down_fp32, scale_down)

    output = _blockwise_fp8_gemm(gated_fp8, down_fp8, scale_gated, scale_down)

    return output * routing_weight
