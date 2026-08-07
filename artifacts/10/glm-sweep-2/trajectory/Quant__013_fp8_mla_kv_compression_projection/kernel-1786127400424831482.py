import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 256, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 64, "BN": 256, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 256, "BN": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 256, "GROUP_M": 8}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fp8_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    sa_ptr, sb_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_sam, stride_sak,
    stride_sbk, stride_sbn,
    BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr,
):
    BK: tl.constexpr = 128
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    rm = offs_m % M
    rn = offs_n % N
    a_ptrs = a_ptr + rm[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + rn[None, :] * stride_bn

    n_blk = (offs_n // 128).to(tl.int64)

    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BK)
    for k in range(0, num_k_tiles):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc_tile = tl.dot(a, b, out_dtype=tl.float32)
        sa_k = tl.load(sa_ptr + offs_m * stride_sam + k * stride_sak, mask=offs_m < M, other=0.0)
        sb_k = tl.load(sb_ptr + k * stride_sbk + n_blk * stride_sbn, mask=offs_n < N, other=0.0)
        accumulator += acc_tile * sa_k[:, None] * sb_k[None, :]
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


def fp8_mm_blockwise(a, b, sa, sb):
    """a:(M,K) fp8 row-major; b:(K,N) fp8; sa:(M,K//128); sb:(K//128,N//128)."""
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    _fp8_mm_kernel[grid](
        a, b, c, sa, sb,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        sa.stride(0), sa.stride(1),
        sb.stride(0), sb.stride(1),
    )
    return c


def _compute_act_scales_1x128(x_fp32):
    """x_fp32: (M,K), K%128==0. Returns (M, K//128) inverse scales, quantized x fp8."""
    M, K = x_fp32.shape
    xb = x_fp32.view(M, K // 128, 128)
    amax = xb.abs().amax(dim=2)  # (M, K//128)
    scales = (amax / E4M3_MAX).clamp(min=1e-12)
    qx = (xb / scales.unsqueeze(2)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return qx.reshape(M, K), scales


def _compute_weight_scales_128x128(w_fp32_t):
    """w_fp32_t: (K,N), K%128==0, N%128==0. Returns (K//128, N//128) scales, quantized w_t fp8 (K,N)."""
    K, N = w_fp32_t.shape
    wb = w_fp32_t.view(K // 128, 128, N // 128, 128)
    amax = wb.abs().amax(dim=3).amax(dim=1)  # (K//128, N//128)
    scales = (amax / E4M3_MAX).clamp(min=1e-12)
    qw = (wb / scales.unsqueeze(1).unsqueeze(3)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return qw.reshape(K, N), scales


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128

    bsz, q_len, hidden_size = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    M = hidden_flat.shape[0]

    # ===== Step 1: FP8 Compression Projection =====
    # hidden:(M,7168) act 1x128 ; weight (640,7168) -> use W^T (7168,640) with 128x128 scales
    x_fp32 = hidden_flat.to(torch.float32)
    w_a_fp32 = kv_a_proj_weight.to(torch.float32)
    qx_a, scale_x_a = _compute_act_scales_1x128(x_fp32)          # (M,7168) fp8, (M,56)
    qw_a_t, scale_w_a = _compute_weight_scales_128x128(w_a_fp32.T)  # (7168,640) fp8, (56,5)

    compressed_kv_with_rope = fp8_mm_blockwise(qx_a, qw_a_t, scale_x_a, scale_w_a)  # (M,640)

    # ===== Step 2: Split and RMSNorm =====
    compressed_kv = compressed_kv_with_rope[:, :kv_lora_rank]
    k_pe = compressed_kv_with_rope[:, kv_lora_rank:kv_lora_rank + qk_rope_head_dim]

    compressed_kv_fp32 = compressed_kv.to(torch.float32)
    variance = compressed_kv_fp32.pow(2).mean(-1, keepdim=True)
    compressed_kv_norm = compressed_kv_fp32 * torch.rsqrt(variance + rms_norm_eps)
    compressed_kv_norm = kv_a_layernorm_weight * compressed_kv_norm.to(kv_a_layernorm_weight.dtype)

    # ===== Step 3: FP8 Expansion Projection =====
    x_b_fp32 = compressed_kv_norm.to(torch.float32)
    w_b_fp32 = kv_b_proj_weight.to(torch.float32)
    qx_b, scale_x_b = _compute_act_scales_1x128(x_b_fp32)          # (M,512) fp8, (M,4)
    qw_b_t, scale_w_b = _compute_weight_scales_128x128(w_b_fp32.T)  # (512,24576) fp8, (4,192)

    kv_expanded_flat = fp8_mm_blockwise(qx_b, qw_b_t, scale_x_b, scale_w_b)  # (M,24576)

    kv_expanded = kv_expanded_flat.view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
    k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim)
    return kv_expanded, k_pe


if __name__ == "__main__":
    inputs = get_inputs(
        axes_and_scalars={"batch_size": 2, "seq_len": 128},
        device=torch.device("cuda:0"),
    )
    kv_expanded, k_pe = run(**inputs)
    print(f"kv_expanded shape: {kv_expanded.shape}")
    print(f"k_pe shape: {k_pe.shape}")
