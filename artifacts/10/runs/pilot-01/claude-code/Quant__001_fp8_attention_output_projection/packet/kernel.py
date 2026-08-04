import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0
FP8MAX = tl.constexpr(448.0)
RECIP_FP8MAX = tl.constexpr(1.0 / 448.0)


@triton.jit
def _fused_fp8_blockwise_gemm(
    A, W, BIAS, C,
    M, N, K,
    stride_am, stride_wn, stride_cm,
    BM: tl.constexpr, BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_NBLK: tl.constexpr,   # BN // 128
):
    """out[m,n] = sum_k q(A)[m,k]*sa[m,k//128] * q(W)[n,k]*sw[n//128,k//128] + bias[n]

    Activations: BlockWise1x128 (per row, per 128-wide K block)
    Weights:     BlockWise128x128 over W^T (K,N) == 128x128 blocks of W (N,K)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)

    mask_m = offs_m < M
    am = tl.max_contiguous(tl.multiple_of(tl.where(mask_m, offs_m, 0), BM), BM)
    an = tl.max_contiguous(tl.multiple_of(offs_n % N, BN), BN)

    a_ptrs = A + am[:, None] * stride_am + offs_k[None, :]
    w_ptrs = W + an[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, 128)):
        a = tl.load(a_ptrs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)

        # --- activation scales: 1x128 blocks ---
        sa = tl.maximum(tl.max(tl.abs(a), axis=1) * RECIP_FP8MAX, 1e-12)
        qa = tl.clamp(a / sa[:, None], -FP8MAX, FP8MAX).to(tl.float8e4nv)

        # --- weight scales: 128x128 blocks ---
        if NUM_NBLK == 1:
            sw_s = tl.maximum(tl.max(tl.abs(w)) * RECIP_FP8MAX, 1e-12)
            qw = tl.clamp(w / sw_s, -FP8MAX, FP8MAX).to(tl.float8e4nv)
            acc += tl.dot(qa, tl.trans(qw), out_dtype=tl.float32) * (sa[:, None] * sw_s)
        else:
            wr = tl.reshape(w, (NUM_NBLK, 128, 128))
            swv = tl.maximum(
                tl.max(tl.max(tl.abs(wr), axis=2), axis=1) * RECIP_FP8MAX, 1e-12
            )
            swb = tl.reshape(
                tl.broadcast_to(swv[:, None], (NUM_NBLK, 128)), (BN,)
            )
            qw = tl.clamp(w / swb[:, None], -FP8MAX, FP8MAX).to(tl.float8e4nv)
            acc += tl.dot(qa, tl.trans(qw), out_dtype=tl.float32) * (
                sa[:, None] * swb[None, :]
            )

        a_ptrs += 128
        w_ptrs += 128

    bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    out = acc.to(tl.bfloat16)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, out, mask=mask_m[:, None] & (offs_n[None, :] < N))


def _cfg(M, N):
    # (BM, BN, GROUP_M, num_warps, num_stages, waves_per_eu)
    if M <= 256:
        return (128, 128, 1, 4, 2, 0)
    if M <= 512:
        return (128, 256, 1, 8, 2, 0)
    return (256, 256, 4, 8, 2, 0)


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor, o_proj_bias: torch.Tensor):
    M, K = attn_output.shape
    N = o_proj_weight.shape[0]

    a = attn_output if attn_output.is_contiguous() else attn_output.contiguous()
    w = o_proj_weight if o_proj_weight.is_contiguous() else o_proj_weight.contiguous()
    bias = o_proj_bias.contiguous()

    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    BM, BN, GM, nw, ns, wpe = _cfg(M, N)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _fused_fp8_blockwise_gemm[grid](
        a, w, bias, out,
        M, N, K,
        a.stride(0), w.stride(0), out.stride(0),
        BM=BM, BN=BN, GROUP_M=GM, NUM_NBLK=BN // 128,
        num_warps=nw, num_stages=ns,
        waves_per_eu=wpe,
    )
    return out
