import torch
import triton
import triton.language as tl


E4M3_MAX = 448.0
K_DIM = 7680
N_DIM = 7680
K_BLOCKS = K_DIM // 128


@triton.jit
def _scaled_fp8_gemm_kernel(
    qx,
    qw,
    scale_x,
    scale_w,
    bias,
    out,
    M: tl.constexpr,
    K_TOTAL: tl.constexpr,
    N_TOTAL: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for kb in tl.range(0, NUM_K_BLOCKS):
        k_base = kb * 128
        a = tl.load(
            qx + offs_m[:, None] * K_TOTAL + k_base + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        b = tl.load(
            qw + offs_n[None, :] * K_TOTAL + k_base + offs_k[:, None],
            mask=offs_n[None, :] < N_TOTAL,
            other=0.0,
        )
        partial = tl.dot(a, b, out_dtype=tl.float32)
        sx = tl.load(scale_x + offs_m * NUM_K_BLOCKS + kb, mask=mask_m, other=0.0)
        sw = tl.load(
            scale_w + (offs_n // 128) * NUM_K_BLOCKS + kb,
            mask=offs_n < N_TOTAL,
            other=0.0,
        )
        acc += partial * (sx[:, None] * sw[None, :])

    b = tl.load(bias + offs_n, mask=offs_n < N_TOTAL, other=0.0).to(tl.float32)
    acc += b[None, :]
    tl.store(
        out + offs_m[:, None] * N_TOTAL + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & (offs_n[None, :] < N_TOTAL),
    )


def _scaled_fp8_gemm(qx, qw, scale_x, scale_w, bias, m: int):
    out = torch.empty((m, N_DIM), device=qx.device, dtype=torch.bfloat16)
    block_m = 128
    block_n = 128
    grid = (triton.cdiv(m, block_m), triton.cdiv(N_DIM, block_n))
    _scaled_fp8_gemm_kernel[grid](
        qx,
        qw,
        scale_x,
        scale_w,
        bias,
        out,
        m,
        K_TOTAL=K_DIM,
        N_TOTAL=N_DIM,
        NUM_K_BLOCKS=K_BLOCKS,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor, o_proj_bias: torch.Tensor):
    m, k = attn_output.shape
    n = o_proj_weight.shape[0]

    x_f32 = attn_output.to(torch.float32)
    scale_x = (
        x_f32.reshape(m, k // 128, 128).abs().amax(dim=2).clamp(min=1e-12)
        / E4M3_MAX
    )
    qx = (
        (x_f32.reshape(m, k // 128, 128) / scale_x.unsqueeze(2))
        .clamp(min=-E4M3_MAX, max=E4M3_MAX)
        .reshape(m, k)
        .to(torch.float8_e4m3fn)
    )

    w_f32_t = o_proj_weight.to(torch.float32).T
    scale_w_t = (
        w_f32_t.reshape(k // 128, 128, n // 128, 128)
        .abs()
        .amax(dim=3)
        .amax(dim=1)
        .clamp(min=1e-12)
        / E4M3_MAX
    )
    qw = (
        (w_f32_t.reshape(k // 128, 128, n // 128, 128) / scale_w_t[:, None, :, None])
        .clamp(min=-E4M3_MAX, max=E4M3_MAX)
        .reshape(k, n)
        .T.contiguous()
        .to(torch.float8_e4m3fn)
    )
    scale_w = scale_w_t.T.contiguous()

    return _scaled_fp8_gemm(qx, qw, scale_x, scale_w, o_proj_bias, m)
