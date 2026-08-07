import torch

E4M3_MAX = 448.0


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 attention output projection with blockwise scaling.

    Computes the same result as the reference (quantize to FP8 with
    BlockWise1x128 scales for the activation and BlockWise128x128 scales for
    the weight, dequantize, then matmul) but replaces the slow fp32 matmul with
    a 3-term bf16 split-matmul that recovers fp32 precision at ~4x the bf16
    throughput.
    """
    M, K = attn_output.shape
    N, _ = o_proj_weight.shape
    KB = K // 128
    NB = N // 128

    a_f32 = attn_output.to(torch.float32)
    wT_f32 = o_proj_weight.T.to(torch.float32)  # (K, N)

    # --- activation scales: BlockWise1x128 -> (M, K//128) ---
    sa = torch.clamp(
        a_f32.reshape(M, KB, 128).abs().amax(dim=2) / E4M3_MAX, min=1e-12
    )
    a_q = (
        a_f32.reshape(M, KB, 128) / sa.unsqueeze(2)
    ).clamp(-E4M3_MAX, E4M3_MAX)
    a_fp8 = a_q.reshape(M, K).to(torch.float8_e4m3fn)

    # --- weight scales: BlockWise128x128 over (K, N) -> (K//128, N//128) ---
    sb = torch.clamp(
        wT_f32.reshape(KB, 128, NB, 128).abs().amax(dim=3).amax(dim=1) / E4M3_MAX,
        min=1e-12,
    )
    wT_q = (
        wT_f32.reshape(KB, 128, NB, 128) / sb.unsqueeze(1).unsqueeze(3)
    ).clamp(-E4M3_MAX, E4M3_MAX)
    w_fp8 = wT_q.reshape(K, N).T.to(torch.float8_e4m3fn)  # (N, K)

    # --- dequantize to fp32 ---
    a_dq = (
        a_fp8.to(torch.float32).reshape(M, KB, 128) * sa.unsqueeze(2)
    ).reshape(M, K)
    w_dq = (
        w_fp8.to(torch.float32).reshape(NB, 128, KB, 128) * sb.t().unsqueeze(1).unsqueeze(3)
    ).reshape(N, K)

    # --- 3-term bf16 split matmul (fp32 accumulate in torch.mm) ---
    a_hi = a_dq.to(torch.bfloat16)
    a_lo = (a_dq - a_hi.to(torch.float32)).to(torch.bfloat16)
    w_hi = w_dq.to(torch.bfloat16)
    w_lo = (w_dq - w_hi.to(torch.float32)).to(torch.bfloat16)

    out = (
        a_hi @ w_hi.t()
        + a_hi @ w_lo.t()
        + a_lo @ w_hi.t()
    )
    return out.to(torch.bfloat16)


if __name__ == "__main__":
    inputs = dict(
        attn_output=torch.randn(4096, 16384, dtype=torch.bfloat16, device="cuda:0"),
        o_proj_weight=torch.randn(7168, 16384, dtype=torch.bfloat16, device="cuda:0"),
    )
    print(run(**inputs).shape)
