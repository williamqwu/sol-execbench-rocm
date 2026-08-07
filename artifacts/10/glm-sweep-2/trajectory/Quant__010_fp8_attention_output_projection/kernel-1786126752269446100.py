import torch

E4M3_MAX = 448.0


@torch._dynamo.disable
def _split(x: torch.Tensor):
    hi = x.to(torch.bfloat16)
    lo = (x - hi.to(torch.float32)).to(torch.bfloat16)
    return hi, lo


def _run_compiled(attn_output, o_proj_weight, M, K, N, KB, NB):
    a_f32 = attn_output.to(torch.float32)
    wT_f32 = o_proj_weight.T.to(torch.float32)

    sa = torch.clamp(
        a_f32.reshape(M, KB, 128).abs().amax(dim=2) / E4M3_MAX, min=1e-12
    )
    a_fp8 = (
        a_f32.reshape(M, KB, 128) / sa.unsqueeze(2)
    ).clamp(-E4M3_MAX, E4M3_MAX).reshape(M, K).to(torch.float8_e4m3fn)

    sb = torch.clamp(
        wT_f32.reshape(KB, 128, NB, 128).abs().amax(dim=3).amax(dim=1) / E4M3_MAX,
        min=1e-12,
    )
    w_fp8 = (
        wT_f32.reshape(KB, 128, NB, 128) / sb.unsqueeze(1).unsqueeze(3)
    ).clamp(-E4M3_MAX, E4M3_MAX).reshape(K, N).T.to(torch.float8_e4m3fn)

    a_dq = (
        a_fp8.to(torch.float32).reshape(M, KB, 128) * sa.unsqueeze(2)
    ).reshape(M, K)
    w_dq = (
        w_fp8.to(torch.float32).reshape(NB, 128, KB, 128)
        * sb.t().unsqueeze(1).unsqueeze(3)
    ).reshape(N, K)

    a_hi, a_lo = _split(a_dq)
    w_hi, w_lo = _split(w_dq)

    return a_hi @ w_hi.t() + a_hi @ w_lo.t() + a_lo @ w_hi.t()


_run_compiled = torch.compile(_run_compiled, mode="max-autotune-no-cudagraphs")


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    M, K = attn_output.shape
    N, _ = o_proj_weight.shape
    KB = K // 128
    NB = N // 128
    out = _run_compiled(attn_output, o_proj_weight, M, K, N, KB, NB)
    return out.to(torch.bfloat16)


if __name__ == "__main__":
    inputs = dict(
        attn_output=torch.randn(4096, 16384, dtype=torch.bfloat16, device="cuda:0"),
        o_proj_weight=torch.randn(7168, 16384, dtype=torch.bfloat16, device="cuda:0"),
    )
    print(run(**inputs).shape)
