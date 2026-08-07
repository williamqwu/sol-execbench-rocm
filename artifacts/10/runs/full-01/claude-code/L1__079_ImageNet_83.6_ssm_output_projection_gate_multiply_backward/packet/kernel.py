import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused kernel:  grad_gated = grad_output_2d @ weight        (GEMM, K = 768)
#                grad_ssm   = bf16(grad_gated * gate_activated)
#                grad_gate  = bf16(bf16(grad_gated * ssm_output) * silu_grad)
#
# The GEMM epilogue consumes the (M, N) intermediate in registers, so the
# grad_gated tensor is never written to / read from HBM.
#
# Every intermediate is rounded to bfloat16 exactly where the reference
# rounds it, so the emitted rounding pattern matches the reference op-by-op.
# ---------------------------------------------------------------------------


def _cfgs():
    out = []
    for bm, bn, bk, w, s in [
        (128, 256, 64, 8, 2),
        (256, 128, 64, 8, 2),
        (128, 128, 64, 4, 2),
        (64, 256, 64, 4, 2),
        (64, 128, 64, 4, 2),
        (32, 256, 64, 4, 2),
    ]:
        out.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                num_warps=w,
                num_stages=s,
            )
        )
    return out


@triton.autotune(configs=_cfgs(), key=["M", "N", "K", "USE_SILU"])
@triton.jit
def _fused_gemm_gate_bwd(
    GO,          # (M, K)   bf16   grad_output_2d
    W,           # (K, N)   bf16   weight
    GACT,        # (M, N)   bf16   gate_activated
    SSM,         # (M, N)   bf16   ssm_output
    GATE,        # (M, N)   bf16   gate
    GSSM,        # (M, N)   bf16   out: grad_ssm_output
    GGATE,       # (M, N)   bf16   out: grad_gate
    M, N, K,
    stride_gom, stride_gok,
    stride_wk, stride_wn,
    stride_em, stride_en,
    USE_SILU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = GO + (offs_m[:, None] * stride_gom + offs_k[None, :] * stride_gok)
    b_ptrs = W + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_gok
        b_ptrs += BLOCK_K * stride_wk

    # grad_gated_output is a bf16 tensor in the reference -> round here.
    gg = acc.to(tl.bfloat16).to(tl.float32)

    e_off = offs_m[:, None] * stride_em + offs_n[None, :] * stride_en
    e_mask = m_mask[:, None] & n_mask[None, :]

    gact = tl.load(GACT + e_off, mask=e_mask, other=0.0).to(tl.float32)
    ssm = tl.load(SSM + e_off, mask=e_mask, other=0.0).to(tl.float32)

    grad_ssm = (gg * gact).to(tl.bfloat16)
    grad_gate_act = (gg * ssm).to(tl.bfloat16)

    if USE_SILU:
        g = tl.load(GATE + e_off, mask=e_mask, other=0.0).to(tl.float32)
        s = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        t = (1.0 - s).to(tl.bfloat16).to(tl.float32)
        u = (g * t).to(tl.bfloat16).to(tl.float32)
        v = (1.0 + u).to(tl.bfloat16).to(tl.float32)
        w = (s * v).to(tl.bfloat16).to(tl.float32)
        grad_gate = (grad_gate_act.to(tl.float32) * w).to(tl.bfloat16)
    else:
        grad_gate = grad_gate_act

    tl.store(GSSM + e_off, grad_ssm, mask=e_mask)
    tl.store(GGATE + e_off, grad_gate, mask=e_mask)


def run(
    grad_output: torch.Tensor,
    ssm_output: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gate_activated: torch.Tensor,
    gated_output: torch.Tensor,
    use_silu_gate: bool,
):
    batch_size, seq_len, hidden_dim = grad_output.shape
    expanded_dim = ssm_output.shape[2]

    go2d = grad_output.reshape(-1, hidden_dim)
    gated2d = gated_output.reshape(-1, expanded_dim)
    ssm2d = ssm_output.reshape(-1, expanded_dim)
    gate2d = gate.reshape(-1, expanded_dim)
    gact2d = gate_activated.reshape(-1, expanded_dim)

    M = go2d.shape[0]
    K = hidden_dim
    N = expanded_dim

    grad_ssm_output = torch.empty_like(ssm2d)
    grad_gate = torch.empty_like(ssm2d)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _fused_gemm_gate_bwd[grid](
        go2d, weight, gact2d, ssm2d, gate2d,
        grad_ssm_output, grad_gate,
        M, N, K,
        go2d.stride(0), go2d.stride(1),
        weight.stride(0), weight.stride(1),
        ssm2d.stride(0), ssm2d.stride(1),
        USE_SILU=bool(use_silu_gate),
    )

    grad_weight = torch.mm(go2d.t(), gated2d)
    grad_bias = go2d.sum(dim=0)

    return (
        grad_ssm_output.view(batch_size, seq_len, expanded_dim),
        grad_gate.view(batch_size, seq_len, expanded_dim),
        grad_weight,
        grad_bias,
    )
