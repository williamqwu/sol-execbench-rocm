import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Fused kernel:  grad_gated = grad_output @ weight   (M x 768 @ 768 x 1536)
# followed, in the epilogue, by
#     grad_ssm_output = grad_gated * gate_activated
#     grad_gate       = grad_gated * ssm_output  [* silu'(gate)]
# The reference rounds the matmul result to bfloat16 before the elementwise
# multiplies, so we do the same.
# -----------------------------------------------------------------------------


@triton.jit
def _fused_gemm_epi(
    GO, W, GA, SSM, GATE,
    OUT_SSM, OUT_GATE,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_gom, stride_wk, stride_om,
    USE_SILU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

    mask_m = offs_m < M
    offs_am = tl.where(mask_m, offs_m, 0)

    a_ptrs = GO + offs_am[:, None] * stride_gom + offs_k[None, :]
    b_ptrs = W + offs_k[:, None] * stride_wk + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_wk

    # round the projected gradient to bfloat16 exactly like the reference
    g = acc.to(tl.bfloat16).to(tl.float32)

    eptrs = offs_m[:, None] * stride_om + offs_n[None, :]
    emask = mask_m[:, None]

    ga = tl.load(GA + eptrs, mask=emask, other=0.0).to(tl.float32)
    ssm = tl.load(SSM + eptrs, mask=emask, other=0.0).to(tl.float32)

    out_ssm = (g * ga).to(tl.bfloat16)
    tl.store(OUT_SSM + eptrs, out_ssm, mask=emask)

    gga = (g * ssm).to(tl.bfloat16)
    if USE_SILU:
        gt = tl.load(GATE + eptrs, mask=emask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gt).to(tl.bfloat16).to(tl.float32)
        silu_grad = (sig * (1.0 + gt * (1.0 - sig))).to(tl.bfloat16).to(tl.float32)
        out_gate = (gga.to(tl.float32) * silu_grad).to(tl.bfloat16)
    else:
        out_gate = gga
    tl.store(OUT_GATE + eptrs, out_gate, mask=emask)


@triton.jit
def _colsum(GO, OUT, M, K: tl.constexpr, stride_gom,
            BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    nsplit = tl.num_programs(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    rows_per = tl.cdiv(tl.cdiv(M, BLOCK_M), nsplit) * BLOCK_M
    start = pid_m * rows_per
    end = min(start + rows_per, M)
    offs_m = start + tl.arange(0, BLOCK_M)
    while start < end:
        mask = offs_m < end
        v = tl.load(GO + offs_m[:, None] * stride_gom + offs_k[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(v, axis=0)
        offs_m += BLOCK_M
        start += BLOCK_M
    tl.atomic_add(OUT + offs_k, acc)


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
    M = batch_size * seq_len

    go2d = grad_output.reshape(M, hidden_dim)
    gated2d = gated_output.reshape(M, expanded_dim)

    grad_weight = torch.mm(go2d.t(), gated2d)

    fbias = torch.zeros((hidden_dim,), dtype=torch.float32, device=go2d.device)
    nsplit = max(1, min(64, triton.cdiv(M, 512)))
    _colsum[(hidden_dim // 256, nsplit)](
        go2d, fbias, M, hidden_dim, go2d.stride(0),
        BLOCK_M=32, BLOCK_K=256, num_warps=4,
    )
    grad_bias = fbias.to(torch.bfloat16)

    grad_ssm_output = torch.empty_like(ssm_output)
    grad_gate = torch.empty_like(gate)

    ssm2d = ssm_output.reshape(M, expanded_dim)
    ga2d = gate_activated.reshape(M, expanded_dim)
    gate2d = gate.reshape(M, expanded_dim)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(expanded_dim, BLOCK_N),)
    _fused_gemm_epi[grid](
        go2d, weight, ga2d, ssm2d, gate2d,
        grad_ssm_output.reshape(M, expanded_dim),
        grad_gate.reshape(M, expanded_dim),
        M, expanded_dim, hidden_dim,
        go2d.stride(0), weight.stride(0), expanded_dim,
        bool(use_silu_gate),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
        num_warps=8, num_stages=2,
    )

    return grad_ssm_output, grad_gate, grad_weight, grad_bias
