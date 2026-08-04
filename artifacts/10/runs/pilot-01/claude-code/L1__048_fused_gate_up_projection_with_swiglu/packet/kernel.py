import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# gelu_tanh(x @ gate.T) * (x @ up.T)   fused into a single pass.
#
# Numerics follow reference.py exactly:
#   gate_out = bf16(fp32_acc)                (torch matmul rounds to bf16)
#   g   = fp32(gate_out)
#   act = bf16(0.5*g*(1+tanh(0.7978845608*(g + 0.044715*g^3))))
#   out = bf16(fp32(act) * fp32(up_out_bf16))

SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
COEF = tl.constexpr(0.044715)


@triton.jit
def _fused_gate_up_swiglu(
    X, G, U, O,
    M,
    stride_xm,
    stride_gn,
    stride_un,
    stride_om,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n: tl.constexpr = N // BLOCK_N

    # grouped ordering -> L2 reuse of the weight panels
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_gn > 0)
    tl.assume(stride_un > 0)
    tl.assume(stride_om > 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if EVEN_M:
        offs_xm = offs_m
    else:
        offs_xm = tl.where(offs_m < M, offs_m, 0)

    x_ptrs = X + offs_xm[:, None] * stride_xm + offs_k[None, :]
    g_ptrs = G + offs_n[:, None] * stride_gn + offs_k[None, :]
    u_ptrs = U + offs_n[:, None] * stride_un + offs_k[None, :]

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K // BLOCK_K, num_stages=NUM_STAGES):
        a = tl.load(x_ptrs)
        bg = tl.load(g_ptrs)
        bu = tl.load(u_ptrs)
        acc_g = tl.dot(a, bg.T, acc_g)
        acc_u = tl.dot(a, bu.T, acc_u)
        x_ptrs += BLOCK_K
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K

    # --- epilogue, matching the reference's intermediate rounding ---
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)

    inner = SQRT_2_OVER_PI * (g + COEF * g * g * g)
    act = (0.5 * g) * (1.0 + libdevice.tanh(inner))
    act = act.to(tl.bfloat16).to(tl.float32)

    out = (act * u).to(tl.bfloat16)

    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :]
    if EVEN_M:
        tl.store(o_ptrs, out)
    else:
        tl.store(o_ptrs, out, mask=offs_m[:, None] < M)


def _config(M):
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
    if M <= 256:
        return (64, 128, 64, 8, 4, 2)
    if M <= 1024:
        return (128, 128, 64, 8, 8, 2)
    return (256, 128, 64, 8, 8, 2)


def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    out_shape = x.shape[:-1] + (gate_proj.shape[0],)
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    N = gate_proj.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages = _config(M)

    grid = (triton.cdiv(M, BLOCK_M) * (N // BLOCK_N),)
    _fused_gate_up_swiglu[grid](
        x2, gate_proj, up_proj, out,
        M,
        x2.stride(0),
        gate_proj.stride(0),
        up_proj.stride(0),
        out.stride(0),
        N, K,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
        (M % BLOCK_M) == 0,
        num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.view(out_shape)
