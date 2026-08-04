import torch
import torch.nn.functional as F
import triton
import triton.language as tl


E4M3_MAX = 448.0


@triton.jit
def _fp8_block_gemm_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, 128):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :])
        b = tl.load(b_ptr + offs_n[None, :] * K + (k0 + offs_k)[:, None])
        dots = tl.dot(a, b, out_dtype=tl.float32)
        kb = k0 // 128
        sa = tl.load(sa_ptr + offs_m * (K // 128) + kb)
        sb = tl.load(sb_ptr + (offs_n // 128) * (K // 128) + kb)
        acc += dots * sa[:, None] * sb[None, :]
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def _quant_act_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * 128 + tl.arange(0, 128)
    vals = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :]).to(tl.float32)
    row_max = tl.max(tl.abs(vals), axis=1)
    scale = tl.maximum(row_max / 448.0, 1.0e-12)
    qvals = vals / scale[:, None]
    qvals = tl.minimum(tl.maximum(qvals, -448.0), 448.0)
    tl.store(q_ptr + offs_m[:, None] * K + offs_k[None, :], qvals)
    tl.store(s_ptr + offs_m * (K // 128) + pid_k, scale)


@triton.jit
def _act_amax_kernel(
    x_ptr,
    amax_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * 128 + tl.arange(0, 128)
    vals = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :]).to(tl.float32)
    row_max = tl.max(tl.abs(vals), axis=1)
    tl.store(amax_ptr + offs_m * (K // 128) + pid_k, row_max)


@triton.jit
def _quant_weight_kernel(
    w_ptr,
    q_ptr,
    s_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = pid_k * 128 + tl.arange(0, 128)
    vals = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :]).to(tl.float32)
    row_max = tl.max(tl.abs(vals), axis=1)
    tile_max = tl.max(row_max, axis=0)
    scale = tl.maximum(tile_max / 448.0, 1.0e-12)
    qvals = vals / scale
    qvals = tl.minimum(tl.maximum(qvals, -448.0), 448.0)
    tl.store(q_ptr + offs_n[:, None] * K + offs_k[None, :], qvals)
    tl.store(s_ptr + pid_n * (K // 128) + pid_k, scale)


@triton.jit
def _weight_amax_kernel(
    w_ptr,
    amax_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = pid_k * 128 + tl.arange(0, 128)
    vals = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :]).to(tl.float32)
    row_max = tl.max(tl.abs(vals), axis=1)
    tile_max = tl.max(row_max, axis=0)
    tl.store(amax_ptr + pid_n * (K // 128) + pid_k, tile_max)


def _act_scales_triton(x: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    k = x.shape[1]
    amax = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
    _act_amax_kernel[(triton.cdiv(m, 16), k // 128)](
        x,
        amax,
        m,
        k,
        BLOCK_M=16,
        num_warps=4,
    )
    return torch.clamp(amax / E4M3_MAX, min=1.0e-12)


def _weight_scales_triton(weight: torch.Tensor) -> torch.Tensor:
    n = weight.shape[0]
    k = weight.shape[1]
    amax = torch.empty((n // 128, k // 128), device=weight.device, dtype=torch.float32)
    _weight_amax_kernel[(n // 128, k // 128)](
        weight,
        amax,
        n,
        k,
        num_warps=8,
    )
    return torch.clamp(amax / E4M3_MAX, min=1.0e-12)


def _quant_act_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    k = x.shape[1]
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
    _quant_act_kernel[(triton.cdiv(m, 16), k // 128)](
        x,
        q,
        scales,
        m,
        k,
        BLOCK_M=16,
        num_warps=4,
    )
    return q, scales


def _quant_weight_triton(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = weight.shape[0]
    k = weight.shape[1]
    q = torch.empty((n, k), device=weight.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((n // 128, k // 128), device=weight.device, dtype=torch.float32)
    _quant_weight_kernel[(n // 128, k // 128)](
        weight,
        q,
        scales,
        n,
        k,
        num_warps=8,
    )
    return q, scales


def _fp8_block_gemm(qx: torch.Tensor, qw: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    m = qx.shape[0]
    n = qw.shape[0]
    k = qx.shape[1]
    out = torch.empty((m, n), device=qx.device, dtype=torch.bfloat16)
    if n == 2048 and m < 2048:
        block_m = 64
        block_n = 64
    elif n == 7168 and m == 256:
        block_m = 64
        block_n = 64
    else:
        block_m = 128
        block_n = 128
    _fp8_block_gemm_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        qx,
        qw,
        sx,
        sw,
        out,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


def _act_scales(x_f32: torch.Tensor) -> torch.Tensor:
    m = x_f32.shape[0]
    k = x_f32.shape[1]
    return torch.clamp(
        x_f32.reshape(m, k // 128, 128).abs().amax(dim=2) / E4M3_MAX,
        min=1.0e-12,
    )


def _weight_scales_transposed(weight_f32: torch.Tensor) -> torch.Tensor:
    wt = weight_f32.T
    k = wt.shape[0]
    n = wt.shape[1]
    scales = torch.clamp(
        wt.reshape(k // 128, 128, n // 128, 128).abs().amax(dim=3).amax(dim=1)
        / E4M3_MAX,
        min=1.0e-12,
    )
    return scales.T.contiguous()


def _apply_act_quant(x_f32: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    m = x_f32.shape[0]
    k = x_f32.shape[1]
    x_scaled = x_f32.reshape(m, k // 128, 128) / scales.unsqueeze(2)
    x_scaled = torch.clamp(x_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    return x_scaled.reshape(m, k).contiguous().to(torch.float8_e4m3fn)


def _apply_weight_quant(weight_f32: torch.Tensor, scales_nk: torch.Tensor) -> torch.Tensor:
    n = weight_f32.shape[0]
    k = weight_f32.shape[1]
    scales_kn = scales_nk.T
    wt_scaled = weight_f32.T.reshape(k // 128, 128, n // 128, 128) / scales_kn.unsqueeze(1).unsqueeze(3)
    wt_scaled = torch.clamp(wt_scaled, min=-E4M3_MAX, max=E4M3_MAX)
    return wt_scaled.reshape(k, n).T.contiguous().to(torch.float8_e4m3fn)


def _dequant_act(qx: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    m = qx.shape[0]
    k = qx.shape[1]
    return (qx.to(torch.float32).reshape(m, k // 128, 128) * scales.unsqueeze(2)).reshape(m, k)


def _dequant_weight(qw: torch.Tensor, scales_nk: torch.Tensor) -> torch.Tensor:
    n = qw.shape[0]
    k = qw.shape[1]
    return (
        qw.to(torch.float32).reshape(n // 128, 128, k // 128, 128)
        * scales_nk.unsqueeze(1).unsqueeze(3)
    ).reshape(n, k)


def _fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    sx = _act_scales_triton(x)
    qx = _apply_act_quant(x_f32, sx)
    return _fp8_linear_from_quant(qx, sx, weight)


def _fp8_linear_from_quant(qx: torch.Tensor, sx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight_f32 = weight.to(torch.float32)
    sw = _weight_scales_triton(weight)
    qw = _apply_weight_quant(weight_f32, sw)
    return _fp8_block_gemm(qx, qw, sx, sw)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    hidden_f32 = hidden_states.to(torch.float32)
    hidden_scales = _act_scales_triton(hidden_states)
    q_hidden = _apply_act_quant(hidden_f32, hidden_scales)
    gate = _fp8_linear_from_quant(q_hidden, hidden_scales, gate_proj_weight)
    up = _fp8_linear_from_quant(q_hidden, hidden_scales, up_proj_weight)
    intermediate = F.silu(gate) * up
    return _fp8_linear(intermediate, down_proj_weight)
