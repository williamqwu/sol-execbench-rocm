import torch
import triton
import triton.language as tl


@triton.jit
def _tile_id(pid, num_m: tl.constexpr, num_n: tl.constexpr, group_m: tl.constexpr):
    num_in_group = group_m * num_n
    group_id = pid // num_in_group
    first_m = group_id * group_m
    actual_group_m = tl.minimum(num_m - first_m, group_m)
    pid_m = first_m + (pid % num_in_group) % actual_group_m
    pid_n = (pid % num_in_group) // actual_group_m
    return pid_m, pid_n


@triton.jit
def _both_mm(
    g_ptr,
    r_ptr,
    w_ptr,
    ga_ptr,
    gw_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    GA_BM: tl.constexpr,
    GA_BN: tl.constexpr,
    GA_BK: tl.constexpr,
    GW_BM: tl.constexpr,
    GW_BN: tl.constexpr,
    GW_BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    ga_num_m: tl.constexpr = tl.cdiv(M, GA_BM)
    ga_num_n: tl.constexpr = tl.cdiv(N, GA_BN)
    ga_tiles: tl.constexpr = ga_num_m * ga_num_n
    pid = tl.program_id(0)

    if pid < ga_tiles:
        ga_pid_m, ga_pid_n = _tile_id(pid, ga_num_m, ga_num_n, GROUP_M)
        ga_offs_m = ga_pid_m * GA_BM + tl.arange(0, GA_BM)
        ga_offs_n = ga_pid_n * GA_BN + tl.arange(0, GA_BN)
        ga_offs_k = tl.arange(0, GA_BK)
        ga_a = g_ptr + ga_offs_m[:, None] * N + ga_offs_k[None, :]
        ga_b = w_ptr + ga_offs_k[:, None] * N + ga_offs_n[None, :]
        ga_acc = tl.zeros((GA_BM, GA_BN), tl.float32)
        for _ in range(0, N, GA_BK):
            ga_acc += tl.dot(tl.load(ga_a), tl.load(ga_b))
            ga_a += GA_BK
            ga_b += GA_BK * N
        tl.store(ga_ptr + ga_offs_m[:, None] * N + ga_offs_n[None, :], ga_acc)
    else:
        gw_pid = pid - ga_tiles
        gw_num_m: tl.constexpr = tl.cdiv(N, GW_BM)
        gw_num_n: tl.constexpr = tl.cdiv(N, GW_BN)
        gw_pid_m, gw_pid_n = _tile_id(gw_pid, gw_num_m, gw_num_n, GROUP_M)
        gw_offs_m = gw_pid_m * GW_BM + tl.arange(0, GW_BM)
        gw_offs_n = gw_pid_n * GW_BN + tl.arange(0, GW_BN)
        gw_offs_k = tl.arange(0, GW_BK)
        gw_a = g_ptr + gw_offs_k[None, :] * N + gw_offs_m[:, None]
        gw_b = r_ptr + gw_offs_k[:, None] * N + gw_offs_n[None, :]
        gw_acc = tl.zeros((GW_BM, GW_BN), tl.float32)
        for _ in range(0, M, GW_BK):
            gw_acc += tl.dot(tl.load(gw_a), tl.load(gw_b))
            gw_a += GW_BK * N
            gw_b += GW_BK * N
        tl.store(gw_ptr + gw_offs_m[:, None] * N + gw_offs_n[None, :], gw_acc)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
):
    batch_size, seq_len, hidden_size = grad_output.shape
    m = batch_size * seq_len

    if m >= 4096:
        grad_output_2d = grad_output.reshape(m, hidden_size)
        ga = grad_output_2d.mm(weight)
        gw = grad_output_2d.t().mm(reshaped.reshape(m, hidden_size))
        return ga.view(batch_size, seq_len, 32, 64).transpose(1, 2), gw

    ga = torch.empty((m, hidden_size), device=grad_output.device, dtype=torch.bfloat16)
    gw = torch.empty((hidden_size, hidden_size), device=grad_output.device, dtype=torch.bfloat16)

    if m <= 512:
        ga_bm, ga_bn, ga_bk = 64, 64, 128
        group_m, num_warps = 8, 8
    elif m <= 1024:
        ga_bm, ga_bn, ga_bk = 64, 128, 64
        group_m, num_warps = 4, 8
    else:
        ga_bm, ga_bn, ga_bk = 128, 128, 64
        group_m, num_warps = 8, 4
    gw_bm, gw_bn, gw_bk = 128, 128, 64
    grid = (triton.cdiv(m, ga_bm) * triton.cdiv(hidden_size, ga_bn) + 16 * 16,)
    _both_mm[grid](
        grad_output,
        reshaped,
        weight,
        ga,
        gw,
        M=m,
        N=hidden_size,
        GA_BM=ga_bm,
        GA_BN=ga_bn,
        GA_BK=ga_bk,
        GW_BM=gw_bm,
        GW_BN=gw_bn,
        GW_BK=gw_bk,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
    ga = ga.view(batch_size, seq_len, 32, 64).transpose(1, 2)
    return ga, gw
