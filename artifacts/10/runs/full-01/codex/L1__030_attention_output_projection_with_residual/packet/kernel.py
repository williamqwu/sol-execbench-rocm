import torch
import triton
import triton.language as tl


@triton.jit
def _project_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    # weight is [N, K], so the desired B tile is weight[n, k].T.
    w_ptrs = weight_ptr + offs_n[None, :] * K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out_offsets = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # torch.matmul first materializes a bf16 result, then performs the add.
    projected = acc.to(tl.bfloat16)
    residual = tl.load(residual_ptr + out_offsets, mask=mask)
    tl.store(out_ptr + out_offsets, projected + residual, mask=mask)


@triton.jit
def _project_residual_transposed_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, H: tl.constexpr,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_F: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_f = tl.cdiv(H, BLOCK_F)
    num_pid_t = tl.cdiv(M, BLOCK_T)
    num_pid_in_group = GROUP_F * num_pid_t
    group_id = pid // num_pid_in_group
    first_pid_f = group_id * GROUP_F
    group_size_f = tl.minimum(num_pid_f - first_pid_f, GROUP_F)
    pid_f = first_pid_f + (pid % num_pid_in_group) % group_size_f
    pid_t = (pid % num_pid_in_group) // group_size_f

    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K)
    w_ptrs = weight_ptr + offs_f[:, None] * H + offs_k[None, :]
    x_ptrs = x_ptr + offs_t[None, :] * H + offs_k[:, None]

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)
    for k in range(0, H, BLOCK_K):
        w = tl.load(w_ptrs, mask=offs_f[:, None] < H, other=0.0)
        x = tl.load(x_ptrs, mask=offs_t[None, :] < M, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K

    out_offsets = offs_t[None, :] * H + offs_f[:, None]
    mask = (offs_f[:, None] < H) & (offs_t[None, :] < M)
    projected = acc.to(tl.bfloat16)
    residual = tl.load(residual_ptr + out_offsets, mask=mask)
    tl.store(out_ptr + out_offsets, projected + residual, mask=mask)


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    m = attn_output.numel() // attn_output.shape[-1]
    hidden = attn_output.shape[-1]

    # At this size a compact tile with a deeper software pipeline minimizes
    # launch and tail costs.
    if m <= 256:
        out = torch.empty_like(attn_output)
        grid = (
            triton.cdiv(m, 32) * triton.cdiv(hidden, 32),
        )
        _project_residual_kernel[grid](
            attn_output, residual, o_proj_weight, out,
            M=m, N=hidden, K=hidden,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, GROUP_M=2,
            num_warps=4, num_stages=3,
            matrix_instr_nonkdim=16,
        )
        return out

    # The irregular 1571-row workload has enough independent 256x64 tiles to
    # occupy the machine exactly once.  Computing W @ X.T also keeps X hot
    # while sweeping output features.
    if 1200 < m < 1800:
        out = torch.empty_like(attn_output)
        grid = (
            triton.cdiv(hidden, 256) * triton.cdiv(m, 64),
        )
        _project_residual_transposed_kernel[grid](
            attn_output, residual, o_proj_weight, out,
            M=m, H=hidden,
            BLOCK_F=256, BLOCK_T=64, BLOCK_K=64, GROUP_F=4,
            num_warps=8, num_stages=3,
        )
        return out

    # rocBLAS is faster for the remaining shapes once Python launch overhead
    # and matrix-core tail utilization are both included.
    else:
        out = torch.addmm(
            residual.view(m, hidden),
            attn_output.view(m, hidden),
            o_proj_weight.t(),
        )
        return out.view_as(attn_output)
