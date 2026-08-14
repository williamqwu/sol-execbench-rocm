import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_shuffle(
    hidden,
    grid_thw,
    ln_weight,
    ln_bias,
    shuffled,
    eps,
    NUM_GRIDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    patch = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    cmask = c < 1536

    x = tl.load(hidden + patch * 1536 + c, mask=cmask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / 1536.0
    d = x - mean
    var = tl.sum(d * d, axis=0) / 1536.0
    y = d * tl.rsqrt(var + eps)
    scale = tl.load(ln_weight + c, mask=cmask, other=0.0).to(tl.float32)
    bias = tl.load(ln_bias + c, mask=cmask, other=0.0).to(tl.float32)
    y = y * scale + bias

    # Locate this patch's grid without copying the tiny device-side metadata
    # back to the host.  Grid counts are multiples of four because H and W
    # are both divisible by the merge size.
    grid_start = 0
    selected_start = 0
    selected_h = 0
    selected_w = 0
    for g in range(NUM_GRIDS):
        t = tl.load(grid_thw + 3 * g).to(tl.int32)
        h = tl.load(grid_thw + 3 * g + 1).to(tl.int32)
        w = tl.load(grid_thw + 3 * g + 2).to(tl.int32)
        count = t * h * w
        belongs = (patch >= grid_start) & (patch < grid_start + count)
        selected_start = tl.where(belongs, grid_start, selected_start)
        selected_h = tl.where(belongs, h, selected_h)
        selected_w = tl.where(belongs, w, selected_w)
        grid_start += count

    local_patch = patch - selected_start
    hw = selected_h * selected_w
    ti = local_patch // hw
    rc = local_patch - ti * hw
    row = rc // selected_w
    col = rc - row * selected_w
    merged_w = selected_w // 2
    merged_local = ti * (selected_h // 2) * merged_w + (row // 2) * merged_w + col // 2
    merged_patch = selected_start // 4 + merged_local
    quadrant = (row & 1) * 2 + (col & 1)
    out_col = quadrant * 1536 + c
    tl.store(shuffled + merged_patch * 6144 + out_col, y, mask=cmask)


@triton.jit
def _small_fc1_gelu(
    x,
    weight,
    bias,
    output,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = 6144 // BLOCK_N
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x + offs_m[:, None] * 6144 + offs_k[None, :]
    # The PyTorch weight is [N, K], so this pointer tile is [K, N].
    w_ptrs = weight + offs_k[:, None] + offs_n[None, :] * 6144

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, 6144, BLOCK_K):
        a = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(w_ptrs)
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    acc += tl.load(bias + offs_n)[None, :]
    # linear returns BF16 in the reference, so round before GELU.
    z = acc.to(tl.bfloat16).to(tl.float32)
    z3 = z * z * z
    z = 0.5 * z * (1.0 + tl.tanh(0.7978845608028654 * (z + 0.044715 * z3)))
    out_ptrs = output + offs_m[:, None] * 6144 + offs_n[None, :]
    tl.store(out_ptrs, z, mask=offs_m[:, None] < M)


@torch.no_grad()
def run(
    hidden: torch.Tensor,
    grid_thw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    eps: float,
):
    x = torch.empty(
        (hidden.shape[0] // 4, 6144), dtype=torch.bfloat16, device=hidden.device
    )
    _layer_norm_shuffle[(hidden.shape[0],)](
        hidden,
        grid_thw,
        ln_weight,
        ln_bias,
        x,
        eps,
        NUM_GRIDS=grid_thw.shape[0],
        BLOCK=2048,
        num_warps=1,
        waves_per_eu=4,
    )
    if x.shape[0] == 128:
        activated = torch.empty_like(x)
        _small_fc1_gelu[(2 * (6144 // 64),)](
            x,
            fc1_weight,
            fc1_bias,
            activated,
            128,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=256,
            GROUP_M=4,
            num_warps=8,
            num_stages=2,
            kpack=2,
            matrix_instr_nonkdim=16,
            waves_per_eu=0,
        )
        x = activated
    else:
        x = torch.ops.aten._addmm_activation(
            fc1_bias, x, fc1_weight.t(), use_gelu=True
        )
    return torch.nn.functional.linear(x, fc2_weight, fc2_bias)
