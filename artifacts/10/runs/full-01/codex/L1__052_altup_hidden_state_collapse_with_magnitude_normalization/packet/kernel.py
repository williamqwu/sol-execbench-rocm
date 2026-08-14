import torch
import triton
import triton.language as tl


HIDDEN_SIZE = 3072


@triton.jit
def _pack_weights_kernel(weight_1_ptr, weight_2_ptr, packed_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    weight_1 = tl.load(weight_1_ptr + offsets, mask=mask)
    weight_2 = tl.load(weight_2_ptr + offsets, mask=mask)
    tl.store(packed_ptr + offsets, weight_1, mask=mask)
    tl.store(packed_ptr + N + offsets, weight_2, mask=mask)


@triton.jit
def _project_kernel(
    hidden_ptr,
    weight_1_ptr,
    weight_2_ptr,
    projected_ptr,
    rows,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(H, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    batch = pid // tiles_per_batch
    tile = pid - batch * tiles_per_batch

    # Group neighboring M tiles so that they reuse the same weight tiles.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile % num_pid_in_group) % group_size_m
    pid_n = (tile % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    stream_ptr = hidden_ptr + (batch + 1) * rows * H
    weight_ptr = tl.where(batch == 0, weight_1_ptr, weight_2_ptr)
    a_ptrs = stream_ptr + offs_m[:, None] * H + offs_k[None, :]
    # Weights are [out_features, in_features], so this directly loads W.T.
    b_ptrs = weight_ptr + offs_n[None, :] * H + offs_k[:, None]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, H, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < rows, other=0.0)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    output_ptrs = projected_ptr + batch * rows * H + offs_m[:, None] * H + offs_n[None, :]
    tl.store(output_ptrs, accumulator, mask=offs_m[:, None] < rows)


@triton.jit
def _collapse_kernel(
    first_ptr,
    projected_1_ptr,
    projected_2_ptr,
    output_ptr,
    epsilon,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    offsets = row * H + cols

    first = tl.load(first_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    projected_1 = tl.load(projected_1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    projected_2 = tl.load(projected_2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    inv_h = 1.0 / H
    target = tl.sqrt(tl.sum(first * first, axis=0) * inv_h)
    magnitude_1 = tl.sqrt(tl.maximum(tl.sum(projected_1 * projected_1, axis=0) * inv_h, epsilon))
    magnitude_2 = tl.sqrt(tl.maximum(tl.sum(projected_2 * projected_2, axis=0) * inv_h, epsilon))

    result = (first + projected_1 * (target / magnitude_1) + projected_2 * (target / magnitude_2)) * (1.0 / 3.0)
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.no_grad()
def run(hidden_states, unembed_proj_1, unembed_proj_2, epsilon):
    rows = hidden_states.numel() // (3 * HIDDEN_SIZE)
    stream_1 = hidden_states[1].reshape(rows, HIDDEN_SIZE)
    stream_2 = hidden_states[2].reshape(rows, HIDDEN_SIZE)

    if rows in (128, 256, 422, 613):
        # Small M leaves a library GEMM launch badly under-occupied.  Project
        # both streams in one launch, using narrower tiles to expose enough
        # independent work across the 256 CUs.
        projected = torch.empty(
            (2, rows, HIDDEN_SIZE),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        if rows == 128:
            block_m, block_k, waves_per_eu = 64, 128, 1
        elif rows == 256:
            block_m, block_k, waves_per_eu = 64, 64, 2
        else:
            block_m, block_k, waves_per_eu = 128, 64, 2
        block_n = 64
        grid = (
            2
            * triton.cdiv(rows, block_m)
            * triton.cdiv(HIDDEN_SIZE, block_n),
        )
        _project_kernel[grid](
            hidden_states,
            unembed_proj_1,
            unembed_proj_2,
            projected,
            rows,
            H=HIDDEN_SIZE,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=4,
            num_stages=2,
            waves_per_eu=waves_per_eu,
            matrix_instr_nonkdim=16,
        )
        projected_1, projected_2 = projected[0], projected[1]
        collapse_warps = 4
    elif rows == 8192:
        # At this shape a single batched launch is faster even after packing
        # the independently supplied weights.  GEMM still accumulates in FP32;
        # only its materialized output is rounded to BF16.
        packed_weights = torch.empty(
            (2, HIDDEN_SIZE, HIDDEN_SIZE),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
        _pack_weights_kernel[(2304,)](
            unembed_proj_1,
            unembed_proj_2,
            packed_weights,
            N=HIDDEN_SIZE * HIDDEN_SIZE,
            BLOCK=4096,
            num_warps=4,
        )
        projected = torch.bmm(
            hidden_states[1:].reshape(2, rows, HIDDEN_SIZE),
            packed_weights.transpose(1, 2),
        )
        projected_1, projected_2 = projected[0], projected[1]
        collapse_warps = 4
    elif rows in (2984, 4096) or rows >= 16384:
        # For these products hipBLASLt plus the lower-traffic collapse is
        # faster when the materialized projection is BF16.
        projected_1 = torch.mm(stream_1, unembed_proj_1.t())
        projected_2 = torch.mm(stream_2, unembed_proj_2.t())
        collapse_warps = 8 if rows == 17312 else 4
    else:
        # BF16 inputs are represented exactly in FP32.  This dispatches the
        # products to matrix units while retaining an FP32 accumulator output.
        projected_1 = torch.mm(stream_1, unembed_proj_1.t(), out_dtype=torch.float32)
        projected_2 = torch.mm(stream_2, unembed_proj_2.t(), out_dtype=torch.float32)
        collapse_warps = 8

    output = torch.empty_like(hidden_states[0])
    _collapse_kernel[(rows,)](
        hidden_states,
        projected_1,
        projected_2,
        output,
        epsilon,
        H=HIDDEN_SIZE,
        BLOCK=4096,
        num_warps=collapse_warps,
    )
    return output
