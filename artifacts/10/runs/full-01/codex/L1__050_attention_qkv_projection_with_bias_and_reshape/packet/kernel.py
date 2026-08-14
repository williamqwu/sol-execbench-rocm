import torch
import triton
import triton.language as tl


@triton.jit
def _project_tile(x, weight, bias, output, pid_m, local_n,
                  M: tl.constexpr, OUTPUT_STRIDE: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = local_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, 640, BLOCK_K):
        a = tl.load(x + offs_m[:, None] * 640 + (k + offs_k[None, :]),
                    mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(weight + offs_n[None, :] * 640 + (k + offs_k[:, None]))
        acc = tl.dot(a, b, acc)

    # torch.matmul materializes bfloat16 before the separate bias addition.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    rounded += tl.load(bias + offs_n)[None, :].to(tl.float32)
    tl.store(output + offs_m[:, None] * OUTPUT_STRIDE + offs_n[None, :],
             rounded, mask=offs_m[:, None] < M)


@triton.jit
def _qkv_kernel(
    x, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
    q_out, k_out, v_out,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = 1536 // BLOCK_N
    programs_per_group = GROUP_M * num_n
    group_id = pid // programs_per_group
    first_m = group_id * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_in_group = pid % programs_per_group
    pid_m = first_m + (pid_in_group % group_m)
    pid_n = pid_in_group // group_m

    q_tiles: tl.constexpr = 1024 // BLOCK_N
    kv_tiles: tl.constexpr = 256 // BLOCK_N
    if pid_n < q_tiles:
        _project_tile(x, q_weight, q_bias, q_out, pid_m, pid_n,
                      M, 1024, BLOCK_M, BLOCK_N, BLOCK_K)
    elif pid_n < q_tiles + kv_tiles:
        _project_tile(x, k_weight, k_bias, k_out, pid_m, pid_n - q_tiles,
                      M, 256, BLOCK_M, BLOCK_N, BLOCK_K)
    else:
        _project_tile(x, v_weight, v_bias, v_out, pid_m,
                      pid_n - q_tiles - kv_tiles,
                      M, 256, BLOCK_M, BLOCK_N, BLOCK_K)


def _run_triton(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
                block_m=64, block_n=64, block_k=64, num_warps=4,
                num_stages=2, group_m=8, waves_per_eu=1,
                matrix_instr_nonkdim=0, kpack=1):
    batch_size, seq_len, _ = hidden_states.shape
    m = batch_size * seq_len
    query_states = torch.empty((batch_size, seq_len, 4, 256),
                               device=hidden_states.device, dtype=hidden_states.dtype)
    key_states = torch.empty((batch_size, seq_len, 1, 256),
                             device=hidden_states.device, dtype=hidden_states.dtype)
    value_states = torch.empty_like(key_states)
    grid = (triton.cdiv(m, block_m) * (1536 // block_n),)
    _qkv_kernel[grid](
        hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
        query_states, key_states, value_states, M=m,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=num_warps, num_stages=num_stages, waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=matrix_instr_nonkdim, kpack=kpack,
    )
    return query_states, key_states, value_states


@triton.jit
def _copy_projection(weight, bias, all_weight, all_bias, local_pid,
                     WEIGHT_OFFSET: tl.constexpr, BIAS_OFFSET: tl.constexpr,
                     OUT_DIM: tl.constexpr, COPY_BLOCK: tl.constexpr):
    offs = local_pid * COPY_BLOCK + tl.arange(0, COPY_BLOCK)
    mask = offs < OUT_DIM * 640
    tl.store(all_weight + WEIGHT_OFFSET + offs,
             tl.load(weight + offs, mask=mask), mask=mask)
    if local_pid == 0:
        bias_offs = tl.arange(0, 1024)
        bias_mask = bias_offs < OUT_DIM
        tl.store(all_bias + BIAS_OFFSET + bias_offs,
                 tl.load(bias + bias_offs, mask=bias_mask), mask=bias_mask)


@triton.jit
def _concat_qkv_kernel(q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
                       all_weight, all_bias, COPY_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    q_blocks: tl.constexpr = tl.cdiv(1024 * 640, COPY_BLOCK)
    kv_blocks: tl.constexpr = tl.cdiv(256 * 640, COPY_BLOCK)
    if pid < q_blocks:
        _copy_projection(q_weight, q_bias, all_weight, all_bias, pid,
                         0, 0, 1024, COPY_BLOCK)
    elif pid < q_blocks + kv_blocks:
        _copy_projection(k_weight, k_bias, all_weight, all_bias, pid - q_blocks,
                         1024 * 640, 1024, 256, COPY_BLOCK)
    else:
        _copy_projection(v_weight, v_bias, all_weight, all_bias,
                         pid - q_blocks - kv_blocks,
                         1280 * 640, 1280, 256, COPY_BLOCK)


def _run_concat(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
                copy_block=4096, copy_warps=8):
    batch_size, seq_len, _ = hidden_states.shape
    all_weight = torch.empty((1536, 640), device=hidden_states.device,
                             dtype=hidden_states.dtype)
    all_bias = torch.empty((1536,), device=hidden_states.device,
                           dtype=hidden_states.dtype)
    grid = (triton.cdiv(1024 * 640, copy_block)
            + 2 * triton.cdiv(256 * 640, copy_block),)
    _concat_qkv_kernel[grid](
        q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
        all_weight, all_bias, COPY_BLOCK=copy_block, num_warps=copy_warps,
    )
    output = torch.empty((batch_size * seq_len, 1536),
                         device=hidden_states.device, dtype=hidden_states.dtype)
    torch.addmm(all_bias, hidden_states.view(-1, 640), all_weight.t(), out=output)
    return (
        output[:, :1024].view(batch_size, seq_len, 4, 256),
        output[:, 1024:1280].view(batch_size, seq_len, 1, 256),
        output[:, 1280:].view(batch_size, seq_len, 1, 256),
    )


@torch.no_grad()
def run(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias):
    m = hidden_states.shape[0] * hidden_states.shape[1]
    if m <= 512:
        config = (64, 64, 64, 4, 4, 2, 3, 0)
    elif m <= 1024:
        config = (64, 128, 64, 8, 3, 8, 1, 0)
    elif m <= 2200:
        config = (128, 128, 32, 4, 3, 4, 4, 32)
    elif m <= 4096:
        config = (128, 64, 64, 8, 3, 8, 1, 0)
    else:
        return _run_concat(
            hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
            copy_block=4096, copy_warps=4,
        )
    return _run_triton(
        hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
        block_m=config[0], block_n=config[1], block_k=config[2],
        num_warps=config[3], num_stages=config[4],
        group_m=config[5], waves_per_eu=config[6],
        matrix_instr_nonkdim=config[7],
    )
