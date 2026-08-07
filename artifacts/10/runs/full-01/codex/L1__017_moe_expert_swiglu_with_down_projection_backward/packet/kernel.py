import torch
import triton
import triton.language as tl


HIDDEN = 2048
INTERMEDIATE = 768


_down_stream = torch.cuda.Stream()
_x_stream = torch.cuda.Stream()
_weight_stream = torch.cuda.Stream()


@triton.jit
def _activation_bwd(
    grad_intermediate,
    gate,
    gate_sigmoid,
    gate_silu,
    up,
    grad_pair,
    grad_pair_precise,
    n_elements: tl.constexpr,
    PAIR_WIDTH: tl.constexpr,
    WITH_RESIDUAL: tl.constexpr,
    GATE_ONLY_RESIDUAL: tl.constexpr,
    WITH_FP16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    cols = offsets % 768
    rows = offsets // 768

    di = tl.load(grad_intermediate + offsets, mask=mask).to(tl.float32)
    u = tl.load(up + offsets, mask=mask).to(tl.float32)
    silu = tl.load(gate_silu + offsets, mask=mask).to(tl.float32)
    sig = tl.load(gate_sigmoid + offsets, mask=mask).to(tl.float32)
    g = tl.load(gate + offsets, mask=mask).to(tl.float32)

    grad_gate = di * u
    grad_up = di * silu
    grad_gate = grad_gate * sig
    grad_gate = grad_gate * (1.0 + g * (1.0 - sig))

    pair_offsets = rows * PAIR_WIDTH + cols
    tl.store(grad_pair + pair_offsets, grad_gate, mask=mask)
    tl.store(grad_pair + pair_offsets + 768, grad_up, mask=mask)

    if WITH_RESIDUAL:
        grad_gate_hi = grad_gate.to(tl.bfloat16).to(tl.float32)
        grad_up_hi = grad_up.to(tl.bfloat16).to(tl.float32)
        tl.store(
            grad_pair + pair_offsets + 1536,
            grad_gate - grad_gate_hi,
            mask=mask,
        )
        if not GATE_ONLY_RESIDUAL:
            tl.store(
                grad_pair + pair_offsets + 2304,
                grad_up - grad_up_hi,
                mask=mask,
            )
    if WITH_FP16:
        precise_offsets = rows * 1536 + cols
        tl.store(grad_pair_precise + precise_offsets, grad_gate, mask=mask)
        tl.store(
            grad_pair_precise + precise_offsets + 768,
            grad_up,
            mask=mask,
        )


@triton.jit
def _pack_weights(
    gate_weight,
    up_weight,
    packed_weight,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // 2048
    col = offsets % 2048
    source_row = row % 1536
    is_gate = source_row < 768
    gate_value = tl.load(
        gate_weight + source_row * 2048 + col,
        mask=mask & is_gate,
        other=0.0,
    )
    up_value = tl.load(
        up_weight + (source_row - 768) * 2048 + col,
        mask=mask & ~is_gate,
        other=0.0,
    )
    source = tl.where(is_gate, gate_value, up_value)
    tl.store(packed_weight + offsets, source, mask=mask)


@triton.jit
def _grad_x_fused(
    grad_pair,
    gate_weight,
    up_weight,
    grad_x,
    M: tl.constexpr,
    PAIR_WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(2048, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, PAIR_WIDTH, BLOCK_K):
        ks = k_start + offs_k
        a = tl.load(
            grad_pair + offs_m[:, None] * PAIR_WIDTH + ks[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        source_row = ks % 1536
        is_gate = source_row < 768
        gate_values = tl.load(
            gate_weight + source_row[:, None] * 2048 + offs_n[None, :],
            mask=is_gate[:, None] & (offs_n[None, :] < 2048),
            other=0.0,
        )
        up_values = tl.load(
            up_weight
            + (source_row[:, None] - 768) * 2048
            + offs_n[None, :],
            mask=(~is_gate[:, None]) & (offs_n[None, :] < 2048),
            other=0.0,
        )
        b = tl.where(is_gate[:, None], gate_values, up_values)
        accumulator += tl.dot(a, b)

    tl.store(
        grad_x + offs_m[:, None] * 2048 + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < 2048),
    )


@triton.jit
def _down_activation_fused(
    grad_output,
    down_weight,
    gate,
    gate_sigmoid,
    gate_silu,
    up,
    grad_pair,
    M: tl.constexpr,
    PAIR_WIDTH: tl.constexpr,
    WITH_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(768, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 2048, BLOCK_K):
        ks = k_start + offs_k
        a = tl.load(
            grad_output + offs_m[:, None] * 2048 + ks[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        b = tl.load(down_weight + ks[:, None] * 768 + offs_n[None, :])
        accumulator += tl.dot(a, b)

    saved_offsets = offs_m[:, None] * 768 + offs_n[None, :]
    saved_mask = offs_m[:, None] < M
    u = tl.load(up + saved_offsets, mask=saved_mask).to(tl.float32)
    silu = tl.load(gate_silu + saved_offsets, mask=saved_mask).to(tl.float32)
    sig = tl.load(gate_sigmoid + saved_offsets, mask=saved_mask).to(tl.float32)
    g = tl.load(gate + saved_offsets, mask=saved_mask).to(tl.float32)
    grad_gate = accumulator * u
    grad_up_value = accumulator * silu
    grad_gate = grad_gate * sig
    grad_gate = grad_gate * (1.0 + g * (1.0 - sig))

    pair_offsets = offs_m[:, None] * PAIR_WIDTH + offs_n[None, :]
    tl.store(grad_pair + pair_offsets, grad_gate, mask=saved_mask)
    tl.store(grad_pair + pair_offsets + 768, grad_up_value, mask=saved_mask)
    if WITH_RESIDUAL:
        gate_hi = grad_gate.to(tl.bfloat16).to(tl.float32)
        up_hi = grad_up_value.to(tl.bfloat16).to(tl.float32)
        tl.store(
            grad_pair + pair_offsets + 1536,
            grad_gate - gate_hi,
            mask=saved_mask,
        )
        tl.store(
            grad_pair + pair_offsets + 2304,
            grad_up_value - up_hi,
            mask=saved_mask,
        )


@triton.jit
def _small_weight_grads(
    grad_output,
    x,
    intermediate,
    grad_pair,
    grad_down_weight,
    grad_weights,
    M: tl.constexpr,
    PAIR_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    down_size: tl.constexpr = 2048 * 768
    total_size: tl.constexpr = 3 * 2048 * 768
    is_down = offsets < down_size
    valid = offsets < total_size

    down_h = offsets // 768
    down_i = offsets % 768
    weight_offset = offsets - down_size
    weight_i = weight_offset // 2048
    weight_h = weight_offset % 2048

    down_acc = tl.zeros((BLOCK,), tl.float32)
    weight_acc = tl.zeros((BLOCK,), tl.float32)
    for token in range(0, M):
        go = tl.load(
            grad_output + token * 2048 + down_h,
            mask=is_down,
            other=0.0,
        ).to(tl.float32)
        inter = tl.load(
            intermediate + token * 768 + down_i,
            mask=is_down,
            other=0.0,
        ).to(tl.float32)
        grad = tl.load(
            grad_pair + token * PAIR_WIDTH + weight_i,
            mask=valid & ~is_down,
            other=0.0,
        ).to(tl.float32)
        xv = tl.load(
            x + token * 2048 + weight_h,
            mask=valid & ~is_down,
            other=0.0,
        ).to(tl.float32)
        down_acc += go * inter
        weight_acc += grad * xv

    tl.store(grad_down_weight + offsets, down_acc, mask=is_down)
    tl.store(
        grad_weights + weight_offset,
        weight_acc,
        mask=valid & ~is_down,
    )


@triton.jit
def _small_weight_grads_dot(
    grad_output,
    x,
    intermediate,
    grad_pair,
    grad_down_weight,
    grad_weights,
    M: tl.constexpr,
    PAIR_WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOWN_PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(0)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if pid < DOWN_PROGRAMS:
        num_n = tl.cdiv(768, BLOCK_N)
        pid_m = pid // num_n
        pid_n = pid % num_n
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        for k_start in range(0, M, BLOCK_K):
            tokens = k_start + ks
            a = tl.load(
                grad_output + tokens[None, :] * 2048 + rows[:, None],
                mask=(tokens[None, :] < M) & (rows[:, None] < 2048),
                other=0.0,
            )
            b = tl.load(
                intermediate + tokens[:, None] * 768 + cols[None, :],
                mask=(tokens[:, None] < M) & (cols[None, :] < 768),
                other=0.0,
            )
            accumulator += tl.dot(a, b)
        tl.store(
            grad_down_weight + rows[:, None] * 768 + cols[None, :],
            accumulator,
            mask=(rows[:, None] < 2048) & (cols[None, :] < 768),
        )
    else:
        weight_pid = pid - DOWN_PROGRAMS
        num_n = tl.cdiv(2048, BLOCK_N)
        pid_m = weight_pid // num_n
        pid_n = weight_pid % num_n
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        for k_start in range(0, M, BLOCK_K):
            tokens = k_start + ks
            a = tl.load(
                grad_pair + tokens[None, :] * PAIR_WIDTH + rows[:, None],
                mask=(tokens[None, :] < M) & (rows[:, None] < 1536),
                other=0.0,
            )
            b = tl.load(
                x + tokens[:, None] * 2048 + cols[None, :],
                mask=(tokens[:, None] < M) & (cols[None, :] < 2048),
                other=0.0,
            )
            accumulator += tl.dot(a, b)
        tl.store(
            grad_weights + rows[:, None] * 2048 + cols[None, :],
            accumulator,
            mask=(rows[:, None] < 1536) & (cols[None, :] < 2048),
        )


@torch.no_grad()
def run(
    grad_output,
    x,
    gate_weight,
    up_weight,
    down_weight,
    gate,
    gate_sigmoid,
    gate_silu,
    up,
    intermediate,
):
    num_tokens = grad_output.shape[0]
    with_residual = num_tokens <= 128
    gate_only_residual = num_tokens == 128
    with_fp16 = False
    if gate_only_residual:
        pair_width = 3 * INTERMEDIATE
    else:
        pair_width = 4 * INTERMEDIATE if with_residual else 2 * INTERMEDIATE
    grad_pair = torch.empty(
        (num_tokens, pair_width), device=grad_output.device, dtype=torch.bfloat16
    )
    if with_fp16:
        grad_pair_precise = torch.empty(
            (num_tokens, 2 * INTERMEDIATE),
            device=grad_output.device,
            dtype=torch.float16,
        )
    else:
        grad_pair_precise = grad_pair
    n_activation = num_tokens * INTERMEDIATE

    use_parallel_streams = False
    use_small_weight_kernel = num_tokens <= 4093
    if use_parallel_streams:
        current_stream = torch.cuda.current_stream()
        _down_stream.wait_stream(current_stream)
        _x_stream.wait_stream(current_stream)
        with torch.cuda.stream(_down_stream):
            grad_down_weight = torch.mm(grad_output.t(), intermediate)
        with torch.cuda.stream(_x_stream):
            packed_weight = torch.empty(
                (pair_width, HIDDEN),
                device=grad_output.device,
                dtype=torch.bfloat16,
            )
            n_weights = pair_width * HIDDEN
            _pack_weights[(triton.cdiv(n_weights, 1024),)](
                gate_weight,
                up_weight,
                packed_weight,
                n_elements=n_weights,
                BLOCK=1024,
                num_warps=8,
            )
    else:
        if not use_small_weight_kernel:
            grad_down_weight = torch.mm(grad_output.t(), intermediate)

    # Both operands are exactly representable BF16 values.  The BF16 matrix
    # cores still accumulate the reduction in FP32.
    grad_intermediate = torch.mm(
        grad_output, down_weight, out_dtype=torch.float32
    )
    _activation_bwd[(triton.cdiv(n_activation, 2048),)](
        grad_intermediate,
        gate,
        gate_sigmoid,
        gate_silu,
        up,
        grad_pair,
        grad_pair_precise,
        n_elements=n_activation,
        PAIR_WIDTH=pair_width,
        WITH_RESIDUAL=with_residual,
        GATE_ONLY_RESIDUAL=gate_only_residual,
        WITH_FP16=with_fp16,
        BLOCK=2048,
        num_warps=4,
    )

    # The first half is [grad_gate, grad_up].  A single GEMM forms both weight
    # gradients.  A non-unit leading dimension is directly supported by BLAS.
    grad_pair_hi = grad_pair[:, : 2 * INTERMEDIATE]
    if use_parallel_streams:
        # The activation kernel above is the producer for both final GEMMs.
        # Each wait records that dependency without serializing the GEMMs with
        # one another.
        _weight_stream.wait_stream(current_stream)
        _x_stream.wait_stream(current_stream)
        with torch.cuda.stream(_weight_stream):
            grad_weights = torch.mm(grad_pair_hi.t(), x)
        with torch.cuda.stream(_x_stream):
            grad_x = torch.mm(grad_pair, packed_weight)
        current_stream.wait_stream(_down_stream)
        current_stream.wait_stream(_weight_stream)
        current_stream.wait_stream(_x_stream)
    else:
        if use_small_weight_kernel:
            grad_down_weight = torch.empty(
                (HIDDEN, INTERMEDIATE),
                device=grad_output.device,
                dtype=torch.bfloat16,
            )
            grad_weights = torch.empty(
                (2 * INTERMEDIATE, HIDDEN),
                device=grad_output.device,
                dtype=torch.bfloat16,
            )
            if num_tokens <= 16:
                block_m, block_n, block_k = 64, 64, 16
            elif num_tokens == 128:
                block_m, block_n, block_k = 64, 128, 32
            else:
                block_m, block_n, block_k = 128, 64, 32
            down_programs = triton.cdiv(HIDDEN, block_m) * triton.cdiv(
                INTERMEDIATE, block_n
            )
            weight_programs = triton.cdiv(
                2 * INTERMEDIATE, block_m
            ) * triton.cdiv(HIDDEN, block_n)
            _small_weight_grads_dot[(down_programs + weight_programs,)](
                grad_output,
                x,
                intermediate,
                grad_pair,
                grad_down_weight,
                grad_weights,
                M=num_tokens,
                PAIR_WIDTH=pair_width,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                DOWN_PROGRAMS=down_programs,
                num_warps=4,
                num_stages=2,
            )
        else:
            grad_weights = torch.mm(grad_pair_hi.t(), x)

        # Pack [gate_weight, up_weight], duplicated once when the BF16
        # residual is present, so grad_x is one matrix-core GEMM.
        grad_x_input = grad_pair_precise if with_fp16 else grad_pair
        grad_x_width = 2 * INTERMEDIATE if with_fp16 else pair_width
        packed_weight = torch.empty(
            (grad_x_width, HIDDEN),
            device=grad_output.device,
            dtype=torch.float16 if with_fp16 else torch.bfloat16,
        )
        n_weights = grad_x_width * HIDDEN
        _pack_weights[(triton.cdiv(n_weights, 2048),)](
            gate_weight,
            up_weight,
            packed_weight,
            n_elements=n_weights,
            BLOCK=2048,
            num_warps=4,
        )
        grad_x = torch.mm(grad_x_input, packed_weight)
        if with_fp16:
            grad_x = grad_x.bfloat16()

    return (
        grad_x,
        grad_weights[:INTERMEDIATE],
        grad_weights[INTERMEDIATE:],
        grad_down_weight,
    )
