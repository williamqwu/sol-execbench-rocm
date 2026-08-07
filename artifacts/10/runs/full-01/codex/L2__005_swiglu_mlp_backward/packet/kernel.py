import torch
import triton
import triton.language as tl
import aiter


# AITER exposes hipBLASLt's solution-index interface.  Creating the handle at
# module load keeps all descriptor/library setup outside the measured call.
aiter.hipb_create_extension()


@triton.jit
def _swiglu_pointwise_backward(
    grad_gated_ptr,
    gate_ptr,
    up_ptr,
    activated_ptr,
    gated_ptr,
    grad_pair_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    grad_gated = tl.load(grad_gated_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    activated = tl.load(activated_ptr + offsets, mask=mask).to(tl.float32)

    # Every explicit BF16 conversion below corresponds to a materialized BF16
    # tensor in the reference.  In particular, grad_activated must round before
    # it is multiplied by the FP32 SiLU derivative.
    gated = (activated * up).to(tl.bfloat16)
    grad_activated = (grad_gated * up).to(tl.bfloat16)
    grad_up = (grad_gated * activated).to(tl.bfloat16)

    sigmoid = tl.sigmoid(gate)
    silu_grad = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_gate = (grad_activated.to(tl.float32) * silu_grad).to(tl.bfloat16)

    tl.store(gated_ptr + offsets, gated, mask=mask)
    tl.store(grad_pair_ptr + offsets, grad_gate, mask=mask)
    tl.store(grad_pair_ptr + n_elements + offsets, grad_up, mask=mask)


@triton.jit
def _fused_grad_x(
    grad_pair_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    grad_x_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_a_ptrs = grad_pair_ptr + offs_m[:, None] * K + offs_k[None, :]
    up_a_ptrs = gate_a_ptrs + M * K
    gate_b_ptrs = gate_weight_ptr + offs_k[:, None] * N + offs_n[None, :]
    up_b_ptrs = up_weight_ptr + offs_k[:, None] * N + offs_n[None, :]

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        gate_a = tl.load(gate_a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        up_a = tl.load(up_a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        gate_b = tl.load(gate_b_ptrs)
        up_b = tl.load(up_b_ptrs)
        gate_acc += tl.dot(gate_a, gate_b)
        up_acc += tl.dot(up_a, up_b)
        gate_a_ptrs += BLOCK_K
        up_a_ptrs += BLOCK_K
        gate_b_ptrs += BLOCK_K * N
        up_b_ptrs += BLOCK_K * N

    # The reference materializes both matmuls as BF16 before adding them.
    gate_rounded = gate_acc.to(tl.bfloat16).to(tl.float32)
    up_rounded = up_acc.to(tl.bfloat16).to(tl.float32)
    result = (gate_rounded + up_rounded).to(tl.bfloat16)
    out_ptrs = grad_x_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, result, mask=offs_m[:, None] < M)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    activated_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    hidden_size = grad_output.shape[-1]
    intermediate_size = gate_output.shape[-1]
    grad_output_2d = grad_output.reshape(-1, hidden_size)
    x_2d = x.reshape(-1, hidden_size)
    tokens = grad_output_2d.shape[0]
    n_elements = tokens * intermediate_size

    # Shape-tuned hipBLASLt solutions on gfx950.  The M=128 forward solution
    # is the sole case where PyTorch's default heuristic is faster.  The
    # fallback keeps arbitrary token counts correct as well.
    if tokens == 128:
        grad_gated_output = torch.mm(grad_output_2d, down_weight)
    else:
        forward_solution = {
            256: 436613,
            512: 436613,
            1024: 436551,
            2048: 436627,
            2984: 436639,
            3376: 436761,
            4096: 436627,
            6284: 436613,
            8192: 436627,
        }.get(tokens, 436551)
        grad_gated_output = aiter.hipb_mm(
            grad_output_2d, down_weight, forward_solution
        )

    gated_output = torch.empty_like(gate_output).view(tokens, intermediate_size)
    grad_pair = torch.empty(
        (2, tokens, intermediate_size),
        device=gate_output.device,
        dtype=gate_output.dtype,
    )
    _swiglu_pointwise_backward[(triton.cdiv(n_elements, 512),)](
        grad_gated_output,
        gate_output,
        up_output,
        activated_gate,
        gated_output,
        grad_pair,
        n_elements,
        BLOCK=512,
        num_warps=4,
    )
    grad_down_weight = aiter.hipb_mm(
        grad_output_2d.t(), gated_output, 435613
    )

    grad_gate_weight = aiter.hipb_mm(grad_pair[0].t(), x_2d, 435613)
    grad_up_weight = aiter.hipb_mm(grad_pair[1].t(), x_2d, 435613)

    grad_x_gate = aiter.hipb_mm(grad_pair[0], gate_weight, 436555)
    grad_x_up = aiter.hipb_mm(grad_pair[1], up_weight, 436555)
    grad_x_gate.add_(grad_x_up)

    return (
        grad_x_gate.view_as(grad_output),
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
    )
