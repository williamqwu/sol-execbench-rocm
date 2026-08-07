import torch
import triton
import triton.language as tl


_AUX_STREAM = torch.cuda.Stream()


@triton.jit
def _gating_backward(
    grad_gated_ptr,
    ssm_ptr,
    gate_ptr,
    gate_activated_ptr,
    grad_ssm_ptr,
    grad_gate_ptr,
    n_elements: tl.constexpr,
    USE_SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    grad_gated = tl.load(grad_gated_ptr + offsets, mask=mask)
    ssm = tl.load(ssm_ptr + offsets, mask=mask)
    gate_activated = tl.load(gate_activated_ptr + offsets, mask=mask)

    tl.store(grad_ssm_ptr + offsets, grad_gated * gate_activated, mask=mask)

    # The reference materializes this bfloat16 intermediate before applying
    # the SiLU derivative, so preserve that rounding point explicitly.
    grad_gate_activated = (grad_gated * ssm).to(tl.bfloat16)
    if USE_SILU:
        gate = tl.load(gate_ptr + offsets, mask=mask)
        sigmoid_gate = (1.0 / (1.0 + tl.exp(-gate.to(tl.float32)))).to(tl.bfloat16)
        one_minus_sigmoid = (1.0 - sigmoid_gate).to(tl.bfloat16)
        gate_term = (gate * one_minus_sigmoid).to(tl.bfloat16)
        gate_term = (1.0 + gate_term).to(tl.bfloat16)
        silu_grad = (sigmoid_gate * gate_term).to(tl.bfloat16)
        grad_gate_activated *= silu_grad
    tl.store(grad_gate_ptr + offsets, grad_gate_activated, mask=mask)


@triton.jit
def _matmul_gating_backward(
    grad_output_ptr,
    weight_ptr,
    ssm_ptr,
    gate_ptr,
    gate_activated_ptr,
    grad_ssm_ptr,
    grad_gate_ptr,
    M,
    USE_SILU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Group neighboring M tiles so their projection-weight tiles remain hot.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = 1536 // BLOCK_N
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = grad_output_ptr + offs_m[:, None] * 768 + offs_k[None, :]
    b_ptrs = weight_ptr + offs_k[:, None] * 1536 + offs_n[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in tl.static_range(0, 768, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * 1536

    # torch.mm returns bfloat16.  The following multiplies must consume that
    # rounded value rather than the float32 matrix-product accumulator.
    grad_gated = accumulator.to(tl.bfloat16)
    offsets = offs_m[:, None] * 1536 + offs_n[None, :]
    mask = offs_m[:, None] < M
    ssm = tl.load(ssm_ptr + offsets, mask=mask)
    gate_activated = tl.load(gate_activated_ptr + offsets, mask=mask)
    tl.store(grad_ssm_ptr + offsets, grad_gated * gate_activated, mask=mask)

    grad_gate_activated = (grad_gated * ssm).to(tl.bfloat16)
    if USE_SILU:
        gate = tl.load(gate_ptr + offsets, mask=mask)
        sigmoid_gate = (1.0 / (1.0 + tl.exp(-gate.to(tl.float32)))).to(tl.bfloat16)
        one_minus_sigmoid = (1.0 - sigmoid_gate).to(tl.bfloat16)
        gate_term = (gate * one_minus_sigmoid).to(tl.bfloat16)
        gate_term = (1.0 + gate_term).to(tl.bfloat16)
        silu_grad = (sigmoid_gate * gate_term).to(tl.bfloat16)
        grad_gate_activated *= silu_grad
    tl.store(grad_gate_ptr + offsets, grad_gate_activated, mask=mask)


@torch.no_grad()
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
    m = grad_output.shape[0] * grad_output.shape[1]
    go = grad_output.reshape(m, 768)
    gated = gated_output.reshape(m, 1536)

    overlap = use_silu_gate and m >= 4096
    if overlap:
        current_stream = torch.cuda.current_stream()
        _AUX_STREAM.wait_stream(current_stream)
        with torch.cuda.stream(_AUX_STREAM):
            grad_weight = go.t() @ gated
            grad_bias = go.sum(dim=0)
    else:
        grad_weight = go.t() @ gated
        grad_bias = go.sum(dim=0)

    # For large SiLU cases this independent projection and its sizable
    # epilogue can execute alongside the weight/bias reduction above.
    grad_gated = go @ weight
    grad_ssm = torch.empty_like(ssm_output)
    grad_gate = torch.empty_like(gate)
    n_elements = m * 1536
    _gating_backward[(triton.cdiv(n_elements, 1024),)](
        grad_gated,
        ssm_output,
        gate,
        gate_activated,
        grad_ssm,
        grad_gate,
        n_elements,
        USE_SILU=use_silu_gate,
        BLOCK=1024,
        num_warps=8,
    )
    if overlap:
        current_stream.wait_stream(_AUX_STREAM)
    return grad_ssm, grad_gate, grad_weight, grad_bias
