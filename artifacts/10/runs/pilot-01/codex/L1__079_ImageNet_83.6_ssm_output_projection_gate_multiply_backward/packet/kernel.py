import torch
import triton
import triton.language as tl


_STREAMS = None


def _get_streams():
    global _STREAMS
    if _STREAMS is None:
        _STREAMS = (torch.cuda.Stream(), torch.cuda.Stream())
    return _STREAMS


@triton.jit
def _gate_backward_kernel(
    grad_gated_output,
    ssm_output,
    gate,
    gate_activated,
    grad_ssm_output,
    grad_gate,
    n_elements: tl.constexpr,
    use_silu_gate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_gated = tl.load(grad_gated_output + offsets, mask=mask, other=0.0)
    ssm = tl.load(ssm_output + offsets, mask=mask, other=0.0)
    gate_act = tl.load(gate_activated + offsets, mask=mask, other=0.0)

    grad_ssm = (grad_gated * gate_act).to(tl.bfloat16)
    grad_gate_act = (grad_gated * ssm).to(tl.bfloat16)

    if use_silu_gate:
        gate_value = tl.load(gate + offsets, mask=mask, other=0.0)
        sigmoid_gate = tl.sigmoid(gate_value.to(tl.float32)).to(tl.bfloat16)
        one_minus = (1.0 - sigmoid_gate).to(tl.bfloat16)
        gate_term = (gate_value * one_minus).to(tl.bfloat16)
        silu_grad = (sigmoid_gate * (1.0 + gate_term).to(tl.bfloat16)).to(tl.bfloat16)
        grad_gate_value = (grad_gate_act * silu_grad).to(tl.bfloat16)
    else:
        grad_gate_value = grad_gate_act

    tl.store(grad_ssm_output + offsets, grad_ssm, mask=mask)
    tl.store(grad_gate + offsets, grad_gate_value, mask=mask)


@triton.jit
def _matmul_gate_backward_kernel(
    grad_output,
    weight,
    ssm_output,
    gate,
    gate_activated,
    grad_ssm_output,
    grad_gate,
    m_size: tl.constexpr,
    use_silu_gate: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in tl.range(0, 768, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            grad_output + offs_m[:, None] * 768 + k[None, :],
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
        b = tl.load(
            weight + k[:, None] * 1536 + offs_n[None, :],
            mask=offs_n[None, :] < 1536,
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    offsets = offs_m[:, None] * 1536 + offs_n[None, :]
    mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < 1536)
    grad_gated = acc.to(tl.bfloat16)
    ssm = tl.load(ssm_output + offsets, mask=mask, other=0.0)
    gate_act = tl.load(gate_activated + offsets, mask=mask, other=0.0)

    grad_ssm = (grad_gated * gate_act).to(tl.bfloat16)
    grad_gate_act = (grad_gated * ssm).to(tl.bfloat16)

    if use_silu_gate:
        gate_value = tl.load(gate + offsets, mask=mask, other=0.0)
        sigmoid_gate = tl.sigmoid(gate_value.to(tl.float32)).to(tl.bfloat16)
        one_minus = (1.0 - sigmoid_gate).to(tl.bfloat16)
        gate_term = (gate_value * one_minus).to(tl.bfloat16)
        silu_grad = (sigmoid_gate * (1.0 + gate_term).to(tl.bfloat16)).to(tl.bfloat16)
        grad_gate_value = (grad_gate_act * silu_grad).to(tl.bfloat16)
    else:
        grad_gate_value = grad_gate_act

    tl.store(grad_ssm_output + offsets, grad_ssm, mask=mask)
    tl.store(grad_gate + offsets, grad_gate_value, mask=mask)


def _run_triton_matmul_tail(
    grad_output: torch.Tensor,
    ssm_output: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gate_activated: torch.Tensor,
    gated_output: torch.Tensor,
    use_silu_gate: bool,
):
    batch_size, seq_len, hidden_dim = grad_output.shape
    expanded_dim = ssm_output.shape[2]
    m_size = batch_size * seq_len
    grad_output_2d = grad_output.reshape(-1, hidden_dim)
    gated_output_2d = gated_output.reshape(-1, expanded_dim)

    grad_weight = torch.empty_like(weight)
    torch.mm(grad_output_2d.t(), gated_output_2d, out=grad_weight)
    grad_bias = torch.empty_like(bias)
    torch.sum(grad_output_2d, dim=0, out=grad_bias)

    grad_ssm_output = torch.empty_like(ssm_output)
    grad_gate = torch.empty_like(ssm_output)
    _matmul_gate_backward_kernel[(triton.cdiv(m_size, 16), triton.cdiv(expanded_dim, 64))](
        grad_output_2d,
        weight,
        ssm_output,
        gate,
        gate_activated,
        grad_ssm_output,
        grad_gate,
        m_size,
        use_silu_gate,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
    return grad_ssm_output, grad_gate, grad_weight, grad_bias


def _run_split_streams(
    grad_output: torch.Tensor,
    ssm_output: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gate_activated: torch.Tensor,
    gated_output: torch.Tensor,
    use_silu_gate: bool,
):
    batch_size, seq_len, hidden_dim = grad_output.shape
    expanded_dim = ssm_output.shape[2]
    m_size = batch_size * seq_len
    grad_output_2d = grad_output.reshape(-1, hidden_dim)
    gated_output_2d = gated_output.reshape(-1, expanded_dim)

    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty_like(bias)
    grad_gated_output = torch.empty_like(ssm_output)
    grad_ssm_output = torch.empty_like(ssm_output)
    grad_gate = torch.empty_like(ssm_output)

    weight_stream, gated_stream = _get_streams()
    current_stream = torch.cuda.current_stream()
    weight_stream.wait_stream(current_stream)
    gated_stream.wait_stream(current_stream)

    with torch.cuda.stream(weight_stream):
        torch.mm(grad_output_2d.t(), gated_output_2d, out=grad_weight)
        torch.sum(grad_output_2d, dim=0, out=grad_bias)

    with torch.cuda.stream(gated_stream):
        torch.mm(grad_output_2d, weight, out=grad_gated_output.reshape(-1, expanded_dim))
        n_elements = m_size * expanded_dim
        _gate_backward_kernel[(triton.cdiv(n_elements, 1024),)](
            grad_gated_output,
            ssm_output,
            gate,
            gate_activated,
            grad_ssm_output,
            grad_gate,
            n_elements,
            use_silu_gate,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

    grad_weight.record_stream(weight_stream)
    grad_bias.record_stream(weight_stream)
    grad_gated_output.record_stream(gated_stream)
    grad_ssm_output.record_stream(gated_stream)
    grad_gate.record_stream(gated_stream)
    current_stream.wait_stream(weight_stream)
    current_stream.wait_stream(gated_stream)
    return grad_ssm_output, grad_gate, grad_weight, grad_bias


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
    batch_size, seq_len, hidden_dim = grad_output.shape
    expanded_dim = ssm_output.shape[2]
    if batch_size * seq_len >= 10000:
        return _run_split_streams(
            grad_output,
            ssm_output,
            gate,
            weight,
            bias,
            gate_activated,
            gated_output,
            use_silu_gate,
        )

    grad_output_2d = grad_output.reshape(-1, hidden_dim)
    gated_output_2d = gated_output.reshape(-1, expanded_dim)

    grad_weight = torch.empty_like(weight)
    torch.mm(grad_output_2d.t(), gated_output_2d, out=grad_weight)
    grad_bias = torch.empty_like(bias)
    torch.sum(grad_output_2d, dim=0, out=grad_bias)

    grad_gated_output = torch.empty_like(ssm_output)
    torch.mm(grad_output_2d, weight, out=grad_gated_output.reshape(-1, expanded_dim))

    grad_ssm_output = torch.empty_like(ssm_output)
    grad_gate = torch.empty_like(ssm_output)

    n_elements = batch_size * seq_len * expanded_dim
    _gate_backward_kernel[(triton.cdiv(n_elements, 1024),)](
        grad_gated_output,
        ssm_output,
        gate,
        gate_activated,
        grad_ssm_output,
        grad_gate,
        n_elements,
        use_silu_gate,
        BLOCK_SIZE=1024,
        num_warps=4,
    )

    return grad_ssm_output, grad_gate, grad_weight, grad_bias
