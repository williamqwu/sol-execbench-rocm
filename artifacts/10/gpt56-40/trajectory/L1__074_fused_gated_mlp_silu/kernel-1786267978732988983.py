import torch
import triton
import triton.language as tl


@triton.jit
def _mul_gate_inplace(gate_up, sigmoid_gate, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    sigmoid_offset = row * 4096 + col
    gate_offset = row * 8192 + col
    up_offset = gate_offset + 4096
    gate = tl.load(gate_up + gate_offset)
    sigmoid = tl.load(sigmoid_gate + sigmoid_offset)
    up = tl.load(gate_up + up_offset)
    silu = gate * sigmoid
    result = up * silu
    tl.store(gate_up + up_offset, result)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    output_shape = hidden_states.shape
    hidden_2d = hidden_states.view(-1, 1024)
    n_tokens = hidden_2d.shape[0]
    gate_up = torch.empty((n_tokens, 8192), device=hidden_states.device, dtype=hidden_states.dtype)
    torch.mm(hidden_2d, gate_up_weight.t(), out=gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    sigmoid_gate = torch.sigmoid(gate)
    n_elements = hidden_states.numel() * 4
    _mul_gate_inplace[(n_elements // 4096, 4)](gate_up, sigmoid_gate, BLOCK=1024)
    output = torch.empty((n_tokens, 1024), device=hidden_states.device, dtype=hidden_states.dtype)
    torch.mm(up, down_weight.t(), out=output)
    return output.view(output_shape)
