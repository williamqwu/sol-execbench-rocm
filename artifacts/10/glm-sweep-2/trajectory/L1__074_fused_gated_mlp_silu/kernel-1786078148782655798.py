import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(up_ptr, out_ptr, M, I2, I, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * I
    row = offs // I
    col = offs % I
    g = tl.load(up_ptr + row * I2 + col, mask=mask)
    u = tl.load(up_ptr + row * I2 + I + col, mask=mask)
    tl.store(out_ptr + offs, u * g * tl.sigmoid(g), mask=mask)


@torch.no_grad()
def run(hidden_states: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    up = F.linear(hidden_states, gate_up_weight)
    B, S, H = hidden_states.shape
    M = B * S
    I2 = gate_up_weight.shape[0]
    I = I2 // 2
    out = torch.empty(M, I, device=hidden_states.device, dtype=hidden_states.dtype)
    BLOCK = 1024
    _silu_mul_kernel[(triton.cdiv(M * I, BLOCK),)](up.reshape(M, I2), out, M, I2, I, BLOCK=BLOCK)
    return F.linear(out.reshape(B, S, I), down_weight)
