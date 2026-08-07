import torch
import math

@torch.no_grad()
def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    gate_output = torch.matmul(x, gate_proj.t())
    up_output = torch.matmul(x, up_proj.t())
    gate_float = gate_output.to(torch.float32)
    inner = sqrt_2_over_pi * (gate_float + 0.044715 * gate_float.pow(3))
    activated_gate = 0.5 * gate_float * (1.0 + torch.tanh(inner))
    activated_gate = activated_gate.to(gate_output.dtype)
    output = activated_gate * up_output
    return output

run = torch.compile(run, dynamic=True, mode="max-autotune")
