import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    x: torch.Tensor,
    scale_x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    scale_gate: torch.Tensor,
    up_proj_weight: torch.Tensor,
    scale_up: torch.Tensor,
):
    m, k = x.shape
    n = gate_proj_weight.shape[0]
    # Preserve the exact block mapping while reducing the dequantized operands
    # to the native matrix-engine input type.
    a = (x.to(torch.bfloat16).reshape(m, k // 128, 128)
         * scale_x.to(torch.bfloat16).unsqueeze(-1)).reshape(m, k)

    def dequant_weight(w, s):
        return (w.to(torch.bfloat16).reshape(n // 128, 128, k // 128, 128)
                * s.to(torch.bfloat16)[:, None, :, None]).reshape(n, k)

    gate = a @ dequant_weight(gate_proj_weight, scale_gate).T
    up = a @ dequant_weight(up_proj_weight, scale_up).T
    return F.silu(gate) * up
