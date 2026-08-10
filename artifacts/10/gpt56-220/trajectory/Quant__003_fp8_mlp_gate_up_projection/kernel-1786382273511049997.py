import torch
import torch.nn.functional as F


@torch.no_grad()
@torch.compile(dynamic=True)
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
    # Dequantize the shared activation only once for both projections.
    a = (x.to(torch.float32).reshape(m, k // 128, 128)
         * scale_x.unsqueeze(-1)).reshape(m, k)

    def dequant_weight(w, s):
        return (w.to(torch.float32).reshape(n // 128, 128, k // 128, 128)
                * s[:, None, :, None]).reshape(n, k)

    gate = (a @ dequant_weight(gate_proj_weight, scale_gate).T).to(torch.bfloat16)
    up = (a @ dequant_weight(up_proj_weight, scale_up).T).to(torch.bfloat16)
    return F.silu(gate) * up
