import torch
import torch.nn.functional as F


@torch.no_grad()
@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
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
        wn = w.shape[0]
        return (w.to(torch.float32).reshape(wn // 128, 128, k // 128, 128)
                * s[:, None, :, None]).reshape(wn, k)

    both_weight = torch.cat((gate_proj_weight, up_proj_weight), dim=0)
    both_scale = torch.cat((scale_gate, scale_up), dim=0)
    both = (a @ dequant_weight(both_weight, both_scale).T).to(torch.bfloat16)
    gate, up = both.split(n, dim=1)
    return F.silu(gate) * up
