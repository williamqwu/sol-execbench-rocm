import torch


@torch.no_grad()
@torch.compile(fullgraph=True)
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = hidden_states.shape
    n = weight.shape[0]

    x = hidden_states.float()
    w = weight.float()

    # Activation scales: one independent scale for each 1x128 block.
    sx = (x.reshape(m, k // 128, 128).abs().amax(dim=2) / 448.0).clamp(min=1e-12)
    # Weight scales: one scale for each 128x128 block. Keep the reference's
    # reduction order (K within each row, then the 128 rows).
    sw = (w.reshape(n // 128, 128, k // 128, 128).abs()
            .amax(dim=3).amax(dim=1) / 448.0).clamp(min=1e-12)

    qx = x.reshape(m, k // 128, 128) / sx[:, :, None]
    qw = (w.reshape(n // 128, 128, k // 128, 128) /
          sw[:, None, :, None])
    qx = qx.reshape(m, k).to(torch.float8_e4m3fn)
    qw = qw.reshape(n, k).to(torch.float8_e4m3fn)

    dx = qx.to(torch.bfloat16).reshape(m, k // 128, 128) * sx.to(torch.bfloat16)[:, :, None]
    dw = (qw.to(torch.bfloat16).reshape(n // 128, 128, k // 128, 128) *
          sw.to(torch.bfloat16)[:, None, :, None])
    return dx.reshape(m, k) @ dw.reshape(n, k).T
