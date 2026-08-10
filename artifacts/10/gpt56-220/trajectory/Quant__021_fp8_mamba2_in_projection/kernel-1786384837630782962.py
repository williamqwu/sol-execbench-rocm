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

    qx = (x.reshape(m, k // 128, 128) / sx[:, :, None]).clamp(-448, 448)
    qw = (w.reshape(n // 128, 128, k // 128, 128) /
          sw[:, None, :, None]).clamp(-448, 448)
    qx = qx.reshape(m, k).to(torch.float8_e4m3fn)
    qw = qw.reshape(n, k).to(torch.float8_e4m3fn)

    qxb = qx.reshape(m, k // 128, 128)
    qwb = qw.reshape(n // 128, 128, k // 128, 128).permute(2, 0, 1, 3).reshape(k // 128, n, 128)
    pieces = []
    for kb in range(k // 128):
        a = qxb[:, kb, :].contiguous()
        b = qwb[kb].contiguous().T
        sa = sx[:, kb:kb + 1].contiguous()
        sb = sw[:, kb].repeat_interleave(128).reshape(1, n).contiguous()
        pieces.append(torch._scaled_mm(a, b, sa, sb, out_dtype=torch.float32))
    return torch.stack(pieces).sum(dim=0).to(torch.bfloat16)
