import torch

@torch.compile(dynamic=True)
def _compute(pos, theta, device):
    axes_dim = [16, 56, 56]
    cos_list = []
    sin_list = []
    for axis_idx in range(3):
        dim = axes_dim[axis_idx]
        axis_pos = pos[:, axis_idx]
        half_dim = dim // 2
        freq_exponents = torch.arange(half_dim, dtype=torch.float32, device=device)
        freq_bands = 1.0 / (theta ** (freq_exponents / half_dim))
        angles = axis_pos.unsqueeze(-1) * freq_bands.unsqueeze(0)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        cos_interleaved = torch.repeat_interleave(cos_angles, 2, dim=-1)
        sin_interleaved = torch.repeat_interleave(sin_angles, 2, dim=-1)
        cos_list.append(cos_interleaved)
        sin_list.append(sin_interleaved)
    freqs_cos = torch.cat(cos_list, dim=-1)
    freqs_sin = torch.cat(sin_list, dim=-1)
    return freqs_cos, freqs_sin


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    pos = ids.float()
    return _compute(pos, theta, ids.device)
