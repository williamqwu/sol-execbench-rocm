import torch


@torch.compile(dynamic=True)
def _compute(pos, freq_bands_list, total_dim):
    # axes_dim = [16, 56, 56]; half = [8, 28, 28]; offsets = [0, 16, 72]
    offsets = [0, 16, 72]
    dims = [16, 56, 56]
    cos_out = pos.new_empty((pos.shape[0], total_dim))
    sin_out = pos.new_empty((pos.shape[0], total_dim))
    for axis_idx in range(3):
        dim = dims[axis_idx]
        off = offsets[axis_idx]
        axis_pos = pos[:, axis_idx]
        fb = freq_bands_list[axis_idx]
        angles = axis_pos.unsqueeze(-1) * fb.unsqueeze(0)  # [seq_len, half]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        # repeat_interleave via strided writes into final buffer: [c0,c0,c1,c1,...]
        cos_out[:, off:off + dim:2] = cos_angles
        cos_out[:, off + 1:off + dim:2] = cos_angles
        sin_out[:, off:off + dim:2] = sin_angles
        sin_out[:, off + 1:off + dim:2] = sin_angles
    return cos_out, sin_out


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    device = ids.device
    pos = ids.float()
    # Precompute frequency bands (depend only on theta and fixed dims)
    half_dims = [8, 28, 28]
    freq_bands_list = [
        (1.0 / (theta ** (torch.arange(h, dtype=torch.float32, device=device) / h)))
        for h in half_dims
    ]
    return _compute(pos, freq_bands_list, 128)
