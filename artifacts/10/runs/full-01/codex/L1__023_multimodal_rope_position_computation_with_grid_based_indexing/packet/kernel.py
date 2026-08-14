import torch
import triton
import triton.language as tl


# CPU and ROCm produce identical float32 linspace bits for this interval.  The
# table is input-independent and built once when the solution module loads, so
# common small grids need no extra device kernels just to materialize it.
_SMALL_COORDS = tuple(
    tuple(torch.linspace(0.0, 34.0, n, dtype=torch.float32).tolist())
    for n in range(2, 35)
)


@triton.jit
def _token_coords(grid_ptr, token, N_IMAGES: tl.constexpr):
    """Map the reference's concatenated/permuted token order to (row, col)."""
    start = tl.zeros(token.shape, tl.int64)
    height = tl.full(token.shape, 2, tl.int64)
    width = tl.full(token.shape, 2, tl.int64)
    running = tl.zeros(token.shape, tl.int64)
    for image_idx in tl.static_range(N_IMAGES):
        t = tl.load(grid_ptr + image_idx * 3 + 0)
        h = tl.load(grid_ptr + image_idx * 3 + 1)
        w = tl.load(grid_ptr + image_idx * 3 + 2)
        end = running + t * h * w
        inside = (token >= running) & (token < end)
        start = tl.where(inside, running, start)
        height = tl.where(inside, h, height)
        width = tl.where(inside, w, width)
        running = end

    local = token.to(tl.int64) - start
    spatial = local % (height * width)
    block = spatial // 4
    intra = spatial % 4
    merged_w = width // 2
    row = (block // merged_w) * 2 + intra // 2
    col = (block % merged_w) * 2 + intra % 2
    return row, col, height, width


@triton.jit
def _patch_kernel(
    grid_ptr, pos_ptr, coord_ptr, inv_ptr, out_ptr, cos_ptr, sin_ptr,
    total_tokens: tl.constexpr,
    N_IMAGES: tl.constexpr, HIDDEN: tl.constexpr,
    MAX_DIM: tl.constexpr, FAST_DIMS: tl.constexpr,
    FAST_INDICES: tl.constexpr, FAST_VALUES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row, col, height, width = _token_coords(grid_ptr, token[None], N_IMAGES)

    # The small lookup holds the exact values produced by torch.linspace on
    # this backend. Its fused range kernel has observably different rounding
    # from an algebraically equivalent expression in Triton.
    y = tl.load(coord_ptr + height * MAX_DIM + row, mask=height > 34, other=0.0)
    x = tl.load(coord_ptr + width * MAX_DIM + col, mask=width > 34, other=0.0)
    for i in tl.static_range(len(FAST_VALUES)):
        y = tl.where(
            (height == FAST_DIMS[i]) & (row == FAST_INDICES[i]),
            FAST_VALUES[i], y,
        )
        x = tl.where(
            (width == FAST_DIMS[i]) & (col == FAST_INDICES[i]),
            FAST_VALUES[i], x,
        )
    y0 = tl.floor(y).to(tl.int32)
    x0 = tl.floor(x).to(tl.int32)
    y1 = tl.minimum(y0 + 1, 34)
    x1 = tl.minimum(x0 + 1, 34)
    dy = y - y0.to(tl.float32)
    dx = x - x0.to(tl.float32)
    omy = 1.0 - dy
    omx = 1.0 - dx
    w0 = omy * omx
    w1 = omy * dx
    w2 = dy * omx
    w3 = dy * dx

    mask = cols < HIDDEN
    i0 = (y0 * 35 + x0) * HIDDEN + cols
    i1 = (y0 * 35 + x1) * HIDDEN + cols
    i2 = (y1 * 35 + x0) * HIDDEN + cols
    i3 = (y1 * 35 + x1) * HIDDEN + cols
    p0 = tl.load(pos_ptr + i0, mask=mask)
    p1 = tl.load(pos_ptr + i1, mask=mask)
    p2 = tl.load(pos_ptr + i2, mask=mask)
    p3 = tl.load(pos_ptr + i3, mask=mask)
    v0 = p0 * w0
    v1 = p1 * w1
    v2 = p2 * w2
    v3 = p3 * w3
    value = ((v0 + v1) + v2) + v3
    tl.store(out_ptr + token * HIDDEN + cols, value, mask=mask)

    # The row already owns this token and has its spatial coordinates, so
    # produce both rotary outputs here as a small sidecar rather than paying
    # for a second launch and another grid scan.
    rope_dim = tl.arange(0, 128)
    base_dim = rope_dim % 64
    rope_coord = tl.where(base_dim < 32, row, col)
    angle = rope_coord.to(tl.float32) * tl.load(inv_ptr + base_dim % 32)
    rope_offset = token * 128 + rope_dim
    tl.store(cos_ptr + rope_offset, tl.cos(angle))
    tl.store(sin_ptr + rope_offset, tl.sin(angle))


@triton.jit
def _rope_kernel(
    grid_ptr, inv_ptr, cos_ptr, sin_ptr,
    total_tokens: tl.constexpr, N_IMAGES: tl.constexpr, BLOCK_M: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 128)
    row, col, _, _ = _token_coords(grid_ptr, tokens, N_IMAGES)
    base_dim = dims % 64
    coord = tl.where(base_dim[None, :] < 32, row[:, None], col[:, None])
    inv_idx = base_dim % 32
    angle = coord.to(tl.float32) * tl.load(inv_ptr + inv_idx)[None, :]
    c = tl.cos(angle)
    s = tl.sin(angle)
    mask = tokens[:, None] < total_tokens
    offsets = tokens[:, None] * 128 + dims[None, :]
    tl.store(cos_ptr + offsets, c, mask=mask)
    tl.store(sin_ptr + offsets, s, mask=mask)


@torch.no_grad()
def run(grid_thw: torch.Tensor, pos_embed_weight: torch.Tensor, inv_freq: torch.Tensor):
    # The output's leading dimension is data-dependent, so obtain the tiny
    # grid descriptor once. All substantial work remains in the two kernels.
    grid = grid_thw.tolist()
    total_tokens = sum(t * h * w for t, h, w in grid)
    n_images = len(grid)
    hidden = pos_embed_weight.shape[1]
    dims = sorted({d for _, h, w in grid for d in (h, w)})
    max_dim = max(dims)
    if max_dim > 34:
        coord = torch.empty(((max_dim + 1) * max_dim,), device=grid_thw.device,
                            dtype=torch.float32)
    else:
        # Masked out by the specialized lookup below; avoid a dummy allocation.
        coord = pos_embed_weight
    for dim in dims:
        if dim > 34:
            torch.linspace(0.0, 34.0, dim, device=grid_thw.device,
                           dtype=torch.float32, out=coord[dim * max_dim:dim * max_dim + dim])

    fast_dims = []
    fast_indices = []
    fast_values = []
    for dim in dims:
        if dim <= 34:
            for index, value in enumerate(_SMALL_COORDS[dim - 2]):
                fast_dims.append(dim)
                fast_indices.append(index)
                fast_values.append(value)

    patch = torch.empty((total_tokens, hidden), device=grid_thw.device, dtype=torch.float32)
    cos = torch.empty((total_tokens, 128), device=grid_thw.device, dtype=torch.float32)
    sin = torch.empty_like(cos)

    _patch_kernel[(total_tokens, triton.cdiv(hidden, 2048))](
        grid_thw, pos_embed_weight, coord, inv_freq, patch, cos, sin,
        total_tokens, N_IMAGES=n_images, HIDDEN=hidden, MAX_DIM=max_dim,
        FAST_DIMS=tuple(fast_dims), FAST_INDICES=tuple(fast_indices),
        FAST_VALUES=tuple(fast_values),
        BLOCK_N=2048, num_warps=2,
    )
    return patch, cos, sin
