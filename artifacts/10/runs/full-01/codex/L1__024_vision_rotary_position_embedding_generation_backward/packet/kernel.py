import torch
import triton
import triton.language as tl


@triton.jit
def _backward_kernel(
    grad_cos_ptr,
    grad_sin_ptr,
    pos_ids_ptr,
    emb_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    mask = rows < n_elements

    p0 = tl.load(pos_ids_ptr + rows * 2, mask=mask, other=0).to(tl.float32)
    p1 = tl.load(pos_ids_ptr + rows * 2 + 1, mask=mask, other=0).to(tl.float32)

    # The two copies made by cat() contribute columns j and j+36 for
    # height, and columns j+18 and j+54 for width.
    base = rows * 72 + j

    x0 = tl.load(emb_ptr + base, mask=mask, other=0.0)
    c0 = tl.load(grad_cos_ptr + base, mask=mask, other=0.0)
    s0 = tl.load(grad_sin_ptr + base, mask=mask, other=0.0)
    v0 = -c0 * tl.sin(x0) + s0 * tl.cos(x0)

    x1 = tl.load(emb_ptr + base + 36, mask=mask, other=0.0)
    c1 = tl.load(grad_cos_ptr + base + 36, mask=mask, other=0.0)
    s1 = tl.load(grad_sin_ptr + base + 36, mask=mask, other=0.0)
    v1 = -c1 * tl.sin(x1) + s1 * tl.cos(x1)

    x2 = tl.load(emb_ptr + base + 18, mask=mask, other=0.0)
    c2 = tl.load(grad_cos_ptr + base + 18, mask=mask, other=0.0)
    s2 = tl.load(grad_sin_ptr + base + 18, mask=mask, other=0.0)
    v2 = -c2 * tl.sin(x2) + s2 * tl.cos(x2)

    x3 = tl.load(emb_ptr + base + 54, mask=mask, other=0.0)
    c3 = tl.load(grad_cos_ptr + base + 54, mask=mask, other=0.0)
    s3 = tl.load(grad_sin_ptr + base + 54, mask=mask, other=0.0)
    v3 = -c3 * tl.sin(x3) + s3 * tl.cos(x3)

    value = p0 * (v0 + v1) + p1 * (v2 + v3)
    tl.store(out_ptr + j, tl.sum(value, axis=0))


@triton.jit
def _partial_kernel(
    grad_cos_ptr,
    grad_sin_ptr,
    pos_ids_ptr,
    emb_ptr,
    partial_ptr,
    n_elements: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    cols = tl.arange(0, 32)[None, :]
    mask = (rows < n_elements) & (cols < 18)

    p0 = tl.load(pos_ids_ptr + rows * 2, mask=rows < n_elements, other=0).to(tl.float32)
    p1 = tl.load(pos_ids_ptr + rows * 2 + 1, mask=rows < n_elements, other=0).to(tl.float32)
    base = rows * 72 + cols

    x0 = tl.load(emb_ptr + base, mask=mask, other=0.0)
    c0 = tl.load(grad_cos_ptr + base, mask=mask, other=0.0)
    s0 = tl.load(grad_sin_ptr + base, mask=mask, other=0.0)
    v0 = -c0 * tl.sin(x0) + s0 * tl.cos(x0)

    x1 = tl.load(emb_ptr + base + 36, mask=mask, other=0.0)
    c1 = tl.load(grad_cos_ptr + base + 36, mask=mask, other=0.0)
    s1 = tl.load(grad_sin_ptr + base + 36, mask=mask, other=0.0)
    v1 = -c1 * tl.sin(x1) + s1 * tl.cos(x1)

    x2 = tl.load(emb_ptr + base + 18, mask=mask, other=0.0)
    c2 = tl.load(grad_cos_ptr + base + 18, mask=mask, other=0.0)
    s2 = tl.load(grad_sin_ptr + base + 18, mask=mask, other=0.0)
    v2 = -c2 * tl.sin(x2) + s2 * tl.cos(x2)

    x3 = tl.load(emb_ptr + base + 54, mask=mask, other=0.0)
    c3 = tl.load(grad_cos_ptr + base + 54, mask=mask, other=0.0)
    s3 = tl.load(grad_sin_ptr + base + 54, mask=mask, other=0.0)
    v3 = -c3 * tl.sin(x3) + s3 * tl.cos(x3)

    value = p0 * (v0 + v1) + p1 * (v2 + v3)
    sums = tl.sum(value, axis=0)
    tl.store(partial_ptr + block * 32 + cols, sums, mask=cols < 18)


@triton.jit
def _finish_kernel(partial_ptr, out_ptr, n_blocks: tl.constexpr, BLOCK_B: tl.constexpr):
    blocks = tl.arange(0, BLOCK_B)[:, None]
    cols = tl.arange(0, 32)[None, :]
    values = tl.load(
        partial_ptr + blocks * 32 + cols,
        mask=(blocks < n_blocks) & (cols < 18),
        other=0.0,
    )
    sums = tl.sum(values, axis=0)
    tl.store(out_ptr + cols, sums, mask=cols < 18)


def run(grad_cos, grad_sin, pos_ids, inv_freq, emb):
    n_elements = grad_cos.shape[0]
    output = torch.empty((18,), device=grad_cos.device, dtype=torch.float32)
    if n_elements >= 384:
        block_r = 32
        n_blocks = triton.cdiv(n_elements, block_r)
        partial = torch.empty((n_blocks * 32,), device=grad_cos.device, dtype=torch.float32)
        _partial_kernel[(n_blocks,)](
            grad_cos,
            grad_sin,
            pos_ids,
            emb,
            partial,
            n_elements=n_elements,
            BLOCK_R=block_r,
            num_warps=8,
        )
        _finish_kernel[(1,)](
            partial,
            output,
            n_blocks=n_blocks,
            BLOCK_B=triton.next_power_of_2(n_blocks),
            num_warps=8,
        )
        return output

    block_n = triton.next_power_of_2(n_elements)
    _backward_kernel[(18,)](
        grad_cos,
        grad_sin,
        pos_ids,
        emb,
        output,
        n_elements=n_elements,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return output
