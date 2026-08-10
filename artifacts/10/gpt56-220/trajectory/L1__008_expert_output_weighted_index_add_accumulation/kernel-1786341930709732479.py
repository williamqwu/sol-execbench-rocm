import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add(out, src, indices, n_rows: tl.constexpr, H: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    dst_row = tl.load(indices + row)
    values = tl.load(src + row * H + cols, mask=mask)
    tl.atomic_add(out + dst_row * H + cols, values, mask=mask)


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    output = final_hidden_states.clone()
    n_rows, hidden = expert_outputs.shape
    _scatter_add[(n_rows, triton.cdiv(hidden, 256))](
        output, expert_outputs, token_indices, n_rows=n_rows, H=hidden,
        BLOCK=256, num_warps=4,
    )
    return output
