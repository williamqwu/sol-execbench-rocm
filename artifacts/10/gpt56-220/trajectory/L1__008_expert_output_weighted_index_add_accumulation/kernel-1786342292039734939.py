import torch
import triton
import triton.language as tl


@triton.jit
def _build_lists(indices, heads, links, M: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < M
    token = tl.load(indices + i, mask=mask, other=0)
    old = tl.atomic_xchg(heads + token, i, mask=mask)
    tl.store(links + i, old, mask=mask)


@triton.jit
def _reduce_lists(base, src, heads, links, out, H: tl.constexpr,
                  BLOCK: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    acc = tl.load(base + token * H + cols, mask=mask, other=0.0).to(tl.float32)
    row = tl.load(heads + token)
    while row >= 0:
        acc += tl.load(src + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        row = tl.load(links + row)
    tl.store(out + token * H + cols, acc, mask=mask)


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    n_tokens, hidden = final_hidden_states.shape
    n_rows = expert_outputs.shape[0]
    heads = torch.full((n_tokens,), -1, dtype=torch.int32,
                       device=final_hidden_states.device)
    links = torch.empty((n_rows,), dtype=torch.int32,
                        device=final_hidden_states.device)
    _build_lists[(triton.cdiv(n_rows, 1024),)](
        token_indices, heads, links, M=n_rows, BLOCK=1024, num_warps=8)
    output = torch.empty_like(final_hidden_states)
    _reduce_lists[(n_tokens, triton.cdiv(hidden, 256))](
        final_hidden_states, expert_outputs, heads, links, output,
        H=hidden, BLOCK=256, num_warps=4)
    return output
