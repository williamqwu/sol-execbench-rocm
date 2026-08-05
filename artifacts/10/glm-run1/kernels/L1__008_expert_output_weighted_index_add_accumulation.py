import torch
import triton
import triton.language as tl


@triton.jit
def _count_kernel(ti_ptr, counts_ptr, S, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    idx = tl.load(ti_ptr + offs, mask=mask, other=0).to(tl.int32)
    tl.atomic_add(counts_ptr + idx, 1, mask=mask)


@triton.jit
def _scatter_kernel(ti_ptr, positions_ptr, order_ptr, S, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    idx = tl.load(ti_ptr + offs, mask=mask, other=0).to(tl.int32)
    pos = tl.atomic_add(positions_ptr + idx, 1, mask=mask)
    tl.store(order_ptr + pos, offs.to(tl.int64), mask=mask)


@triton.jit
def _reduce_kernel(
    out_ptr, fhs_ptr, src_ptr, order_ptr, seg_start_ptr,
    H: tl.constexpr, BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    hblk = tl.program_id(1)
    offs_h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    start = tl.load(seg_start_ptr + row)
    end = tl.load(seg_start_ptr + row + 1)
    acc = tl.load(fhs_ptr + row * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    for j in range(start, end):
        sidx = tl.load(order_ptr + j)
        val = tl.load(src_ptr + sidx * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        acc += val
    tl.store(out_ptr + row * H + offs_h, acc.to(tl.bfloat16), mask=mask_h)


def _counting_scatter_reduce(final_hidden_states, expert_outputs, token_indices):
    N, H = final_hidden_states.shape
    S = expert_outputs.shape[0]
    device = final_hidden_states.device

    counts = torch.zeros(N, dtype=torch.int32, device=device)
    order = torch.empty(S, dtype=torch.int64, device=device)

    if S >= 32768:
        BLOCK = 512
        nw = 4
    else:
        BLOCK = 128
        nw = 1

    grid1 = (triton.cdiv(S, BLOCK),)
    _count_kernel[grid1](token_indices, counts, S, BLOCK=BLOCK, num_warps=nw)

    seg_start = torch.empty(N + 1, dtype=torch.int32, device=device)
    seg_start[0] = 0
    torch.cumsum(counts, dim=0, dtype=torch.int32, out=seg_start[1:])

    positions = seg_start[:N].clone()
    grid2 = (triton.cdiv(S, BLOCK),)
    _scatter_kernel[grid2](token_indices, positions, order, S, BLOCK=BLOCK, num_warps=nw)

    out = torch.empty_like(final_hidden_states)
    BLOCK_H = 2048
    grid3 = (N, triton.cdiv(H, BLOCK_H))
    _reduce_kernel[grid3](
        out, final_hidden_states, expert_outputs, order, seg_start,
        H=H, BLOCK_H=BLOCK_H, num_warps=4,
    )
    return out


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    batch_size, seq_len, hidden_size = (
        axes_and_scalars["batch_size"],
        axes_and_scalars["seq_len"],
        axes_and_scalars["hidden_size"],
    )
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    batch_seq_len = batch_size * seq_len
    num_selected_tokens = batch_size * seq_len * num_experts_per_tok

    final_hidden_states = torch.randn(batch_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    expert_outputs = torch.randn(num_selected_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    token_indices = torch.randint(
        0, batch_seq_len, (num_selected_tokens,), dtype=torch.long, device=device
    )

    return {
        "final_hidden_states": final_hidden_states,
        "expert_outputs": expert_outputs,
        "token_indices": token_indices,
    }


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    N = final_hidden_states.shape[0]

    if N >= 768:
        return _counting_scatter_reduce(final_hidden_states, expert_outputs, token_indices)

    output = final_hidden_states.clone()
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    return output
