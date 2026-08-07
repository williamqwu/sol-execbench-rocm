import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gather_reduce_kernel(
    out_ptr, eo_ptr, perm_ptr, offsets_ptr, n_dest, n_cols,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, MAX_SEG: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_cols
    seg_start = pid * BLOCK_M
    seg_offs = seg_start + tl.arange(0, BLOCK_M)
    seg_mask = seg_offs < n_dest
    starts = tl.load(offsets_ptr + seg_offs, mask=seg_mask, other=0)
    ends = tl.load(offsets_ptr + seg_offs + 1, mask=seg_mask, other=0)
    lengths = ends - starts
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _j in tl.static_range(0, MAX_SEG):
        j = _j
        valid = (j < lengths) & seg_mask
        gidx = starts + j
        perm_val = tl.load(perm_ptr + gidx, mask=valid, other=0)
        src_ptrs = eo_ptr + perm_val[:, None] * n_cols + offs_n[None, :]
        vals = tl.load(src_ptrs, mask=valid[:, None] & mask_n[None, :], other=0.0)
        acc += vals.to(tl.float32)
    out_ptrs = out_ptr + seg_offs[:, None] * n_cols + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=seg_mask[:, None] & mask_n[None, :])


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    nsel = expert_outputs.shape[0]
    bsq = final_hidden_states.shape[0]
    H = final_hidden_states.shape[1]

    if nsel >= 65536:
        _, perm = torch.sort(token_indices)
        counts = torch.bincount(token_indices, minlength=bsq)
        offsets = torch.cat([
            torch.zeros(1, device=final_hidden_states.device, dtype=torch.long),
            counts.cumsum(0),
        ])
        reduced = torch.empty_like(final_hidden_states)
        BLOCK_M, BLOCK_N, MAX_SEG = 8, 1024, 32
        grid = (triton.cdiv(bsq, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _fused_gather_reduce_kernel[grid](
            reduced, expert_outputs, perm, offsets, bsq, H,
            BLOCK_M, BLOCK_N, MAX_SEG,
        )
        return final_hidden_states + reduced
    else:
        output = final_hidden_states.clone()
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        return output
