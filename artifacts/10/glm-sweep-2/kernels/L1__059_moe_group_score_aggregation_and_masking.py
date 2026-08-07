import torch
import triton
import triton.language as tl


@triton.jit
def _moe_group_mask_kernel(
    scores_ptr, masked_scores_ptr, group_mask_ptr,
    BLOCK: tl.constexpr,
    N_GROUP: tl.constexpr,
    EPG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)  # 256
    scores_off = pid * BLOCK + offs
    scores = tl.load(scores_ptr + scores_off, mask=offs < BLOCK, other=float("-inf"))

    group_ids = offs // EPG  # 0..7 for each of 256 elements
    g_arange = tl.arange(0, N_GROUP)  # 0..7

    # top-2 per group -> group_scores (8 values)
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32) - float("inf")
    for g in range(N_GROUP):
        mask_g = group_ids == g
        vals = tl.where(mask_g, scores, float("-inf"))
        max1 = tl.max(vals, axis=0)
        vals2 = tl.where(vals == max1, float("-inf"), vals)
        max2 = tl.max(vals2, axis=0)
        group_scores = tl.where(g_arange == g, max1 + max2, group_scores)

    # top-4 of 8 group scores, lower-index tie-break (matches torch.topk)
    # Repeatedly remove the single largest element (lowest index on ties), 4 times.
    gs_work = group_scores
    for _ in range(4):
        cur_max = tl.max(gs_work, axis=0)
        is_eq = gs_work == cur_max
        eq_idx = tl.where(is_eq, g_arange, N_GROUP)
        min_eq_idx = tl.min(eq_idx, axis=0)
        is_first = is_eq & (g_arange == min_eq_idx)
        gs_work = tl.where(is_first, float("-inf"), gs_work)

    # After removing 4 maxima, gs_work has the 4 smallest. The 4th-largest = max of remaining.
    thr = tl.max(gs_work, axis=0)
    strict_above = group_scores > thr
    n_strict = tl.sum(strict_above.to(tl.int32), axis=0)
    need_from_tied = 4 - n_strict
    is_tied = group_scores == thr
    # rank among tied = count of tied elements with lower index
    tied_rank = tl.zeros([N_GROUP], dtype=tl.int32)
    for g in range(N_GROUP):
        contrib = is_tied & (g_arange < g)
        tied_rank = tl.where(g_arange == g, tl.sum(contrib.to(tl.int32), axis=0), tied_rank)
    selected = strict_above | (is_tied & (tied_rank < need_from_tied))

    # group_mask output
    gm_off = pid * N_GROUP + g_arange
    tl.store(group_mask_ptr + gm_off, selected.to(tl.float32))

    # Expand to 256 and apply mask: emask[i] = selected[group_ids[i]]
    emask = tl.zeros([BLOCK], dtype=tl.int1)
    for g in range(N_GROUP):
        gsel = tl.sum(tl.where(g_arange == g, selected.to(tl.int32), 0), axis=0)  # 0 or 1
        emask = emask | ((group_ids == g) & (gsel == 1))

    out = tl.where(emask, scores, float("-inf"))
    tl.store(masked_scores_ptr + scores_off, out, mask=offs < BLOCK)


def run(scores: torch.Tensor):
    num_tokens = scores.size(0)
    num_experts = 256
    n_group = 8
    epg = num_experts // n_group  # 32

    masked_scores = torch.empty_like(scores)
    group_mask = torch.empty((num_tokens, n_group), device=scores.device, dtype=scores.dtype)

    grid = (num_tokens,)
    _moe_group_mask_kernel[grid](
        scores, masked_scores, group_mask,
        BLOCK=num_experts,
        N_GROUP=n_group,
        EPG=epg,
    )
    return masked_scores, group_mask
