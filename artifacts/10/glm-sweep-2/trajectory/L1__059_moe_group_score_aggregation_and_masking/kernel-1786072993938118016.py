import torch
import triton
import triton.language as tl

@triton.jit
def _moe_group_mask_kernel(
    scores_ptr, masked_scores_ptr, group_mask_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
    N_GROUP: tl.constexpr,
    EPG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)  # 256
    scores_off = pid * BLOCK + offs
    scores = tl.load(scores_ptr + scores_off, mask=offs < BLOCK, other=float("-inf"))

    # Reshape conceptually to (8, 32). group_ids: which group each of 256 elems belongs to
    group_ids = offs // EPG  # 0..7

    # top-2 per group. We iterate per group since only 8 groups.
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32) - float("inf")
    # For each group, compute top-2 sum.
    for g in range(N_GROUP):
        mask_g = group_ids == g
        vals = tl.where(mask_g, scores, float("-inf"))
        # max1
        max1 = tl.max(vals, axis=0)
        # mask out max1 to find max2
        vals2 = tl.where(vals == max1, float("-inf"), vals)
        max2 = tl.max(vals2, axis=0)
        group_scores = tl.where(tl.arange(0, N_GROUP) == g, max1 + max2, group_scores)

    # top-4 of 8 group scores
    # Find 4th largest threshold: iterate 4 times
    gs = group_scores
    selected = tl.zeros([N_GROUP], dtype=tl.int1)
    for _ in range(4):
        m = tl.max(gs, axis=0)
        selected = selected | (gs == m)
        gs = tl.where(gs == m, float("-inf"), gs)

    # group_mask output: 8 values
    gm_off = pid * N_GROUP + tl.arange(0, N_GROUP)
    tl.store(group_mask_ptr + gm_off, selected.to(tl.float32))

    # Expand mask to 256 and apply
    expert_mask = (group_ids == 0)  # placeholder
    # Build per-element mask: selected[group_ids]
    # Use lookup via where chain
    emask = tl.zeros([BLOCK], dtype=tl.int1)
    for g in range(N_GROUP):
        gsel = (tl.arange(0, N_GROUP) == g) & selected
        gsel_scalar = tl.sum(gsel.to(tl.int32), axis=0)
        emask = emask | ((group_ids == g) & (gsel_scalar == 1))

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
        num_tokens,
        BLOCK=num_experts,
        N_GROUP=n_group,
        EPG=epg,
    )
    return masked_scores, group_mask
