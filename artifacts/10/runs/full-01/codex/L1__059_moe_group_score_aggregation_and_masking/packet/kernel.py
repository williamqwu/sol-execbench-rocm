import torch
import triton
import triton.language as tl


@triton.jit
def _merge_top2(a0, a1, b0, b1):
    hi = tl.maximum(a0, b0)
    lo = tl.maximum(tl.minimum(a0, b0), tl.maximum(a1, b1))
    return hi, lo


@triton.jit
def _moe_group_mask_kernel(scores, masked_scores, group_mask):
    token = tl.program_id(0)

    expert = tl.arange(0, 256)
    values = tl.load(scores + token * 256 + expert)
    grouped = tl.reshape(values, (8, 32))

    # Carry the two largest values through one exact tree reduction.
    first, second = tl.reduce(
        (grouped, tl.full((8, 32), -float("inf"), tl.float32)),
        axis=1,
        combine_fn=_merge_top2,
    )
    quality = first + second

    group = tl.arange(0, 8)
    idx0 = tl.argmax(quality, axis=0)
    quality = tl.where(group == idx0, -float("inf"), quality)
    idx1 = tl.argmax(quality, axis=0)
    quality = tl.where(group == idx1, -float("inf"), quality)
    idx2 = tl.argmax(quality, axis=0)
    quality = tl.where(group == idx2, -float("inf"), quality)
    idx3 = tl.argmax(quality, axis=0)
    selected_groups = (
        (group == idx0) | (group == idx1) | (group == idx2) | (group == idx3)
    )
    tl.store(group_mask + token * 8 + group, selected_groups.to(tl.float32))

    selected_experts = tl.reshape(
        tl.broadcast_to(selected_groups[:, None], (8, 32)), (256,)
    )
    result = tl.where(selected_experts, values, -float("inf"))
    tl.store(masked_scores + token * 256 + expert, result)


def run(scores: torch.Tensor):
    n_tokens = scores.shape[0]
    masked_scores = torch.empty_like(scores)
    group_mask = torch.empty((n_tokens, 8), device=scores.device, dtype=torch.float32)
    _moe_group_mask_kernel[(n_tokens,)](
        scores,
        masked_scores,
        group_mask,
        num_warps=1,
    )
    return masked_scores, group_mask
