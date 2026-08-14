import torch
import triton
import triton.language as tl

# out[i, j] = grad_expert_mask[i, j] + (grad_tokens_per_expert[j] + c)
#   c = grad_loss * n_experts * (1/n_experts) / (batch_seq_len * topk)   if training else 0
#
# Memory bound: reads batch_seq_len*256 fp32, writes the same. The 256-element
# broadcast vector and the two scalars stay resident in cache, so the SOL is
# 2 * N * 256 * 4 bytes of HBM traffic. Everything (including the two scalar
# reads that the reference does with .item()) happens on device, so there is no
# host synchronisation in the timed region.

N_EXPERTS: int = 256  # const axis from definition.json


@triton.jit
def _bias_add_kernel(
    gem_ptr,  # [n_elem] fp32, contiguous (batch_seq_len, 256)
    tpe_ptr,  # [256] fp32
    gloss_ptr,  # [1] fp32
    train_ptr,  # [1] bool
    out_ptr,  # [n_elem] fp32
    n_elem,
    denom,  # batch_seq_len * num_experts_per_tok
    BLOCK: tl.constexpr,
    EMASK: tl.constexpr,  # N_EXPERTS - 1
    CG: tl.constexpr,  # load cache modifier
    CS: tl.constexpr,  # store cache modifier
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem

    # Scalar term, evaluated in fp64 to mirror the reference's Python-side
    # arithmetic, then rounded to fp32 exactly as torch does for a
    # float32-tensor + python-float add. Both scalars are read on device, so
    # there is no host synchronisation (a .item() pair costs ~27us here, more
    # than the entire kernel).
    is_train = tl.load(train_ptr) != 0
    gloss = tl.load(gloss_ptr).to(tl.float64)
    # gloss * 256 * (1/256) is exact (powers of two), so this is gloss / denom.
    c = tl.where(is_train, gloss / denom, 0.0).to(tl.float32)

    # Broadcast vector index: rows are 256 contiguous floats and 256 is a power
    # of two, so the expert index is just the low bits of the flat offset. The
    # 1 KiB vector stays resident in cache across the whole grid.
    j = offs & EMASK
    tpe = tl.load(tpe_ptr + j)

    gem = tl.load(
        gem_ptr + offs, mask=mask, other=0.0,
        cache_modifier=CG, eviction_policy="evict_first",
    )
    tl.store(
        out_ptr + offs, gem + (tpe + c), mask=mask,
        cache_modifier=CS, eviction_policy="evict_first",
    )


# Measured on MI355X (gfx950). Below ~1e5 elements every configuration lands on
# the ~12.6us Triton dispatch floor (an empty kernel launch costs the same), so
# the block/warp choice only matters for the streaming sizes.
def _pick_config(n_elem: int):
    # A finer per-size ladder was tried and measured *worse* on the harness at
    # 8.4M and 33.6M elements: the offline differences it was fitted to were
    # within run-to-run noise. This coarse split is what actually reproduced.
    M = 1024 * 1024
    if n_elem >= 100 * M:  # ~1 GB of traffic
        return 512, 2, ".cg", ".wt"  # 6.64 TB/s
    if n_elem >= 8 * M:
        return 2048, 4, ".cg", ".wt"  # 7.1-7.4 TB/s
    return 1024, 2, "", ""  # dispatch-bound below here (~8us floor)


def run(
    grad_tokens_per_expert: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_load_balance_loss: torch.Tensor,
    topk_idx: torch.Tensor,
    expert_mask: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    training: torch.Tensor,
):
    batch_seq_len = topk_idx.shape[0]
    num_experts_per_tok = 8  # const axis

    gem = grad_expert_mask
    if not gem.is_contiguous():
        gem = gem.contiguous()
    tpe = grad_tokens_per_expert
    if not tpe.is_contiguous():
        tpe = tpe.contiguous()

    out = torch.empty_like(gem)
    n_elem = gem.numel()
    if n_elem == 0:
        return out

    block, warps, cg, cs = _pick_config(n_elem)
    grid = (triton.cdiv(n_elem, block),)
    _bias_add_kernel[grid](
        gem,
        tpe,
        grad_load_balance_loss,
        training,
        out,
        n_elem,
        float(batch_seq_len * num_experts_per_tok),
        BLOCK=block,
        EMASK=N_EXPERTS - 1,
        CG=cg,
        CS=cs,
        num_warps=warps,
        num_stages=1,
    )
    return out
