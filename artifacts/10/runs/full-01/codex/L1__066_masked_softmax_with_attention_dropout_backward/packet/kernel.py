import torch
import triton
import triton.language as tl


@triton.jit
def _masked_softmax_dropout_backward(
    grad_output,
    p_attn,
    mask,
    dropout_mask,
    grad_scores,
    n_heads: tl.constexpr,
    seq_len: tl.constexpr,
    keep_prob,
    BLOCK: tl.constexpr,
    WITH_DROPOUT: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < seq_len
    offset = row * seq_len + col

    grad = tl.load(
        grad_output + offset, mask=valid, other=0.0, cache_modifier=".cg"
    )
    prob = tl.load(
        p_attn + offset, mask=valid, other=0.0, cache_modifier=".cg"
    )
    if WITH_DROPOUT:
        kept = tl.load(
            dropout_mask + offset, mask=valid, other=0, cache_modifier=".cg"
        )
        grad = tl.where(kept, grad, 0.0) / keep_prob

    dot = tl.sum(prob * grad, axis=0)
    value = prob * (grad - dot)

    batch = row // (n_heads * seq_len)
    query = row % seq_len
    mask_offset = (batch * seq_len + query) * seq_len + col
    unmasked = tl.load(mask + mask_offset, mask=valid, other=0, cache_modifier=".ca")
    value = tl.where(unmasked, value, 0.0)
    tl.store(grad_scores + offset, value, mask=valid)


@triton.jit
def _masked_softmax_dropout_backward_heads(
    grad_output,
    p_attn,
    mask,
    dropout_mask,
    grad_scores,
    n_heads: tl.constexpr,
    seq_len: tl.constexpr,
    keep_prob,
    BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    WITH_DROPOUT: tl.constexpr,
):
    head_group = tl.program_id(0)
    query = tl.program_id(1)
    batch = tl.program_id(2)

    heads = head_group * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    valid = (heads < n_heads) & (cols < seq_len)
    offsets = ((batch * n_heads + heads) * seq_len + query) * seq_len + cols

    grad = tl.load(
        grad_output + offsets, mask=valid, other=0.0, cache_modifier=".cg"
    )
    prob = tl.load(p_attn + offsets, mask=valid, other=0.0, cache_modifier=".cg")
    if WITH_DROPOUT:
        kept = tl.load(
            dropout_mask + offsets, mask=valid, other=0, cache_modifier=".cg"
        )
        grad = tl.where(kept, grad, 0.0) / keep_prob

    dot = tl.sum(prob * grad, axis=1)
    value = prob * (grad - dot[:, None])

    mask_offsets = (batch * seq_len + query) * seq_len + cols
    unmasked = tl.load(
        mask + mask_offsets,
        mask=cols < seq_len,
        other=0,
        cache_modifier=".ca",
    )
    value = tl.where(unmasked, value, 0.0)
    tl.store(grad_scores + offsets, value, mask=valid)


def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    batch, n_heads, seq_len, _ = grad_output.shape
    output = torch.empty_like(grad_output)
    block = triton.next_power_of_2(seq_len)
    if block <= 256:
        num_warps = 1
    elif block <= 1024:
        num_warps = 2 if p_dropout > 0.0 else 4
    elif block <= 2048:
        num_warps = 8
    else:
        num_warps = 16
    with_dropout = p_dropout > 0.0
    keep_prob = 1.0 - p_dropout if with_dropout else 1.0
    rows = batch * n_heads * seq_len

    head_block = 0
    head_warps = 0
    if block == 256 and seq_len > 128 and rows < 32768:
        head_block = 4 if seq_len < 192 else 2
        head_warps = 1 if seq_len < 192 else 2
    elif block == 512 and seq_len < 384:
        head_block = 2
        head_warps = 2
    elif block == 512 and not with_dropout:
        head_block = 4
        head_warps = 4

    if head_block:
        _masked_softmax_dropout_backward_heads[
            (n_heads // head_block, seq_len, batch)
        ](
            grad_output,
            p_attn,
            mask,
            dropout_mask,
            output,
            n_heads=n_heads,
            seq_len=seq_len,
            keep_prob=keep_prob,
            BLOCK=block,
            HEAD_BLOCK=head_block,
            WITH_DROPOUT=with_dropout,
            num_warps=head_warps,
        )
    else:
        _masked_softmax_dropout_backward[(rows,)](
            grad_output,
            p_attn,
            mask,
            dropout_mask,
            output,
            n_heads=n_heads,
            seq_len=seq_len,
            keep_prob=keep_prob,
            BLOCK=block,
            WITH_DROPOUT=with_dropout,
            num_warps=num_warps,
        )
    return output
