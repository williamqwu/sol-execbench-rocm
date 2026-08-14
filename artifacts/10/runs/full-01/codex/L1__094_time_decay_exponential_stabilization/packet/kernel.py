import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _time_decay_kernel(
    time_decay,
    key,
    time_first,
    value,
    max_state,
    num_state,
    den_state,
    output,
    max_state_out,
    num_state_out,
    den_state_out,
    seq_len,
    hidden_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = channels < hidden_size
    state_offset = batch * hidden_size + channels

    decay = -tl.exp(tl.load(time_decay + channels, mask=mask))
    first = tl.load(time_first + channels, mask=mask)
    max_v = tl.load(max_state + state_offset, mask=mask)
    num_v = tl.load(num_state + state_offset, mask=mask)
    den_v = tl.load(den_state + state_offset, mask=mask)

    base = batch * seq_len * hidden_size + channels
    for t in range(0, seq_len):
        offset = base + t * hidden_size
        key_v = tl.load(key + offset, mask=mask)
        value_v = tl.load(value + offset, mask=mask)

        key_first = key_v + first
        old_is_output_max = max_v >= key_first
        output_delta = tl.where(
            old_is_output_max, key_first - max_v, max_v - key_first
        )
        e_output = tl.exp(output_delta)

        # Exactly one normalized weight is exp(0) == 1.  Form only the
        # nontrivial product, but select its position to retain the eager
        # reference's multiply-then-add ordering.
        weighted_num = e_output * tl.where(old_is_output_max, value_v, num_v)
        numerator = tl.where(old_is_output_max, num_v, weighted_num) + tl.where(
            old_is_output_max, weighted_num, value_v
        )
        weighted_den = e_output * tl.where(old_is_output_max, 1.0, den_v)
        denominator = tl.where(old_is_output_max, den_v, weighted_den) + tl.where(
            old_is_output_max, weighted_den, 1.0
        )
        tl.store(output + offset, numerator / denominator, mask=mask)

        decayed_max = max_v + decay
        next_max = tl.maximum(decayed_max, key_v)
        old_is_state_max = decayed_max >= key_v
        state_delta = tl.where(
            old_is_state_max, key_v - next_max, decayed_max - next_max
        )
        e_state = tl.exp(state_delta)
        weighted_num = e_state * tl.where(old_is_state_max, value_v, num_v)
        num_v = tl.where(old_is_state_max, num_v, weighted_num) + tl.where(
            old_is_state_max, weighted_num, value_v
        )
        weighted_den = e_state * tl.where(old_is_state_max, 1.0, den_v)
        den_v = tl.where(old_is_state_max, den_v, weighted_den) + tl.where(
            old_is_state_max, weighted_den, 1.0
        )
        max_v = next_max

    tl.store(max_state_out + state_offset, max_v, mask=mask)
    tl.store(num_state_out + state_offset, num_v, mask=mask)
    tl.store(den_state_out + state_offset, den_v, mask=mask)


def run(time_decay, key, time_first, value, max_state, num_state, den_state):
    batch_size, seq_len, hidden_size = key.shape
    output = torch.empty_like(key)
    max_out = torch.empty_like(max_state)
    num_out = torch.empty_like(num_state)
    den_out = torch.empty_like(den_state)
    if batch_size <= 2:
        block, num_warps = 16, 1
    elif batch_size <= 8:
        block, num_warps = 32, 1
    elif batch_size <= 16:
        block, num_warps = 128, 2
    elif batch_size <= 32:
        block, num_warps = 256, 4
    else:
        block, num_warps = 512, 8
    grid = (batch_size, triton.cdiv(hidden_size, block))
    _time_decay_kernel[grid](
        time_decay,
        key,
        time_first,
        value,
        max_state,
        num_state,
        den_state,
        output,
        max_out,
        num_out,
        den_out,
        seq_len,
        hidden_size=hidden_size,
        BLOCK=block,
        num_warps=num_warps,
        enable_fp_fusion=True,
    )
    return output, max_out, num_out, den_out
