import torch
import triton
import triton.language as tl


@triton.jit
def _pack_duplicated_kernel(
    image_ptr,
    context_ptr,
    packed_ptr,
    image_seq_len: tl.constexpr,
    context_seq_len: tl.constexpr,
    inner_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    feature = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.program_id(1)
    batch = tl.program_id(2)
    valid = feature < inner_dim
    total_seq_len = image_seq_len + context_seq_len
    row = batch * total_seq_len + seq
    from_image = seq < image_seq_len

    if from_image:
        image_offset = (batch * image_seq_len + seq) * inner_dim + feature
        value = tl.load(image_ptr + image_offset, mask=valid)
    else:
        context_offset = (
            batch * context_seq_len + (seq - image_seq_len)
        ) * inner_dim + feature
        value = tl.load(context_ptr + context_offset, mask=valid)
    packed_offset = row * (2 * inner_dim) + feature
    tl.store(packed_ptr + packed_offset, value, mask=valid)
    tl.store(packed_ptr + packed_offset + inner_dim, value, mask=valid)


@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    batch_size, image_seq_len, inner_dim = image_attention_output.shape
    context_seq_len = context_attention_output.shape[1]
    total_seq_len = image_seq_len + context_seq_len
    rows = batch_size * total_seq_len
    packed = torch.empty(
        (rows, 2 * inner_dim),
        device=image_attention_output.device,
        dtype=image_attention_output.dtype,
    )

    if batch_size >= 64:
        block = 512
        num_warps = 2
    elif rows <= 800:
        block = 256
        num_warps = 2
    else:
        block = 1024
        num_warps = 4
    _pack_duplicated_kernel[
        (triton.cdiv(inner_dim, block), total_seq_len, batch_size)
    ](
        image_attention_output,
        context_attention_output,
        packed,
        image_seq_len=image_seq_len,
        context_seq_len=context_seq_len,
        inner_dim=inner_dim,
        BLOCK=block,
        num_warps=num_warps,
    )

    projected = torch._C._nn.linear(packed, to_out_weight, to_out_bias)
    projected = projected.view(batch_size, total_seq_len, inner_dim)
    return projected[:, :image_seq_len, :], projected[:, image_seq_len:, :]
