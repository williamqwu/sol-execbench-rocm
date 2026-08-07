import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_rows(
    hidden, encoder, output,
    text_len: tl.constexpr, image_len: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column_block = tl.program_id(1)
    columns = column_block * BLOCK + tl.arange(0, BLOCK)
    sequence_len: tl.constexpr = text_len + image_len
    batch = row // sequence_len
    sequence_row = row - batch * sequence_len
    is_text = sequence_row < text_len
    source_row = tl.where(
        is_text,
        batch * text_len + sequence_row,
        batch * image_len + sequence_row - text_len,
    )
    source = tl.where(is_text, encoder, hidden)
    values = tl.load(source + source_row * 3072 + columns)
    tl.store(output + row * 3072 + columns, values)


@torch.no_grad()
def run(hidden_states, encoder_hidden_states, process_weight):
    batch = hidden_states.shape[0]
    text_len = encoder_hidden_states.shape[1]
    image_len = hidden_states.shape[1]
    total_rows = batch * (text_len + image_len)

    # rocBLAS uses the same reduction kernel for the two independent products
    # at these sizes.  Avoiding the concatenation saves a complete read/write
    # pass over the activations.  Irregular single-batch and mid-sized odd-row
    # cases use another reduction/tiling, so retain the concatenated product
    # there for reference-identical rounding (and, in the latter case, speed).
    split_is_exact = batch != 1 or (text_len % 256 == 0 and image_len % 256 == 0)
    split_is_fast = not (2048 <= total_rows < 3072)
    if not (split_is_exact and split_is_fast):
        # A 77-row product selects rocBLAS's small-M reduction, while the
        # concatenated 1K/4K-row products use its regular reduction.  Padding
        # just past the 96-row dispatch boundary reproduces the latter exactly
        # and is much cheaper than copying the full image sequence.
        if batch == 1 and text_len == 77 and image_len % 1024 == 0:
            padded_text = encoder_hidden_states.new_empty((1, 97, 3072))
            padded_text[:, :text_len, :].copy_(encoder_hidden_states)
            weight_t = process_weight.t()
            processed_encoder = torch.matmul(padded_text, weight_t)
            processed_hidden = torch.matmul(hidden_states, weight_t)
            return processed_encoder[:, :text_len, :], processed_hidden

        concatenated = hidden_states.new_empty(
            (batch, text_len + image_len, 3072)
        )
        _concatenate_rows[(total_rows, 3)](
            hidden_states,
            encoder_hidden_states,
            concatenated,
            text_len,
            image_len,
            BLOCK=1024,
            num_warps=4,
        )
        processed = torch.matmul(concatenated, process_weight.t())
        return processed[:, :text_len, :], processed[:, text_len:, :]

    weight_t = process_weight.t()
    processed_encoder = torch.matmul(encoder_hidden_states, weight_t)
    processed_hidden = torch.matmul(hidden_states, weight_t)
    return processed_encoder, processed_hidden
