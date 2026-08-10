import torch


def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_t = process_weight.t()

    # Separate GEMMs avoid materializing the concatenated input for batched
    # streams.  For batch one, keeping a single GEMM also keeps rocBLAS on the
    # same numerically stable reduction path as the reference.
    batch = hidden_states.shape[0]
    text_len = encoder_hidden_states.shape[1]
    image_len = hidden_states.shape[1]
    aligned_streams = (
        text_len % 128 == 0 and image_len % 128 == 0
    )
    aligned_batch_one = (
        batch == 1
        and text_len >= 256
        and text_len % 256 == 0
        and image_len % 256 == 0
    )
    split_batched = (
        batch > 1
        and (
            batch * (text_len + image_len) >= 3000
            or aligned_streams
            or image_len >= 4 * text_len
        )
    )
    if split_batched or aligned_batch_one:
        encoder_out = torch.matmul(encoder_hidden_states, weight_t)
        hidden_out = torch.matmul(hidden_states, weight_t)
        return encoder_out, hidden_out

    encoder_rows = batch * text_len
    combined = torch.cat(
        (
            encoder_hidden_states.view(encoder_rows, -1),
            hidden_states.view(batch * image_len, -1),
        ),
        dim=0,
    )
    projected = torch.mm(combined, weight_t)
    encoder_out = projected[:encoder_rows].view_as(encoder_hidden_states)
    hidden_out = projected[encoder_rows:].view_as(hidden_states)
    return encoder_out, hidden_out
