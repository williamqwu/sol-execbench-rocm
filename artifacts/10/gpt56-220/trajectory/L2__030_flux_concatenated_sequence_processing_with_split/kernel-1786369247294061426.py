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
    batch_one = batch == 1
    split_batched = (
        batch > 1
        and (
            batch * (text_len + image_len) >= 3000
            or aligned_streams
            or image_len >= 4 * text_len
        )
    )
    if batch_one:
        return (
            torch.mm(encoder_hidden_states[0], weight_t).unsqueeze(0),
            torch.mm(hidden_states[0], weight_t).unsqueeze(0),
        )
    if split_batched:
        encoder_out = torch.matmul(encoder_hidden_states, weight_t)
        hidden_out = torch.matmul(hidden_states, weight_t)
        return encoder_out, hidden_out

    combined = torch.cat((encoder_hidden_states, hidden_states), dim=1)
    projected = torch.matmul(combined, weight_t)
    return projected[:, :text_len], projected[:, text_len:]
