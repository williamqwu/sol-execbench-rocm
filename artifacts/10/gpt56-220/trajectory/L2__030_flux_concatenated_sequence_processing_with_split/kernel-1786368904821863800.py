import torch


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_t = process_weight.t()

    # Separate GEMMs avoid materializing the concatenated input for batched
    # streams.  For batch one, keeping a single GEMM also keeps rocBLAS on the
    # same numerically stable reduction path as the reference.
    if hidden_states.shape[0] > 1:
        return (
            torch.matmul(encoder_hidden_states, weight_t),
            torch.matmul(hidden_states, weight_t),
        )

    text_len = encoder_hidden_states.shape[1]
    combined = torch.cat((encoder_hidden_states, hidden_states), dim=1)
    projected = torch.matmul(combined, weight_t)
    return projected[:, :text_len], projected[:, text_len:]
