import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_seq_len = encoder_hidden_states.shape[1]
    img_seq_len = hidden_states.shape[1]
    batch_size = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]

    wt = process_weight.t()
    # Two independent flat GEMMs, avoiding the concatenation copy.
    enc_flat = encoder_hidden_states.reshape(-1, hidden_dim)
    hid_flat = hidden_states.reshape(-1, hidden_dim)
    processed_encoder = torch.mm(enc_flat, wt).reshape(batch_size, text_seq_len, hidden_dim)
    processed_hidden = torch.mm(hid_flat, wt).reshape(batch_size, img_seq_len, hidden_dim)
    return processed_encoder, processed_hidden
