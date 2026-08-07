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

    # Concatenate sequences along the sequence dimension.
    concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # Flatten [B, S, H] -> [B*S, H] so the projection runs as a single dense
    # 2-D GEMM ([B*S, H] @ [H, H]) instead of a batched GEMM. The concatenation
    # is contiguous, so this is a free view, and the 2-D path selects a faster
    # tile configuration on small-batch workloads while keeping the exact same
    # reduction order as the reference.
    flat = concatenated.reshape(-1, hidden_dim)
    processed = torch.matmul(flat, process_weight.t()).reshape(
        batch_size, text_seq_len + img_seq_len, hidden_dim
    )

    processed_encoder = processed[:, :text_seq_len, :]
    processed_hidden = processed[:, text_seq_len:, :]
    return processed_encoder, processed_hidden
