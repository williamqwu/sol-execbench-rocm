import torch

@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
def _compiled(attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch_size, _, seq_len_q, _ = attention_weights.shape
    x = torch.bmm(
        attention_weights.flatten(0, 1),
        value.flatten(0, 1),
    )
    return x.unflatten(0, (batch_size, 20)).transpose(1, 2).contiguous().reshape(batch_size, seq_len_q, 1280)

@torch.no_grad()
def run(
    attention_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute attention_weights @ value and return [B, Q, H*D]."""
    return _compiled(attention_weights, value)
