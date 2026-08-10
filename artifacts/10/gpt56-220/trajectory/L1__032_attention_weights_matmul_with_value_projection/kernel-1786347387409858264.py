import torch


@torch.compile(fullgraph=True, dynamic=True, mode="max-autotune")
def _compiled(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    batch_size, _, seq_len, _ = attn_weights.shape
    x = torch.bmm(
        attn_weights.reshape(batch_size * 40, seq_len, seq_len),
        value_states.reshape(batch_size * 40, seq_len, 128),
    ).reshape(batch_size, 40, seq_len, 128)
    return x.transpose(1, 2).reshape(batch_size, seq_len, 5120).contiguous()


@torch.no_grad()
def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    return _compiled(attn_weights, value_states)
