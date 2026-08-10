import torch


@torch.compile(fullgraph=True, dynamic=True)
def _compiled(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    batch_size, _, seq_len, _ = attn_weights.shape
    x = torch.einsum("bhqk,bhkd->bhqd", attn_weights, value_states)
    return x.transpose(1, 2).reshape(batch_size, seq_len, 5120).contiguous()


@torch.no_grad()
def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    return _compiled(attn_weights, value_states)
