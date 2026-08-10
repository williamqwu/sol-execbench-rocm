import torch

torch._dynamo.config.cache_size_limit = 32


@torch.compile(
    fullgraph=True,
    dynamic=False,
    options={"shape_padding": True, "coordinate_descent_tuning": True},
)
@torch.no_grad()
def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    batch_size, _, seq_len, _ = attn_weights.shape
    x = torch.matmul(attn_weights, value_states)
    return x.transpose(1, 2).reshape(batch_size, seq_len, 5120).contiguous()
