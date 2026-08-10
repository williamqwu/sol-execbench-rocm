import torch


@torch.compile(mode="reduce-overhead", fullgraph=True)
def _scaled_expand(key: torch.Tensor, heads: int, scaling: float) -> torch.Tensor:
    return key.expand(key.shape[0], heads, key.shape[2], key.shape[3]) * scaling


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, kv_heads, slen, head_dim = key.shape
    heads = query.shape[1]
    # Materialize GQA replication and the exact power-of-two scale together in
    # one pointwise kernel, avoiding a separate scale tensor and copy launch.
    key_states = _scaled_expand(key, heads, scaling)
    q = query.reshape(batch * heads, slen, head_dim)
    k = key_states.reshape(batch * heads, slen, head_dim)
    return torch.bmm(q, k.transpose(1, 2)).reshape(batch, heads, slen, slen)
