import torch


@torch.compile
def _scaled_expand(key: torch.Tensor, heads: int, scaling: float) -> torch.Tensor:
    return key.expand(key.shape[0], heads, key.shape[2], key.shape[3]) * scaling


@torch.compile
def _attention_bmm(query: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
    batch, heads, slen, head_dim = query.shape
    q = query.reshape(batch * heads, slen, head_dim)
    k = key_states.reshape(batch * heads, slen, head_dim)
    return torch.bmm(q, k.transpose(1, 2)).reshape(batch, heads, slen, slen)


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, kv_heads, slen, head_dim = key.shape
    heads = query.shape[1]
    # Materialize GQA replication and the exact power-of-two scale together in
    # one pointwise kernel, avoiding a separate scale tensor and copy launch.
    key_states = _scaled_expand(key, heads, scaling)
    return _attention_bmm(query, key_states)
