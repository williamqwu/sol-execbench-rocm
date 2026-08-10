import torch


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    batch, kv_heads, slen, head_dim = key.shape
    heads = query.shape[1]
    key_states = key[:, :, None].expand(
        batch, kv_heads, heads // kv_heads, slen, head_dim
    ).reshape(batch, heads, slen, head_dim)
    q = query.reshape(batch * heads, slen, head_dim)
    k = key_states.reshape(batch * heads, slen, head_dim)
    seed = query[0, 0, 0, 0]
    scores = torch.baddbmm(seed, q, k.transpose(1, 2), beta=0.0, alpha=scaling)
    return scores.reshape(batch, heads, slen, slen)
