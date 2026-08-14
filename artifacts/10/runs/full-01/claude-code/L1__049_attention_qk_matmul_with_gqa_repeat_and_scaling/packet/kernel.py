import torch

# Cache of 0-dim zero tensors per (dtype, device) used as the (unread) bias for
# baddbmm with beta=0. Expanding a 0-dim zero costs no allocation and no memory
# traffic, and it keeps the bias well-defined rather than relying on beta=0
# short-circuiting uninitialised memory.
_ZERO = {}


def _zero(dtype, device):
    key = (dtype, device)
    z = _ZERO.get(key)
    if z is None:
        z = torch.zeros((), dtype=dtype, device=device)
        _ZERO[key] = z
    return z


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    """attn[b,h,i,j] = scaling * sum_d query[b,h,i,d] * key[b,h//rep,j,d]

    The GQA repeat is a pure indexing relation, never a materialised tensor.
    With num_key_value_heads == 1 every query head shares one key matrix, so the
    whole thing collapses to a single batched GEMM over the folded (H*S, D)
    query with the scaling absorbed into the GEMM's alpha -- one kernel launch,
    no separate broadcast pass and no separate multiply pass.
    """
    B, H, S, D = query.shape
    KVH = key.shape[1]

    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()

    out = torch.empty((B, H, S, S), dtype=query.dtype, device=query.device)
    z = _zero(query.dtype, query.device)

    if KVH == 1:
        # (B, H*S, D) @ (B, D, S) -> (B, H*S, S)
        ov = out.view(B, H * S, S)
        torch.baddbmm(
            z.expand(B, H * S, S),
            query.view(B, H * S, D),
            key.view(B, S, D).transpose(1, 2),
            beta=0.0,
            alpha=scaling,
            out=ov,
        )
        return out

    # General GQA: fold the batch and kv-head axes together so the repeat is
    # still expressed as a reshape rather than a materialised broadcast.
    rep = H // KVH
    ov = out.view(B * KVH, rep * S, S)
    torch.baddbmm(
        z.expand(B * KVH, rep * S, S),
        query.view(B * KVH, rep * S, D),
        key.view(B * KVH, S, D).transpose(1, 2),
        beta=0.0,
        alpha=scaling,
        out=ov,
    )
    return out
