import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    LM head projection with logit slicing.

    Compute logits = hidden_states @ weight^T  as  (weight @ hidden_states^T)^T,
    which lets rocBLAS pick a faster kernel for the large-V dimension as M.
    Returns a strided [B, S, V] tensor (values identical, layout differs).
    """
    B, S, H = hidden_states.shape
    V = weight.shape[0]
    # weight: [V, H]; hs2d: [B*S, H]
    hs2d = hidden_states.reshape(B * S, H)
    # out: [V, B*S] = weight @ hs2d^T
    out = torch.mm(weight, hs2d.t())
    # reshape to [V, B, S] then permute to [B, S, V] (strided, no copy)
    return out.reshape(V, B, S).permute(1, 2, 0)
