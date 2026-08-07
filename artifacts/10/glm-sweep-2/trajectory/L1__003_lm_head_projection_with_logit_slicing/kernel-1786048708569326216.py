import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LM head projection: hidden_states @ weight^T -> [B, S, V].

    For the large/skinny GEMM (K=2048, V=102400) rocBLAS picks a faster kernel
    when the big V dimension is the M axis, i.e. computing (w @ hs^T)^T.
    We use that transposed formulation for the compute-bound large-M cases and
    the direct hs @ w.t() for small M where it is bandwidth-bound and faster.
    """
    B, S, H = hidden_states.shape
    V = weight.shape[0]
    M = B * S

    if M >= 2048:
        hs2d = hidden_states.reshape(M, H)
        out = torch.mm(weight, hs2d.t())          # [V, M]
        return out.reshape(V, B, S).permute(1, 2, 0)
    return torch.matmul(hidden_states, weight.t())
