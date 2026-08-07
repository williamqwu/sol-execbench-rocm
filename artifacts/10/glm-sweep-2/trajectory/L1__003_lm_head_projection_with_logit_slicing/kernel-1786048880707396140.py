import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LM head projection: hidden_states @ weight^T -> [B, S, V].

    For the tall-skinny GEMM (K=2048, V=102400) rocBLAS usually selects a
    faster kernel when the large V dimension is the M axis, i.e. computing
    (weight @ hidden_states^T)^T. That holds once the problem is large
    enough to be compute-bound; for the very small (memory-bound) case the
    direct hs @ w.t() is faster. Output is returned as a strided [B,S,V]
    view (no extra copy).
    """
    B, S, H = hidden_states.shape
    V = weight.shape[0]
    M = B * S

    if M >= 256:
        hs2d = hidden_states.reshape(M, H)
        out = torch.mm(weight, hs2d.t())          # [V, M]
        return out.reshape(V, B, S).permute(1, 2, 0)
    return torch.matmul(hidden_states, weight.t())
