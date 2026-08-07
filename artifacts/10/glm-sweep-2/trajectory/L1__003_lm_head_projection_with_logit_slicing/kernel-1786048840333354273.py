import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LM head projection: hidden_states @ weight^T -> [B, S, V].

    Computes the GEMM as (weight @ hidden_states^T)^T, placing the large
    vocabulary dimension (V=102400) as the M axis of rocBLAS. For this
    tall-skinny problem that selects a faster kernel than the direct
    hs @ w.t() on most shapes. Output is returned as a strided [B,S,V]
    view (no extra copy).
    """
    B, S, H = hidden_states.shape
    V = weight.shape[0]
    hs2d = hidden_states.reshape(B * S, H)
    out = torch.mm(weight, hs2d.t())              # [V, B*S]
    return out.reshape(V, B, S).permute(1, 2, 0)
