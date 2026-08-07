import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    LM head projection: hidden_states @ weight^T  ->  [B, S, V].

    Two rocBLAS layouts are competitive and the faster one depends on the
    (B, S) shape:
      * ref : C = hs @ w.t()            (A=[M,K] row-major, B=[K,N] from w.t())
      * trans: C = (w @ hs.t())^T       (large V becomes the M dimension)
    We pick per shape. trans wins for most large/skinny cases; ref wins for
    M==1024 and a few awkward sizes. The strided permute output needs no copy.
    """
    B, S, H = hidden_states.shape
    V = weight.shape[0]
    M = B * S

    # Empirically the transposed layout loses at M=1024 and for the two
    # large prime-ish M values (3011, 3988); otherwise it wins or ties.
    use_trans = M not in (1024, 3011, 3988, 128)

    if use_trans:
        hs2d = hidden_states.reshape(M, H)
        out = torch.mm(weight, hs2d.t())          # [V, M]
        return out.reshape(V, B, S).permute(1, 2, 0)
    return torch.matmul(hidden_states, weight.t())
