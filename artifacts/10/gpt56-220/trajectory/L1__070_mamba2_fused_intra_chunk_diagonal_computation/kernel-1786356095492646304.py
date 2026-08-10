import torch


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _run(hidden_states: torch.Tensor, A_cumsum: torch.Tensor,
         B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    b, nc, s, h, d = hidden_states.shape
    g = B.shape[3]
    q = h // g

    # Reference segment sum is sum(A[j+1:i+1]).  Form it from a single
    # prefix sum instead of cumsumming a broadcast 128x128 tensor.
    a = torch.cumsum(A_cumsum.float(), dim=-1).permute(0, 2, 1, 3)
    seg = a[..., :, None] - a[..., None, :]
    pos = torch.arange(s, device=hidden_states.device)
    causal = pos[:, None] >= pos[None, :]
    decay = torch.exp(seg.masked_fill(~causal, -torch.inf))

    # One Gram matrix per group, shared by four heads.
    cf = C.permute(0, 1, 3, 2, 4)
    bf = B.permute(0, 1, 3, 4, 2)
    gram = torch.matmul(cf, bf)
    weights = (gram[:, :, :, None] *
               decay.reshape(b, nc, g, q, s, s)).reshape(b, nc, h, s, s)

    x = hidden_states.permute(0, 1, 3, 2, 4)
    y = torch.matmul(weights.to(torch.bfloat16), x)
    return y.permute(0, 1, 3, 2, 4).to(torch.bfloat16)


@torch.no_grad()
def run(hidden_states: torch.Tensor, A_cumsum: torch.Tensor,
        B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return _run(hidden_states, A_cumsum, B, C)
