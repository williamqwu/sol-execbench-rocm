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
    causal = torch.tril(torch.ones(s, s, dtype=torch.bool,
                                   device=hidden_states.device))
    decay = torch.exp2(seg.masked_fill(~causal, -torch.inf) * 1.4426950408889634)

    # One Gram matrix per group, shared by four heads.
    cf = C.permute(0, 1, 3, 2, 4)
    bf = B.permute(0, 1, 3, 4, 2)
    gram = torch.bmm(cf.reshape(-1, s, cf.shape[-1]),
                     bf.reshape(-1, bf.shape[-2], s)).reshape(b, nc, g, s, s)
    weights = (gram[:, :, :, None] *
               decay.reshape(b, nc, g, q, s, s)).reshape(b, nc, h, s, s)

    x = hidden_states.permute(0, 1, 3, 2, 4)
    y = torch.bmm(weights.to(torch.bfloat16).reshape(-1, s, s),
                  x.reshape(-1, s, d)).reshape(b, nc, h, s, d)
    return y.permute(0, 1, 3, 2, 4).to(torch.bfloat16)


@torch.no_grad()
def run(hidden_states: torch.Tensor, A_cumsum: torch.Tensor,
        B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return _run(hidden_states, A_cumsum, B, C)
