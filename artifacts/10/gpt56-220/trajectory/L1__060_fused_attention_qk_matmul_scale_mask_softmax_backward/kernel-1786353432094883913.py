import torch


@torch.compile
def _softmax_backward(grad_output, attn_weights, scaling: float):
    go = grad_output.float()
    aw = attn_weights.float()
    dot = torch.sum(go * aw, dim=-1, keepdim=True)
    return (aw * (go - dot)).to(torch.bfloat16) * scaling


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_weights: torch.Tensor,
    scaling: float,
):
    grad = _softmax_backward(grad_output, attn_weights, scaling)
    n = grad.shape[0] * grad.shape[1]
    g = grad.reshape(n, grad.shape[2], grad.shape[3])
    q = query.reshape(n, query.shape[2], query.shape[3])
    k = key.reshape(n, key.shape[2], key.shape[3])
    return (
        torch.bmm(g, k).reshape_as(query),
        torch.bmm(g.transpose(1, 2), q).reshape_as(key),
    )
