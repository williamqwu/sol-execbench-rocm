import torch
import torch.nn.functional as F

_HAS_GMM = hasattr(torch, "_grouped_mm")


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
):
    T, H = hidden_states.shape
    E, I, _ = gate_weights.shape
    k = topk_indices.shape[1]
    dev = hidden_states.device
    N = T * k

    flat = topk_indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    offs = torch.cumsum(counts, 0).to(torch.int32)

    tok_sorted = order.div(k, rounding_mode="floor")
    w_sorted = topk_weights.reshape(-1).index_select(0, order)

    xin = hidden_states.index_select(0, tok_sorted)               # [N,H]
    gwo = grad_output.index_select(0, tok_sorted)                 # [N,H]
    geo = gwo * w_sorted.unsqueeze(-1)                            # [N,H]

    gmm = torch._grouped_mm

    # ---- gate / up projections
    gpre = gmm(xin, gate_weights.transpose(1, 2), offs=offs)      # [N,I]
    uout = gmm(xin, up_weights.transpose(1, 2), offs=offs)        # [N,I]

    gout = F.silu(gpre)
    inter = gout * uout

    # ---- down projection: forward (for grad_topk_weights) and backward
    eout = gmm(inter, down_weights.transpose(1, 2), offs=offs)    # [N,H]
    gew_flat = (gwo * eout).sum(dim=-1)
    del eout, gwo

    ginter = gmm(geo, down_weights, offs=offs)                    # [N,I]
    grad_down = gmm(geo.t(), inter, offs=offs)                    # [E,H,I]
    del inter, geo

    ggo = ginter * uout
    guo = ginter * gout
    del ginter, uout, gout

    sig = torch.sigmoid(gpre)
    silu_grad = sig * (1.0 + gpre * (1.0 - sig))
    del sig, gpre
    ggpre = ggo * silu_grad
    del ggo, silu_grad

    # ---- input grads + weight grads
    gx = gmm(ggpre, gate_weights, offs=offs)                      # [N,H]
    grad_gate = gmm(ggpre.t(), xin, offs=offs)                    # [E,I,H]
    del ggpre

    gx += gmm(guo, up_weights, offs=offs)
    grad_up = gmm(guo.t(), xin, offs=offs)                        # [E,I,H]
    del guo, xin

    grad_topk_weights = torch.empty(N, device=dev, dtype=topk_weights.dtype)
    grad_topk_weights[order] = gew_flat
    grad_topk_weights = grad_topk_weights.view(T, k)

    pos_of_flat = torch.empty(N, dtype=torch.long, device=dev)
    pos_of_flat[order] = torch.arange(N, device=dev)
    asc = topk_indices.argsort(dim=1)
    flatpos = torch.arange(T, device=dev).unsqueeze(1) * k + asc
    rows = pos_of_flat[flatpos.reshape(-1)].view(T, k)

    grad_hidden_states = gx.index_select(0, rows[:, 0].contiguous())
    for r in range(1, k):
        grad_hidden_states = grad_hidden_states + gx.index_select(
            0, rows[:, r].contiguous()
        )

    return grad_hidden_states, grad_topk_weights, grad_gate, grad_up, grad_down
