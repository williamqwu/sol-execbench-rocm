import torch
import torch.nn.functional as F
import triton
import triton.language as tl

E_NUM = 64
TOP_K = 8


# ---------------------------------------------------------------- silu * up
@triton.jit
def _silu_mul(G, U, O, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
    s = g / (1.0 + tl.exp(-g))
    tl.store(O + off, (s.to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16), mask=m)


def silu_mul(g, u):
    o = torch.empty_like(g)
    n = g.numel()
    _silu_mul[(triton.cdiv(n, 4096),)](g, u, o, n, BLOCK=4096, num_warps=4)
    return o


# ------------------------------------------------------------------ combine
@triton.jit
def _combine(D, POS, W, SH, O, H, BLOCK: tl.constexpr, K: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    m = cols < H
    acc = tl.zeros((BLOCK,), dtype=tl.bfloat16)
    for j in tl.static_range(K):
        p = tl.load(POS + t * K + j).to(tl.int64)
        w = tl.load(W + p).to(tl.float32)
        v = tl.load(D + p * H + cols, mask=m, other=0.0).to(tl.float32)
        acc = acc + (v * w).to(tl.bfloat16)
    s = tl.load(SH + t * H + cols, mask=m, other=0.0)
    tl.store(O + t * H + cols, acc + s, mask=m)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    norm_min: float,
):
    B, S, H = hidden_states.shape
    T = B * S
    dev = hidden_states.device
    x = hidden_states.reshape(T, H)

    # ---- router (small, launch first) ----
    logits = x.float() @ gate_weight.t()
    rw = torch.softmax(logits, dim=1) + e_score_correction_bias
    _, sel = torch.topk(rw, TOP_K, dim=-1)
    sw = torch.gather(rw, -1, sel)
    sw = sw / torch.clamp(sw.sum(-1, keepdim=True), min=norm_min)
    sw = sw.to(x.dtype).reshape(-1)

    flat = sel.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = torch.div(order, TOP_K, rounding_mode="floor")
    offs = torch.bincount(flat, minlength=E_NUM).cumsum(0).to(torch.int32)

    # row position of each (token, slot), sorted ascending per token
    M = T * TOP_K
    rank = torch.empty(M, dtype=torch.int32, device=dev)
    rank[order] = torch.arange(M, dtype=torch.int32, device=dev)
    pos = rank.view(T, TOP_K).sort(dim=1).values.contiguous()
    w_by_row = sw[order].contiguous()

    # ---- shared expert ----
    sg = x @ shared_gate_proj.t()
    su = x @ shared_up_proj.t()
    shared = silu_mul(sg, su) @ shared_down_proj.t()
    del sg, su

    # ---- expert MLPs ----
    xs = torch.index_select(x, 0, tok)
    g = torch._grouped_mm(xs, expert_gate_proj.transpose(1, 2), offs)
    u = torch._grouped_mm(xs, expert_up_proj.transpose(1, 2), offs)
    del xs
    h = silu_mul(g, u)
    del g, u
    d = torch._grouped_mm(h, expert_down_proj.transpose(1, 2), offs)
    del h

    # ---- weighted sequential (ascending-expert) combine + shared ----
    out = torch.empty(T, H, dtype=x.dtype, device=dev)
    BLOCK = 2048
    _combine[(T, triton.cdiv(H, BLOCK))](
        d, pos, w_by_row, shared, out, H, BLOCK=BLOCK, K=TOP_K, num_warps=4
    )
    return out.view(B, S, H), logits
