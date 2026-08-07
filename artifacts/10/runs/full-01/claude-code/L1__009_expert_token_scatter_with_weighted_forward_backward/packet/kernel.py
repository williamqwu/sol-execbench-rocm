import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: gather grad_output rows by token_indices, and in one pass produce
#   grad_selected_weights[n] = sum_h gwo[n,h] * expert_output[n,h]   (fp32 acc)
#   geo[n,h]                 = gwo[n,h] * selected_weights[n]        (-> bf16)
# The grad_routing_weights scatter is done inline (store at token_indices[n]).
# ---------------------------------------------------------------------------
@triton.jit
def _k_gather_weight(
    GO, IDX, EO, SW, GEO, GRW,
    stride_go,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    idx = tl.load(IDX + n)
    sw = tl.load(SW + n).to(tl.float32)

    row_go = GO + idx.to(tl.int64) * stride_go
    base = n.to(tl.int64) * H

    acc = tl.zeros((), dtype=tl.float32)
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        g = tl.load(row_go + offs).to(tl.float32)
        e = tl.load(EO + base + offs).to(tl.float32)
        acc += tl.sum(g * e, axis=0)
        tl.store(GEO + base + offs, (g * sw).to(tl.bfloat16))

    tl.store(GRW + idx, acc.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Kernel 2: SwiGLU backward epilogue on the (N, F) plane.
#   ggo   = grad_gated_output           (input, bf16)
#   g_up  = ggo * gate_output                      -> buf[:, 0:F]
#   g_w1  = ggo * up_output * silu'(w1_output)     -> buf[:, F:2F]
# Writing both halves of one (N, 2F) buffer lets the two weight-gradient
# GEMMs collapse into a single (2F, N) x (N, H) call.
# ---------------------------------------------------------------------------
@triton.jit
def _k_swiglu_bwd(
    GGO, GATE, UP, W1O, BUF,
    F: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    n = tl.program_id(0)
    fb = tl.program_id(1)
    offs = fb * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = offs < F

    base = n.to(tl.int64) * F + offs
    c = tl.load(GGO + base, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(GATE + base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(UP + base, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(W1O + base, mask=mask, other=0.0).to(tl.float32)

    g_up = c * g
    g_gate = c * u
    s = tl.sigmoid(x)
    g_w1 = g_gate * (s * (1.0 + x * (1.0 - s)))

    obase = n.to(tl.int64) * (2 * F) + offs
    tl.store(BUF + obase, g_up.to(tl.bfloat16), mask=mask)
    tl.store(BUF + obase + F, g_w1.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    token_indices: torch.Tensor,
    selected_tokens: torch.Tensor,
    w1_output: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    gated_output: torch.Tensor,
    expert_output: torch.Tensor,
    selected_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
):
    B, H = grad_output.shape
    N = token_indices.shape[0]
    F = w1_weight.shape[0]
    dev = grad_output.device

    # ---- stage 1: gather + routing-weight grad + grad_expert_output --------
    geo = torch.empty((N, H), dtype=torch.bfloat16, device=dev)
    grw = torch.zeros(B, dtype=torch.bfloat16, device=dev)
    _k_gather_weight[(N,)](
        grad_output, token_indices, expert_output, selected_weights, geo, grw,
        grad_output.stride(0),
        H=H, BLOCK_H=1024, num_warps=4,
    )

    # ---- stage 2: grad_w2  = geo^T @ gated_output  (H, F) -----------------
    gw2 = torch.mm(geo.t(), gated_output)

    # ---- stage 3: grad_gated_output = geo @ w2_weight   (N, F) ------------
    ggo = torch.mm(geo, w2_weight)

    # ---- stage 4: swiglu backward -> buf (N, 2F) --------------------------
    buf = torch.empty((N, 2 * F), dtype=torch.bfloat16, device=dev)
    BLOCK_F = 1024
    _k_swiglu_bwd[(N, triton.cdiv(F, BLOCK_F))](
        ggo, gate_output, up_output, w1_output, buf,
        F=F, BLOCK_F=BLOCK_F, num_warps=4,
    )

    g_up = buf[:, :F]
    g_w1 = buf[:, F:]

    # ---- stage 5: grad_w3 / grad_w1 as one GEMM  (2F, H) ------------------
    gw31 = torch.mm(buf.t(), selected_tokens)
    gw3 = gw31[:F]
    gw1 = gw31[F:]

    # ---- stage 6: grad_selected_tokens = g_up @ w3 + g_w1 @ w1 ------------
    gst = torch.mm(g_up, w3_weight)
    gst = torch.addmm(gst, g_w1, w1_weight)

    # ---- stage 7: scatter -------------------------------------------------
    ghs = torch.zeros((B, H), dtype=torch.bfloat16, device=dev)
    ghs.index_copy_(0, token_indices, gst)

    return ghs, grw, gw1, gw2, gw3
