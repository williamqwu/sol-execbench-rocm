import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A:  ggo = (gwo @ w2) * sw           (fp32, never rounded)
#            grad_up_output   = ggo * gate_output
#            grad_gate_output = ggo * up_output
#            grad_w1_output   = grad_gate_output * dsilu(w1_output)
# One fused pass: a (M x F) GEMM with K = hidden_dim, plus the whole
# element-wise chain, emitting the two bf16 operands the next stage needs.
# ---------------------------------------------------------------------------
@triton.jit
def _ka(GWO, W2, SW, UP, GATE, W1O, GU, G1,
        M, F, K,
        sg0, sg1, sw0, sw1, sa0, sa1,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(F, BN)
    ng = GM * nn
    gid = pid // ng
    first_m = gid * GM
    gsize = tl.minimum(nm - first_m, GM)
    pid_m = first_m + ((pid % ng) % gsize)
    pid_n = (pid % ng) // gsize

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    mask_n = offs_n < F
    rm = tl.where(mask_m, offs_m, 0)
    rn = tl.where(mask_n, offs_n, 0)

    a_ptrs = GWO + rm[:, None] * sg0 + offs_k[None, :] * sg1
    b_ptrs = W2 + offs_k[:, None] * sw0 + rn[None, :] * sw1

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sg1
        b_ptrs += BK * sw0

    sw = tl.load(SW + rm, mask=mask_m, other=0.0).to(tl.float32)
    ggo = acc * sw[:, None]

    o = rm[:, None] * sa0 + rn[None, :] * sa1
    msk = mask_m[:, None] & mask_n[None, :]
    up = tl.load(UP + o, mask=msk, other=0.0).to(tl.float32)
    gate = tl.load(GATE + o, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(W1O + o, mask=msk, other=0.0).to(tl.float32)

    grad_up = ggo * gate
    grad_gate = ggo * up
    sig = 1.0 / (1.0 + tl.exp(-x))
    dsilu = sig * (1.0 + x * (1.0 - sig))
    g1 = grad_gate * dsilu

    tl.store(GU + o, grad_up.to(tl.bfloat16), mask=msk)
    tl.store(G1 + o, g1.to(tl.bfloat16), mask=msk)


# ---------------------------------------------------------------------------
# Kernel B:  gst = g1 @ w1_weight + gu @ w3_weight   in ONE fp32 accumulator
# (matches the reference, which adds the two fp32 products before rounding)
# ---------------------------------------------------------------------------
@triton.jit
def _kb(G1, GU, W1, W3, OUT,
        M, N, F,
        sa0, sa1, sw0, sw1, so0, so1,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        SPLIT_K: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    ng = GM * nn
    gid = pid // ng
    first_m = gid * GM
    gsize = tl.minimum(nm - first_m, GM)
    pid_m = first_m + ((pid % ng) % gsize)
    pid_n = (pid % ng) // gsize

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    mask_n = offs_n < N
    rm = tl.where(mask_m, offs_m, 0)
    rn = tl.where(mask_n, offs_n, 0)

    klen = F // SPLIT_K
    k0 = pid_k * klen

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    a_ptrs = G1 + rm[:, None] * sa0 + (k0 + offs_k)[None, :] * sa1
    b_ptrs = W1 + (k0 + offs_k)[:, None] * sw0 + rn[None, :] * sw1
    for k in range(0, klen, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sa1
        b_ptrs += BK * sw0

    a_ptrs = GU + rm[:, None] * sa0 + (k0 + offs_k)[None, :] * sa1
    b_ptrs = W3 + (k0 + offs_k)[:, None] * sw0 + rn[None, :] * sw1
    for k in range(0, klen, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sa1
        b_ptrs += BK * sw0

    o = rm[:, None] * so0 + rn[None, :] * so1
    msk = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(OUT + o, acc, mask=msk)
    else:
        tl.atomic_add(OUT + o, acc, mask=msk)


# ---------------------------------------------------------------------------
# Small fused pre-pass: gwo = grad_output[idx]; grad_sw = sum(gwo*eo);
#                       geo = (gwo * sw) -> bf16
# ---------------------------------------------------------------------------
@triton.jit
def _kpre(GO, IDX, EO, SW, GWO, GEO, GSW,
          M, H,
          sg0, sg1, se0, se1,
          BH: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.load(IDX + pid)
    sw = tl.load(SW + pid).to(tl.float32)
    acc = 0.0
    for h0 in range(0, H, BH):
        offs = h0 + tl.arange(0, BH)
        g = tl.load(GO + idx * sg0 + offs * sg1).to(tl.float32)
        e = tl.load(EO + pid * se0 + offs * se1).to(tl.float32)
        acc += tl.sum(g * e, axis=0)
        tl.store(GWO + pid * H + offs, g.to(tl.bfloat16))
        tl.store(GEO + pid * H + offs, (g * sw).to(tl.bfloat16))
    tl.store(GSW + pid, acc)


def _cfg_a(M):
    if M <= 128:
        return dict(BM=64, BN=64, BK=128, GM=8, num_warps=4, num_stages=2)
    if M <= 512:
        return dict(BM=64, BN=128, BK=64, GM=8, num_warps=4, num_stages=2)
    return dict(BM=128, BN=128, BK=64, GM=8, num_warps=8, num_stages=2)


def _cfg_b(M):
    if M <= 128:
        return dict(BM=64, BN=64, BK=128, GM=8, SPLIT_K=8, num_warps=4, num_stages=2)
    if M <= 256:
        return dict(BM=64, BN=64, BK=128, GM=8, SPLIT_K=4, num_warps=4, num_stages=2)
    if M <= 512:
        return dict(BM=64, BN=64, BK=128, GM=8, SPLIT_K=2, num_warps=4, num_stages=2)
    return dict(BM=128, BN=128, BK=64, GM=8, SPLIT_K=1, num_warps=8, num_stages=2)


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
    M = token_indices.shape[0]
    F = w1_weight.shape[0]
    dev = grad_output.device
    bf = torch.bfloat16

    # ---- pre-pass -----------------------------------------------------
    gwo = torch.empty((M, H), device=dev, dtype=bf)
    geo = torch.empty((M, H), device=dev, dtype=bf)
    gsw = torch.empty((M,), device=dev, dtype=torch.float32)
    _kpre[(M,)](
        grad_output, token_indices, expert_output, selected_weights,
        gwo, geo, gsw,
        M, H,
        grad_output.stride(0), grad_output.stride(1),
        expert_output.stride(0), expert_output.stride(1),
        BH=1024, num_warps=8,
    )

    grad_routing_weights = torch.zeros((B,), device=dev, dtype=bf)
    grad_routing_weights.index_copy_(0, token_indices, gsw.to(bf))

    # ---- kernel A -----------------------------------------------------
    g1 = torch.empty((M, F), device=dev, dtype=bf)
    gu = torch.empty((M, F), device=dev, dtype=bf)
    ca = _cfg_a(M)
    grid_a = (triton.cdiv(M, ca["BM"]) * triton.cdiv(F, ca["BN"]),)
    _ka[grid_a](
        gwo, w2_weight, selected_weights, up_output, gate_output, w1_output, gu, g1,
        M, F, H,
        gwo.stride(0), gwo.stride(1),
        w2_weight.stride(0), w2_weight.stride(1),
        g1.stride(0), g1.stride(1),
        BM=ca["BM"], BN=ca["BN"], BK=ca["BK"], GM=ca["GM"],
        num_warps=ca["num_warps"], num_stages=ca["num_stages"],
    )

    # ---- weight gradients (hipBLASLt) ---------------------------------
    grad_w2_weight = torch.mm(geo.t(), gated_output)
    grad_w3_weight = torch.mm(gu.t(), selected_tokens)
    grad_w1_weight = torch.mm(g1.t(), selected_tokens)

    # ---- kernel B: grad_selected_tokens -------------------------------
    cb = _cfg_b(M)
    if cb["SPLIT_K"] == 1:
        gst = torch.empty((M, H), device=dev, dtype=torch.float32)
    else:
        gst = torch.zeros((M, H), device=dev, dtype=torch.float32)
    grid_b = (triton.cdiv(M, cb["BM"]) * triton.cdiv(H, cb["BN"]), cb["SPLIT_K"])
    _kb[grid_b](
        g1, gu, w1_weight, w3_weight, gst,
        M, H, F,
        g1.stride(0), g1.stride(1),
        w1_weight.stride(0), w1_weight.stride(1),
        gst.stride(0), gst.stride(1),
        BM=cb["BM"], BN=cb["BN"], BK=cb["BK"], SPLIT_K=cb["SPLIT_K"], GM=cb["GM"],
        num_warps=cb["num_warps"], num_stages=cb["num_stages"],
    )

    grad_hidden_states = torch.zeros((B, H), device=dev, dtype=bf)
    grad_hidden_states.index_copy_(0, token_indices, gst.to(bf))

    return (
        grad_hidden_states,
        grad_routing_weights,
        grad_w1_weight,
        grad_w2_weight,
        grad_w3_weight,
    )
