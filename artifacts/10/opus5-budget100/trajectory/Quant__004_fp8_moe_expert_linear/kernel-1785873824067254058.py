import torch
import triton
import triton.language as tl

FP8 = tl.float8e4nv
E4M3_MAX = tl.constexpr(448.0)
RECIP_E4M3_MAX = tl.constexpr(1.0 / 448.0)


# ---------------------------------------------------------------- act quant
@triton.jit
def _quant_act_1x128(X, Q, S, M, K, stride_xm, stride_qm, stride_sm,
                     BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * 128 + tl.arange(0, 128)
    mask = rm[:, None] < M
    x = tl.load(X + rm[:, None] * stride_xm + rk[None, :], mask=mask,
                other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(amax * RECIP_E4M3_MAX, 1e-12)
    q = x / scale[:, None]
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
    tl.store(Q + rm[:, None] * stride_qm + rk[None, :], q.to(FP8), mask=mask)
    tl.store(S + rm * stride_sm + pid_k, scale, mask=rm < M)


def quant_act(x):
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((M, K // 128), dtype=torch.float32, device=x.device)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), K // 128)
    _quant_act_1x128[grid](x, q, s, M, K, x.stride(0), q.stride(0), s.stride(0),
                           BLOCK_M=BLOCK_M, num_warps=4)
    return q, s


# ------------------------------------------------------------- weight quant
@triton.jit
def _quant_w_128x128(W, Q, S, N, K, stride_n, stride_sn):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * 128 + tl.arange(0, 128)
    rk = pid_k * 128 + tl.arange(0, 128)
    p = rn[:, None] * stride_n + rk[None, :]
    w = tl.load(W + p).to(tl.float32)
    amax = tl.max(tl.abs(w))
    scale = tl.maximum(amax * RECIP_E4M3_MAX, 1e-12)
    q = w / scale
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
    tl.store(Q + p, q.to(FP8))
    tl.store(S + pid_n * stride_sn + pid_k, scale)


def quant_weight(w):
    N, K = w.shape
    q = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((N // 128, K // 128), dtype=torch.float32, device=w.device)
    _quant_w_128x128[(N // 128, K // 128)](w, q, s, N, K, w.stride(0),
                                           s.stride(0), num_warps=4)
    return q, s


# ------------------------------------------------------------------- gemm 1
@triton.jit
def _gemm1_silu_quant(A, SA, W, SW, Q, S, M, K, I,
                      stride_am, stride_sam, stride_wn, stride_swn,
                      stride_qm, stride_sm,
                      BLOCK_M: tl.constexpr, NUM_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * 128 + tl.arange(0, 128)
    rk = tl.arange(0, 128)

    a_ptr = A + rm[:, None] * stride_am + rk[None, :]
    g_ptr = W + rn[:, None] * stride_wn + rk[None, :]
    u_ptr = W + (rn + I)[:, None] * stride_wn + rk[None, :]
    mmask = rm[:, None] < M

    acc_g = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(a_ptr, mask=mmask, other=0.0)
        bg = tl.load(g_ptr)
        bu = tl.load(u_ptr)
        sa = tl.load(SA + rm * stride_sam + kb, mask=rm < M, other=0.0)
        sg = tl.load(SW + pid_n * stride_swn + kb)
        su = tl.load(SW + (pid_n + I // 128) * stride_swn + kb)
        dg = tl.dot(a, tl.trans(bg))
        du = tl.dot(a, tl.trans(bu))
        acc_g += dg * (sa[:, None] * sg)
        acc_u += du * (sa[:, None] * su)
        a_ptr += 128
        g_ptr += 128
        u_ptr += 128

    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)

    amax = tl.max(tl.abs(y), axis=1)
    scale = tl.maximum(amax * RECIP_E4M3_MAX, 1e-12)
    q = y / scale[:, None]
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
    tl.store(Q + rm[:, None] * stride_qm + rn[None, :], q.to(FP8), mask=mmask)
    tl.store(S + rm * stride_sm + pid_n, scale, mask=rm < M)


# ------------------------------------------------------------------- gemm 2
@triton.jit
def _gemm2(A, SA, W, SW, R, C, M, K, N,
           stride_am, stride_sam, stride_wn, stride_swn, stride_cm,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    a_ptr = A + rm[:, None] * stride_am + rk[None, :]
    b_ptr = W + rn[:, None] * stride_wn + rk[None, :]
    mmask = rm[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    NB: tl.constexpr = BLOCK_N // 128
    for kb in tl.range(0, NUM_K):
        a = tl.load(a_ptr, mask=mmask, other=0.0)
        b = tl.load(b_ptr)
        sa = tl.load(SA + rm * stride_sam + kb, mask=rm < M, other=0.0)
        sb = tl.load(SW + (pid_n * NB + tl.arange(0, NB)) * stride_swn + kb)
        sb_e = tl.reshape(tl.broadcast_to(sb[:, None], (NB, 128)), (BLOCK_N,))
        acc += tl.dot(a, tl.trans(b)) * (sa[:, None] * sb_e[None, :])
        a_ptr += 128
        b_ptr += 128
    r = tl.load(R + rm, mask=rm < M, other=0.0).to(tl.float32)
    o = acc.to(tl.bfloat16).to(tl.float32) * r[:, None]
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], o.to(tl.bfloat16),
             mask=mmask)


def moe(hidden_states, routing_weight, gate_up_weight, down_weight,
        cfg1=None, cfg2=None):
    M, H = hidden_states.shape
    N2, _ = gate_up_weight.shape
    I = N2 // 2
    dev = hidden_states.device

    aq, asc = quant_act(hidden_states)
    wq, wsc = quant_weight(gate_up_weight)
    dq, dsc = quant_weight(down_weight)

    c1 = cfg1 or dict(BLOCK_M=128, num_warps=8, num_stages=2)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
    BM1 = c1["BLOCK_M"]
    _gemm1_silu_quant[(triton.cdiv(M, BM1), I // 128)](
        aq, asc, wq, wsc, gq, gs, M, H, I,
        aq.stride(0), asc.stride(0), wq.stride(0), wsc.stride(0),
        gq.stride(0), gs.stride(0),
        BLOCK_M=BM1, NUM_K=H // 128,
        num_warps=c1["num_warps"], num_stages=c1["num_stages"])

    c2 = cfg2 or dict(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    _gemm2[(triton.cdiv(M, c2["BLOCK_M"]), H // c2["BLOCK_N"])](
        gq, gs, dq, dsc, routing_weight, out, M, I, H,
        gq.stride(0), gs.stride(0), dq.stride(0), dsc.stride(0), out.stride(0),
        BLOCK_M=c2["BLOCK_M"], BLOCK_N=c2["BLOCK_N"], NUM_K=I // 128,
        num_warps=c2["num_warps"], num_stages=c2["num_stages"])
    return out


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = 3584
    intermediate_size = 2048
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    routing_weight = torch.randn(num_tokens, 1, dtype=torch.bfloat16, device=device)
    gate_up_weight = torch.randn(intermediate_size * 2, hidden_size, dtype=torch.bfloat16, device=device) * (hidden_size ** -0.5)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) * (intermediate_size ** -0.5)
    return {
        "hidden_states": hidden_states,
        "routing_weight": routing_weight,
        "gate_up_weight": gate_up_weight,
        "down_weight": down_weight,
    }


@torch.no_grad()
def run(hidden_states, routing_weight, gate_up_weight, down_weight):
    return moe(hidden_states, routing_weight, gate_up_weight, down_weight)
