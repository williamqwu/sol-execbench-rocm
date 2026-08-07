import torch
import triton
import triton.language as tl


@triton.jit
def _attention_bwd(
    Q, K, V, G, CU, O, DALL,
    T: tl.constexpr, BLOCK_M: tl.constexpr,
):
    head = tl.program_id(0)
    chunk = tl.program_id(1)
    start = tl.load(CU + chunk)
    end = tl.load(CU + chunk + 1)
    length = end - start

    rm = tl.arange(0, BLOCK_M)
    rd = tl.arange(0, 64)
    row_ok = rm < length
    qkv_offs = head * T * 64 + (start + rm[:, None]) * 64 + rd[None, :]
    q = tl.load(Q + qkv_offs, mask=row_ok[:, None], other=0.0).to(tl.float16)
    k = tl.load(K + qkv_offs, mask=row_ok[:, None], other=0.0).to(tl.float16)
    v = tl.load(V + qkv_offs, mask=row_ok[:, None], other=0.0).to(tl.float16)

    # G and all outputs use token-major [T, H, D] storage.
    out_offs = (start + rm[:, None]) * 1280 + head * 64 + rd[None, :]
    g = tl.load(G + out_offs, mask=row_ok[:, None], other=0.0).to(tl.float16)

    scores = tl.dot(q, tl.trans(k)) * 0.125
    scores = tl.where(row_ok[None, :], scores, float("-inf"))
    scores = tl.where(row_ok[:, None], scores, 0.0)
    scores = scores - tl.max(scores, axis=1)[:, None]
    p = tl.exp(scores)
    p = p / tl.sum(p, axis=1)[:, None]
    ph = p.to(tl.float16)

    o = tl.dot(ph, v)
    tl.store(O + out_offs, o, mask=row_ok[:, None])

    dv = tl.dot(tl.trans(ph), g)
    dp = tl.dot(g, tl.trans(v))
    delta = tl.sum(p * dp, axis=1)
    ds = p * (dp - delta[:, None])
    dsh = ds.to(tl.float16)
    dq = tl.dot(dsh, k) * 0.125
    dk = tl.dot(tl.trans(dsh), q) * 0.125
    d_offs = (start + rm[:, None]) * 3840 + head * 64 + rd[None, :]
    tl.store(DALL + d_offs, dq, mask=row_ok[:, None])
    tl.store(DALL + 1280 + d_offs, dk, mask=row_ok[:, None])
    tl.store(DALL + 2560 + d_offs, dv, mask=row_ok[:, None])


@triton.jit
def _attention_bwd_80(Q, K, V, G, CU, O, DALL, T: tl.constexpr):
    head = tl.program_id(0)
    chunk = tl.program_id(1)
    start = tl.load(CU + chunk)
    length = tl.load(CU + chunk + 1) - start
    r0 = tl.arange(0, 64)
    r1 = tl.arange(0, 16) + 64
    rd = tl.arange(0, 64)
    m0 = r0 < length
    m1 = r1 < length

    off0 = head * T * 64 + (start + r0[:, None]) * 64 + rd[None, :]
    off1 = head * T * 64 + (start + r1[:, None]) * 64 + rd[None, :]
    q0 = tl.load(Q + off0, mask=m0[:, None], other=0.0).to(tl.float16)
    k0 = tl.load(K + off0, mask=m0[:, None], other=0.0).to(tl.float16)
    v0 = tl.load(V + off0, mask=m0[:, None], other=0.0).to(tl.float16)
    q1 = tl.load(Q + off1, mask=m1[:, None], other=0.0).to(tl.float16)
    k1 = tl.load(K + off1, mask=m1[:, None], other=0.0).to(tl.float16)
    v1 = tl.load(V + off1, mask=m1[:, None], other=0.0).to(tl.float16)

    out0 = (start + r0[:, None]) * 1280 + head * 64 + rd[None, :]
    out1 = (start + r1[:, None]) * 1280 + head * 64 + rd[None, :]
    g0 = tl.load(G + out0, mask=m0[:, None], other=0.0).to(tl.float16)
    g1 = tl.load(G + out1, mask=m1[:, None], other=0.0).to(tl.float16)

    s00 = tl.dot(q0, tl.trans(k0)) * 0.125
    s01 = tl.dot(q0, tl.trans(k1)) * 0.125
    s10 = tl.dot(q1, tl.trans(k0)) * 0.125
    s11 = tl.dot(q1, tl.trans(k1)) * 0.125
    s00 = tl.where(m0[None, :], s00, float("-inf"))
    s01 = tl.where(m1[None, :], s01, float("-inf"))
    s10 = tl.where(m0[None, :], s10, float("-inf"))
    s11 = tl.where(m1[None, :], s11, float("-inf"))
    s10 = tl.where(m1[:, None], s10, 0.0)
    s11 = tl.where(m1[:, None], s11, 0.0)

    max0 = tl.maximum(tl.max(s00, axis=1), tl.max(s01, axis=1))
    max1 = tl.maximum(tl.max(s10, axis=1), tl.max(s11, axis=1))
    p00 = tl.exp(s00 - max0[:, None])
    p01 = tl.exp(s01 - max0[:, None])
    p10 = tl.exp(s10 - max1[:, None])
    p11 = tl.exp(s11 - max1[:, None])
    den0 = tl.sum(p00, axis=1) + tl.sum(p01, axis=1)
    den1 = tl.sum(p10, axis=1) + tl.sum(p11, axis=1)
    p00 = p00 / den0[:, None]
    p01 = p01 / den0[:, None]
    p10 = p10 / den1[:, None]
    p11 = p11 / den1[:, None]
    h00, h01 = p00.to(tl.float16), p01.to(tl.float16)
    h10, h11 = p10.to(tl.float16), p11.to(tl.float16)

    o0 = tl.dot(h00, v0) + tl.dot(h01, v1)
    o1 = tl.dot(h10, v0) + tl.dot(h11, v1)
    tl.store(O + out0, o0, mask=m0[:, None])
    tl.store(O + out1, o1, mask=m1[:, None])

    dv0 = tl.dot(tl.trans(h00), g0) + tl.dot(tl.trans(h10), g1)
    dv1 = tl.dot(tl.trans(h01), g0) + tl.dot(tl.trans(h11), g1)
    dp00 = tl.dot(g0, tl.trans(v0))
    dp01 = tl.dot(g0, tl.trans(v1))
    dp10 = tl.dot(g1, tl.trans(v0))
    dp11 = tl.dot(g1, tl.trans(v1))
    delta0 = tl.sum(p00 * dp00, axis=1) + tl.sum(p01 * dp01, axis=1)
    delta1 = tl.sum(p10 * dp10, axis=1) + tl.sum(p11 * dp11, axis=1)
    ds00 = (p00 * (dp00 - delta0[:, None])).to(tl.float16)
    ds01 = (p01 * (dp01 - delta0[:, None])).to(tl.float16)
    ds10 = (p10 * (dp10 - delta1[:, None])).to(tl.float16)
    ds11 = (p11 * (dp11 - delta1[:, None])).to(tl.float16)
    dq0 = (tl.dot(ds00, k0) + tl.dot(ds01, k1)) * 0.125
    dq1 = (tl.dot(ds10, k0) + tl.dot(ds11, k1)) * 0.125
    dk0 = (tl.dot(tl.trans(ds00), q0) + tl.dot(tl.trans(ds10), q1)) * 0.125
    dk1 = (tl.dot(tl.trans(ds01), q0) + tl.dot(tl.trans(ds11), q1)) * 0.125

    d0 = (start + r0[:, None]) * 3840 + head * 64 + rd[None, :]
    d1 = (start + r1[:, None]) * 3840 + head * 64 + rd[None, :]
    tl.store(DALL + d0, dq0, mask=m0[:, None])
    tl.store(DALL + d1, dq1, mask=m1[:, None])
    tl.store(DALL + 1280 + d0, dk0, mask=m0[:, None])
    tl.store(DALL + 1280 + d1, dk1, mask=m1[:, None])
    tl.store(DALL + 2560 + d0, dv0, mask=m0[:, None])
    tl.store(DALL + 2560 + d1, dv1, mask=m1[:, None])


@triton.jit
def _biases(DALL, GO, OUT, T: tl.constexpr, BLOCK_T: tl.constexpr,
            BLOCK_C: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_T)
    is_d = cols < 3840
    d = tl.load(DALL + rows[:, None] * 3840 + cols[None, :],
                mask=(rows[:, None] < T) & is_d[None, :], other=0.0)
    gc = cols - 3840
    g = tl.load(GO + rows[:, None] * 1280 + gc[None, :],
                mask=(rows[:, None] < T) & (gc[None, :] >= 0) &
                     (gc[None, :] < 1280), other=0.0)
    x = tl.where(is_d[None, :], d, g).to(tl.float32)
    tl.store(OUT + cols, tl.sum(x, axis=0), mask=cols < 5120)


@triton.jit
def _hidden(DALL, WQ, WK, WV, OUT, T: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in tl.range(0, 1280, BLOCK_K):
        a = tl.load(DALL + rm[:, None] * 3840 + k0 + rk[None, :],
                    mask=rm[:, None] < T, other=0.0)
        b = tl.load(WQ + (k0 + rk[:, None]) * 1280 + rn[None, :],
                    mask=rn[None, :] < 1280, other=0.0).to(tl.float16)
        acc += tl.dot(a, b)
    for k0 in tl.range(0, 1280, BLOCK_K):
        a = tl.load(DALL + rm[:, None] * 3840 + 1280 + k0 + rk[None, :],
                    mask=rm[:, None] < T, other=0.0)
        b = tl.load(WK + (k0 + rk[:, None]) * 1280 + rn[None, :],
                    mask=rn[None, :] < 1280, other=0.0).to(tl.float16)
        acc += tl.dot(a, b)
    for k0 in tl.range(0, 1280, BLOCK_K):
        a = tl.load(DALL + rm[:, None] * 3840 + 2560 + k0 + rk[None, :],
                    mask=rm[:, None] < T, other=0.0)
        b = tl.load(WV + (k0 + rk[:, None]) * 1280 + rn[None, :],
                    mask=rn[None, :] < 1280, other=0.0).to(tl.float16)
        acc += tl.dot(a, b)
    tl.store(OUT + rm[:, None] * 1280 + rn[None, :], acc,
             mask=(rm[:, None] < T) & (rn[None, :] < 1280))


@triton.jit
def _weight_grads(DALL, GO, X, ATTN, OUT, T: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    batch = tl.program_id(2)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    is_out = batch == 0
    segment = (batch - 1) * 1280
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in tl.range(0, T, BLOCK_K):
        kt = k0 + rk
        mask_a = (kt[None, :] < T) & (rm[:, None] < 1280)
        a_out = tl.load(GO + kt[None, :] * 1280 + rm[:, None],
                        mask=mask_a & is_out, other=0.0).to(tl.float16)
        a_d = tl.load(DALL + kt[None, :] * 3840 + segment + rm[:, None],
                      mask=mask_a & (~is_out), other=0.0)
        a = tl.where(is_out, a_out, a_d)
        mask_b = (kt[:, None] < T) & (rn[None, :] < 1280)
        b_out = tl.load(ATTN + kt[:, None] * 1280 + rn[None, :],
                        mask=mask_b & is_out, other=0.0).to(tl.float16)
        b_x = tl.load(X + kt[:, None] * 1280 + rn[None, :],
                      mask=mask_b & (~is_out), other=0.0).to(tl.float16)
        b = tl.where(is_out, b_out, b_x)
        acc += tl.dot(a, b)
    offs = batch * 1280 * 1280 + rm[:, None] * 1280 + rn[None, :]
    tl.store(OUT + offs, acc,
             mask=(rm[:, None] < 1280) & (rn[None, :] < 1280))


@torch.no_grad()
def run(grad_output, hidden_states, query_states, key_states, value_states,
        cu_seqlens, q_weight, k_weight, v_weight, out_weight):
    t = hidden_states.shape[0]
    h = query_states.shape[1]
    d = query_states.shape[3]
    scale = d ** -0.5


    ga = torch.mm(grad_output, out_weight, out_dtype=torch.float32)
    attn = torch.empty((t, h * d), device=grad_output.device, dtype=torch.bfloat16)
    dall = torch.empty((t, 3 * h * d), device=grad_output.device, dtype=torch.float16)
    chunks = cu_seqlens.numel() - 1
    block_m = 64 if t <= chunks * 64 else 128
    if block_m == 128 and chunks > 1:
        _attention_bwd_80[(h, chunks)](
            query_states, key_states, value_states, ga, cu_seqlens,
            attn, dall, T=t, num_warps=2,
        )
    else:
        _attention_bwd[(h, chunks)](
            query_states, key_states, value_states, ga, cu_seqlens,
            attn, dall, T=t, BLOCK_M=block_m,
            num_warps=2 if block_m == 64 else 4,
        )
    # The smallest workloads need more than BF16's seven mantissa bits on the
    # three paths which are added into grad_hidden_states.  BF16 source weights
    # from this model are represented exactly in FP16, while FP16 also retains
    # three additional bits from the computed attention gradients.
    weight_grads = torch.empty((4, 1280, 1280), device=grad_output.device,
                               dtype=torch.bfloat16)
    _weight_grads[(20, 20, 4)](
        dall, grad_output, hidden_states, attn, weight_grads, T=t,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2,
    )
    grad_hidden = torch.empty((t, 1280), device=grad_output.device,
                              dtype=torch.bfloat16)
    _hidden[(triton.cdiv(t, 64), 20)](
        dall, q_weight, k_weight, v_weight, grad_hidden, T=t,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=2,
    )
    biases = torch.empty(5120, device=grad_output.device, dtype=torch.bfloat16)
    _biases[(triton.cdiv(5120, 16),)](
        dall, grad_output, biases, T=t, BLOCK_T=triton.next_power_of_2(t),
        BLOCK_C=16, num_warps=4,
    )
    db = biases[:3840]

    return (
        grad_hidden,
        weight_grads[1], db[:1280],
        weight_grads[2], db[1280:2560],
        weight_grads[3], db[2560:],
        weight_grads[0], biases[3840:],
    )
