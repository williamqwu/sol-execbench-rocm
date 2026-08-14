import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rope_q_kernel(X, POS, INV, Y, total: tl.constexpr, L: tl.constexpr,
                   BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    d = off % 128
    token = (off // 4096) % L
    batch = off // (L * 4096)
    partner = tl.where(d < 64, off + 64, off - 64)
    x = tl.load(X + off, mask=mask)
    other = tl.load(X + partner, mask=mask)
    other = tl.where(d < 64, -other, other)
    pos = tl.load(POS + batch * L + token, mask=mask).to(tl.float32)
    angle = pos * tl.load(INV + (d % 64), mask=mask)
    c = tl.cos(angle).to(tl.bfloat16)
    s = tl.sin(angle).to(tl.bfloat16)
    a = (x * c).to(tl.bfloat16)
    r = (other * s).to(tl.bfloat16)
    tl.store(Y + off, (a + r).to(tl.bfloat16), mask=mask)


@triton.jit
def _rope_k_kernel(X, POS, INV, Y, total: tl.constexpr, N: tl.constexpr,
                   BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    d = off % 128
    token = (off // 128) % N
    head = (off // (128 * N)) % 8
    batch = off // (8 * N * 128)

    axis = tl.where(d < 42, 0, tl.where(d < 84, 1, 2))
    local = tl.where(d < 42, d, tl.where(d < 84, d - 42, d - 84))
    half = tl.where(d < 84, 21, 22)
    partner_d = tl.where(local < half, d + half, d - half)
    input_base = (batch * N + token) * 1024 + head * 128
    x = tl.load(X + input_base + d, mask=mask)
    other = tl.load(X + input_base + partner_d, mask=mask)
    other = tl.where(local < half, -other, other)
    pos = tl.load(POS + (batch * N + token) * 3 + axis, mask=mask).to(tl.float32)
    angle = pos * tl.load(INV + d, mask=mask)
    c = tl.cos(angle).to(tl.bfloat16)
    s = tl.sin(angle).to(tl.bfloat16)
    a = (x * c).to(tl.bfloat16)
    r = (other * s).to(tl.bfloat16)
    tl.store(Y + off, (a + r).to(tl.bfloat16), mask=mask)


@triton.jit
def _rope_qk_kernel(Q, K, QPOS, KPOS, QINV, KINV, QOUT, KOUT,
                    q_total: tl.constexpr, k_total: tl.constexpr,
                    L: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    qmask = off < q_total
    qd = off % 128
    qtoken = (off // 4096) % L
    qbatch = off // (L * 4096)
    qpartner = tl.where(qd < 64, off + 64, off - 64)
    qx = tl.load(Q + off, mask=qmask)
    qother = tl.load(Q + qpartner, mask=qmask)
    qother = tl.where(qd < 64, -qother, qother)
    qpos = tl.load(QPOS + qbatch * L + qtoken, mask=qmask).to(tl.float32)
    qangle = qpos * tl.load(QINV + (qd % 64), mask=qmask)
    qc = tl.cos(qangle).to(tl.bfloat16)
    qs = tl.sin(qangle).to(tl.bfloat16)
    qa = (qx * qc).to(tl.bfloat16)
    qr = (qother * qs).to(tl.bfloat16)
    tl.store(QOUT + off, (qa + qr).to(tl.bfloat16), mask=qmask)

    kmask = off < k_total
    kd = off % 128
    ktoken = (off // 128) % N
    khead = (off // (128 * N)) % 8
    kbatch = off // (8 * N * 128)
    axis = tl.where(kd < 42, 0, tl.where(kd < 84, 1, 2))
    local = tl.where(kd < 42, kd, tl.where(kd < 84, kd - 42, kd - 84))
    half = tl.where(kd < 84, 21, 22)
    partner_d = tl.where(local < half, kd + half, kd - half)
    input_base = (kbatch * N + ktoken) * 1024 + khead * 128
    kx = tl.load(K + input_base + kd, mask=kmask)
    kother = tl.load(K + input_base + partner_d, mask=kmask)
    kother = tl.where(local < half, -kother, kother)
    kpos = tl.load(KPOS + (kbatch * N + ktoken) * 3 + axis, mask=kmask).to(tl.float32)
    kangle = kpos * tl.load(KINV + kd, mask=kmask)
    kc = tl.cos(kangle).to(tl.bfloat16)
    ks = tl.sin(kangle).to(tl.bfloat16)
    ka = (kx * kc).to(tl.bfloat16)
    kr = (kother * ks).to(tl.bfloat16)
    tl.store(KOUT + off, (ka + kr).to(tl.bfloat16), mask=kmask)


_rope_freq_cache = {}


def _rope_frequencies(device):
    index = device.index
    cached = _rope_freq_cache.get(index)
    if cached is None:
        fq_half = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128))
        fq = torch.cat((fq_half, fq_half))
        pieces = []
        for dim in (42, 42, 44):
            half = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            pieces.append(torch.cat((half, half)))
        cached = (fq, torch.cat(pieces))
        _rope_freq_cache[index] = cached
    return cached


def _apply_rope(q_linear, k_linear, language_position_ids, vision_grid_thw,
                block=256, warps=4):
    b, l, _ = q_linear.shape
    n = k_linear.shape[1]
    fq, fk = _rope_frequencies(q_linear.device)
    q_out = torch.empty_like(q_linear)
    k_out = torch.empty((b, 8, n, 128), device=k_linear.device, dtype=k_linear.dtype)
    q_total = b * l * 4096
    k_total = b * n * 1024
    _rope_q_kernel[(triton.cdiv(q_total, block),)](
        q_linear, language_position_ids, fq, q_out,
        total=q_total, L=l, BLOCK=block, num_warps=warps,
    )
    _rope_k_kernel[(triton.cdiv(k_total, block),)](
        k_linear, vision_grid_thw, fk, k_out,
        total=k_total, N=n, BLOCK=block, num_warps=warps,
    )
    return q_out.view(b, l, 32, 128).transpose(1, 2), k_out


def _apply_rope_combined(q_linear, k_linear, language_position_ids, vision_grid_thw,
                         block=256, warps=4):
    b, l, _ = q_linear.shape
    n = k_linear.shape[1]
    fq, fk = _rope_frequencies(q_linear.device)
    q_out = torch.empty_like(q_linear)
    k_out = torch.empty((b, 8, n, 128), device=k_linear.device, dtype=k_linear.dtype)
    q_total = b * l * 4096
    k_total = b * n * 1024
    _rope_qk_kernel[(triton.cdiv(q_total, block),)](
        q_linear, k_linear, language_position_ids, vision_grid_thw,
        fq, fk, q_out, k_out,
        q_total=q_total, k_total=k_total, L=l, N=n, BLOCK=block,
        num_warps=warps,
    )
    return q_out.view(b, l, 32, 128).transpose(1, 2), k_out


@triton.jit
def _attention_bf16_kernel(
    Q, K, V, O,
    sqb: tl.constexpr, sqh: tl.constexpr, sqm: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, skn: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svn: tl.constexpr,
    L: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // 32
    head = pid_bh % 32
    kv_head = head // 4

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 128)
    q = tl.load(
        Q + batch * sqb + head * sqh + rows[:, None] * sqm + dims[None, :],
        mask=rows[:, None] < L,
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denom = tl.zeros((BLOCK_M,), tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        k = tl.load(
            K + batch * skb + kv_head * skh + cols[:, None] * skn + dims[None, :],
            mask=cols[:, None] < N,
            other=0.0,
        )
        score = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        # torch.matmul and the following scalar division each produce BF16.
        score = (score / 11.313708498984761).to(tl.bfloat16).to(tl.float32)
        score = tl.where(cols[None, :] < N, score, -float("inf"))
        block_max = tl.max(score, axis=1)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * tl.exp(row_max - new_max) + tl.sum(
            tl.exp(score - new_max[:, None]), axis=1
        )
        row_max = new_max

    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        k = tl.load(
            K + batch * skb + kv_head * skh + cols[:, None] * skn + dims[None, :],
            mask=cols[:, None] < N,
            other=0.0,
        )
        score = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        score = (score / 11.313708498984761).to(tl.bfloat16).to(tl.float32)
        score = tl.where(cols[None, :] < N, score, -float("inf"))
        prob = (tl.exp(score - row_max[:, None]) / denom[:, None]).to(tl.bfloat16)
        value = tl.load(
            V + batch * svb + kv_head * svh + cols[:, None] * svn + dims[None, :],
            mask=cols[:, None] < N,
            other=0.0,
        )
        acc = tl.dot(prob, value, acc)

    out = O + (batch * L + rows[:, None]) * 4096 + head * 128 + dims[None, :]
    tl.store(out, acc.to(tl.bfloat16), mask=rows[:, None] < L)


def _attention_bf16(q, k, value, b, l, v, block_m=16, block_n=64, num_warps=4):
    out = torch.empty((b, l, 4096), device=q.device, dtype=q.dtype)
    _attention_bf16_kernel[(triton.cdiv(l, block_m), b * 32)](
        q, k, value, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        L=l, N=v, BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


@torch.no_grad()
def run(
    language_hidden_states,
    vision_hidden_states,
    language_position_ids,
    vision_grid_thw,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    k_proj_bias,
    v_proj_weight,
    v_proj_bias,
    o_proj_weight,
):
    b, l, _ = language_hidden_states.shape
    v = vision_hidden_states.shape[1]

    q_linear = F.linear(language_hidden_states, q_proj_weight, q_proj_bias)
    k_linear = F.linear(vision_hidden_states, k_proj_weight, k_proj_bias)
    value = F.linear(vision_hidden_states, v_proj_weight, v_proj_bias)
    value = value.view(b, v, 8, 128).transpose(1, 2)
    if b * l <= 256:
        q, k = _apply_rope_combined(
            q_linear, k_linear, language_position_ids, vision_grid_thw, 128, 2
        )
    else:
        q, k = _apply_rope(
            q_linear, k_linear, language_position_ids, vision_grid_thw, 512, 4
        )

    if (b == 1 and l <= 1024) or b * l < 1024:
        out = _attention_bf16(q, k, value, b, l, v, 64, 16, 4)
    else:
        out = _attention_bf16(q, k, value, b, l, v, 128, 32, 4)
    return F.linear(out, o_proj_weight)
