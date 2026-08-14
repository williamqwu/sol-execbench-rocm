import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_MASKS = {}


def _causal_mask(n, device):
    mask = _MASKS.get(n)
    if mask is None or mask.device != device:
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu_(diagonal=1)
        _MASKS[n] = mask
    return mask


@triton.jit
def _flash_fwd_kernel(
    Q, K, V, O, ROW_MAX, ROW_SUM,
    N: tl.constexpr, scale,
    sqb: tl.constexpr, sqh: tl.constexpr, sqm: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, skn: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svn: tl.constexpr,
    sob: tl.constexpr, soh: tl.constexpr, som: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 64
    h = bh % 64
    kh = h // 8
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.arange(0, BLOCK_N)
    od = tl.arange(0, 128)
    q = tl.load(Q + b * sqb + h * sqh + om[:, None] * sqm + od[None, :],
                mask=om[:, None] < N, other=0.0)
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    start_n = 0
    while start_n < (pid_m + 1) * BLOCK_M:
        cols = start_n + on
        k = tl.load(K + b * skb + kh * skh + cols[:, None] * skn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        v = tl.load(V + b * svb + kh * svh + cols[:, None] * svn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32).to(tl.bfloat16)
        s = (s.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        valid = (om[:, None] < N) & (cols[None, :] < N) & (om[:, None] >= cols[None, :])
        s = tl.where(valid, s, -float("inf"))
        new_m = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(s - new_m[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        l = l * alpha + tl.sum(p, axis=1)
        m = new_m
        start_n += BLOCK_N

    tl.store(O + b * sob + h * soh + om[:, None] * som + od[None, :],
             (acc / l[:, None]).to(tl.bfloat16), mask=om[:, None] < N)
    tl.store(ROW_MAX + bh * N + om, m, mask=om < N)
    tl.store(ROW_SUM + bh * N + om, l, mask=om < N)


@triton.jit
def _flash_exact_out_kernel(
    Q, K, V, O, ROW_MAX, ROW_SUM,
    N: tl.constexpr, scale,
    sqb: tl.constexpr, sqh: tl.constexpr, sqm: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, skn: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svn: tl.constexpr,
    sob: tl.constexpr, soh: tl.constexpr, som: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 64
    h = bh % 64
    kh = h // 8
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.arange(0, BLOCK_N)
    od = tl.arange(0, 128)
    q = tl.load(Q + b * sqb + h * sqh + om[:, None] * sqm + od[None, :],
                mask=om[:, None] < N, other=0.0)
    m = tl.load(ROW_MAX + bh * N + om, mask=om < N, other=0.0)
    l = tl.load(ROW_SUM + bh * N + om, mask=om < N, other=1.0)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    start_n = 0
    while start_n < (pid_m + 1) * BLOCK_M:
        cols = start_n + on
        k = tl.load(K + b * skb + kh * skh + cols[:, None] * skn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        v = tl.load(V + b * svb + kh * svh + cols[:, None] * svn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32).to(tl.bfloat16)
        s = (s.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        valid = (om[:, None] < N) & (cols[None, :] < N) & (om[:, None] >= cols[None, :])
        s = tl.where(valid, s, -float("inf"))
        p = tl.exp(s - m[:, None]) / l[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        start_n += BLOCK_N
    tl.store(O + b * sob + h * soh + om[:, None] * som + od[None, :],
             acc.to(tl.bfloat16), mask=om[:, None] < N)


@triton.jit
def _flash_dq_kernel(
    Q, K, V, DO, DQ, ROW_MAX, ROW_SUM, DELTA,
    N: tl.constexpr, scale,
    sqb: tl.constexpr, sqh: tl.constexpr, sqm: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, skn: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svn: tl.constexpr,
    sdob: tl.constexpr, sdoh: tl.constexpr, sdom: tl.constexpr,
    sdqb: tl.constexpr, sdqh: tl.constexpr, sdqm: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 64
    h = bh % 64
    kh = h // 8
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.arange(0, BLOCK_N)
    od = tl.arange(0, 128)
    q = tl.load(Q + b * sqb + h * sqh + om[:, None] * sqm + od[None, :],
                mask=om[:, None] < N, other=0.0)
    do = tl.load(DO + b * sdob + h * sdoh + om[:, None] * sdom + od[None, :],
                 mask=om[:, None] < N, other=0.0)
    m = tl.load(ROW_MAX + bh * N + om, mask=om < N, other=0.0)
    l = tl.load(ROW_SUM + bh * N + om, mask=om < N, other=1.0)
    delta = tl.zeros((BLOCK_M,), tl.float32)
    start_n = 0
    while start_n < (pid_m + 1) * BLOCK_M:
        cols = start_n + on
        k = tl.load(K + b * skb + kh * skh + cols[:, None] * skn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        v = tl.load(V + b * svb + kh * svh + cols[:, None] * svn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32).to(tl.bfloat16)
        s = (s.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        valid = (om[:, None] < N) & (cols[None, :] < N) & (om[:, None] >= cols[None, :])
        s = tl.where(valid, s, -float("inf"))
        p = tl.exp(s - m[:, None]) / l[:, None]
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        delta += tl.sum(dp * p, axis=1)
        start_n += BLOCK_N
    tl.store(DELTA + bh * N + om, delta, mask=om < N)

    dq = tl.zeros((BLOCK_M, 128), tl.float32)
    start_n = 0
    while start_n < (pid_m + 1) * BLOCK_M:
        cols = start_n + on
        k = tl.load(K + b * skb + kh * skh + cols[:, None] * skn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        v = tl.load(V + b * svb + kh * svh + cols[:, None] * svn + od[None, :],
                    mask=cols[:, None] < N, other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32).to(tl.bfloat16)
        s = (s.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        valid = (om[:, None] < N) & (cols[None, :] < N) & (om[:, None] >= cols[None, :])
        s = tl.where(valid, s, -float("inf"))
        p = tl.exp(s - m[:, None]) / l[:, None]
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        ds = (p * (dp - delta[:, None]) * scale).to(tl.bfloat16)
        dq += tl.dot(ds, k, out_dtype=tl.float32)
        start_n += BLOCK_N
    tl.store(DQ + b * sdqb + h * sdqh + om[:, None] * sdqm + od[None, :],
             dq.to(tl.bfloat16), mask=om[:, None] < N)


@triton.jit
def _flash_dkdv_kernel(
    Q, K, V, DO, DKX, DVX, ROW_MAX, ROW_SUM, DELTA,
    N: tl.constexpr, scale,
    sqb: tl.constexpr, sqh: tl.constexpr, sqm: tl.constexpr,
    skb: tl.constexpr, skh: tl.constexpr, skn: tl.constexpr,
    svb: tl.constexpr, svh: tl.constexpr, svn: tl.constexpr,
    sdob: tl.constexpr, sdoh: tl.constexpr, sdom: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 64
    h = bh % 64
    kh = h // 8
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    om0 = tl.arange(0, BLOCK_M)
    od = tl.arange(0, 128)
    k = tl.load(K + b * skb + kh * skh + cols[:, None] * skn + od[None, :],
                mask=cols[:, None] < N, other=0.0)
    v = tl.load(V + b * svb + kh * svh + cols[:, None] * svn + od[None, :],
                mask=cols[:, None] < N, other=0.0)
    dk = tl.zeros((BLOCK_N, 128), tl.float32)
    dv = tl.zeros((BLOCK_N, 128), tl.float32)
    start_m = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    while start_m < N:
        rows = start_m + om0
        q = tl.load(Q + b * sqb + h * sqh + rows[:, None] * sqm + od[None, :],
                    mask=rows[:, None] < N, other=0.0)
        do = tl.load(DO + b * sdob + h * sdoh + rows[:, None] * sdom + od[None, :],
                     mask=rows[:, None] < N, other=0.0)
        m = tl.load(ROW_MAX + bh * N + rows, mask=rows < N, other=0.0)
        l = tl.load(ROW_SUM + bh * N + rows, mask=rows < N, other=1.0)
        delta = tl.load(DELTA + bh * N + rows, mask=rows < N, other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32).to(tl.bfloat16)
        s = (s.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
        valid = (rows[:, None] < N) & (cols[None, :] < N) & (rows[:, None] >= cols[None, :])
        s = tl.where(valid, s, -float("inf"))
        p = tl.exp(s - m[:, None]) / l[:, None]
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        ds = (p * (dp - delta[:, None]) * scale).to(tl.bfloat16)
        dk += tl.dot(tl.trans(ds), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do, out_dtype=tl.float32)
        start_m += BLOCK_M
    base = ((b * 64 + h) * N + cols)[:, None] * 128 + od[None, :]
    tl.store(DKX + base, dk.to(tl.bfloat16), mask=(cols[:, None] < N))
    tl.store(DVX + base, dv.to(tl.bfloat16), mask=(cols[:, None] < N))


def _flash_forward(q, k, v, n, scale, exact_output=False):
    b = q.shape[0]
    out = torch.empty((b, n, 64, 128), device=q.device, dtype=q.dtype)
    row_max = torch.empty((b, 64, n), device=q.device, dtype=torch.float32)
    row_sum = torch.empty_like(row_max)
    bm, bn = 16, 32
    grid = (triton.cdiv(n, bm), b * 64)
    _flash_fwd_kernel[grid](
        q, k, v, out, row_max, row_sum, n, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(2), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, num_warps=4, num_stages=1)
    if exact_output:
        _flash_exact_out_kernel[grid](
            q, k, v, out, row_max, row_sum, n, scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(2), out.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, num_warps=4, num_stages=1)
    return out, row_max, row_sum


def _flash_backward(q, k, v, do, row_max, row_sum, n, scale):
    b = q.shape[0]
    dq_storage = torch.empty((b, n, 64, 128), device=q.device, dtype=q.dtype)
    dkx = torch.empty((b, 64, n, 128), device=q.device, dtype=q.dtype)
    dvx = torch.empty_like(dkx)
    delta = torch.empty((b, 64, n), device=q.device, dtype=torch.float32)
    bm, bn = 16, 32
    _flash_dq_kernel[(triton.cdiv(n, bm), b * 64)](
        q, k, v, do, dq_storage, row_max, row_sum, delta,
        n, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dq_storage.stride(0), dq_storage.stride(2), dq_storage.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, num_warps=4, num_stages=1)
    _flash_dkdv_kernel[(triton.cdiv(n, bn), b * 64)](
        q, k, v, do, dkx, dvx, row_max, row_sum, delta,
        n, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        BLOCK_M=bm, BLOCK_N=bn, num_warps=4, num_stages=1)
    dq = dq_storage.transpose(1, 2)
    dk = dkx.view(b, 8, 8, n, 128).sum(dim=2)
    dv = dvx.view(b, 8, 8, n, 128).sum(dim=2)
    return dq, dk, dv


def _explicit_forward(q, k, v, n, scale):
    b = q.shape[0]
    kx = k[:, :, None, :, :].expand(b, 8, 8, n, 128).reshape(b, 64, n, 128)
    vx = v[:, :, None, :, :].expand(b, 8, 8, n, 128).reshape(b, 64, n, 128)
    scores = torch.matmul(q, kx.transpose(2, 3))
    scores.mul_(scale)
    scores.masked_fill_(_causal_mask(n, q.device)[None, None], float('-inf'))
    probs_f = F.softmax(scores, dim=-1, dtype=torch.float32)
    probs = probs_f.to(q.dtype)
    attn = torch.matmul(probs, vx).transpose(1, 2).contiguous().view(b, n, 8192)
    return attn, (kx, vx, probs_f, probs)


def _explicit_backward(q, do, state, n, scale):
    b = q.shape[0]
    kx, vx, probs_f, probs = state
    dp = torch.matmul(do, vx.transpose(2, 3))
    dvx = torch.matmul(probs.transpose(2, 3), do)
    dpf = dp.float()
    row_dot = (dpf * probs_f).sum(dim=-1, keepdim=True)
    dpf.sub_(row_dot)
    dpf.mul_(probs_f)
    dpf.mul_(scale)
    ds = dpf.to(q.dtype)
    dq = torch.matmul(ds, kx)
    dkx = torch.matmul(q.transpose(2, 3), ds).transpose(2, 3)
    dk = dkx.view(b, 8, 8, n, 128).sum(dim=2)
    dv = dvx.view(b, 8, 8, n, 128).sum(dim=2)
    return dq, dk, dv


@torch.no_grad()
def run(grad_output, hidden_states, q_weight, k_weight, v_weight, o_weight,
        inv_freq, scaling):
    batch_size, seq_len, hidden_size = hidden_states.shape
    nh, nkh, hd, groups = 64, 8, 128, 8
    dtype = hidden_states.dtype

    q = F.linear(hidden_states, q_weight).view(batch_size, seq_len, nh, hd).transpose(1, 2)
    k = F.linear(hidden_states, k_weight).view(batch_size, seq_len, nkh, hd).transpose(1, 2)
    v = F.linear(hidden_states, v_weight).view(batch_size, seq_len, nkh, hd).transpose(1, 2)

    positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.float32)
    freqs = (inv_freq[None, :, None].float() @ positions[None, None, :]).transpose(1, 2)
    cos = freqs.cos().unsqueeze(1).repeat_interleave(2, dim=-1)
    sin = freqs.sin().unsqueeze(1).repeat_interleave(2, dim=-1)

    qf, kf = q.float(), k.float()
    qr = torch.stack((-qf[..., 1::2], qf[..., 0::2]), dim=-1).flatten(-2)
    kr = torch.stack((-kf[..., 1::2], kf[..., 0::2]), dim=-1).flatten(-2)
    q = (qf * cos + qr * sin).to(dtype)
    k = (kf * cos + kr * sin).to(dtype)

    use_flash = seq_len < 768 or batch_size >= 4
    if use_flash:
        exact_output = batch_size * seq_len > 20000
        attn_storage, row_max, row_sum = _flash_forward(
            q, k, v, seq_len, scaling, exact_output)
        attn = attn_storage.view(batch_size, seq_len, hidden_size)
    else:
        attn, attention_state = _explicit_forward(q, k, v, seq_len, scaling)
    da = F.linear(grad_output, o_weight.t())
    dow = grad_output.reshape(-1, hidden_size).t() @ attn.reshape(-1, hidden_size)
    da = da.view(batch_size, seq_len, nh, hd).transpose(1, 2)
    if use_flash:
        dq, dk, dv = _flash_backward(q, k, v, da, row_max, row_sum, seq_len, scaling)
    else:
        dq, dk, dv = _explicit_backward(q, da, attention_state, seq_len, scaling)

    dqf = dq.float()
    dq_out = dqf * cos
    dq_rot = (dqf * sin).view(*dqf.shape[:-1], hd // 2, 2)
    dq_out[..., 0::2].add_(dq_rot[..., 1])
    dq_out[..., 1::2].sub_(dq_rot[..., 0])
    dq = dq_out.to(dtype)

    dkf = dk.float()
    dk_out = dkf * cos
    dk_rot = (dkf * sin).view(*dkf.shape[:-1], hd // 2, 2)
    dk_out[..., 0::2].add_(dk_rot[..., 1])
    dk_out[..., 1::2].sub_(dk_rot[..., 0])
    dk = dk_out.to(dtype)

    dq = dq.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    dk = dk.transpose(1, 2).contiguous().view(batch_size, seq_len, nkh * hd)
    dv = dv.transpose(1, 2).contiguous().view(batch_size, seq_len, nkh * hd)

    dxq = F.linear(dq, q_weight.t())
    dxk = F.linear(dk, k_weight.t())
    dxv = F.linear(dv, v_weight.t())
    dxq.add_(dxk)
    dxq.add_(dxv)
    dx = dxq
    x2 = hidden_states.reshape(-1, hidden_size)
    dqw = dq.reshape(-1, hidden_size).t() @ x2
    dkw = dk.reshape(-1, nkh * hd).t() @ x2
    dvw = dv.reshape(-1, nkh * hd).t() @ x2
    return dx.to(dtype), dqw.to(dtype), dkw.to(dtype), dvw.to(dtype), dow.to(dtype)
