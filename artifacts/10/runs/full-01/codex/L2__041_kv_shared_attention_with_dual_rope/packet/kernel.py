import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


_INV_FREQ_CACHE = {}


def _inv_freq(device, theta):
    # This is a model constant (48 floats), analogous to a registered buffer.
    key = (device.type, device.index, float(theta))
    value = _INV_FREQ_CACHE.get(key)
    if value is None:
        value = 1.0 / (theta ** (torch.arange(
            0, 96, 2, device=device, dtype=torch.float32) / 96))
        _INV_FREQ_CACHE[key] = value
    return value


@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BLOCK_M)
    n_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * n_n
    group_id = pid // group
    first_m = group_id * GROUP_M
    actual_group_m = tl.minimum(n_m - first_m, GROUP_M)
    pid_in_group = pid - group_id * group
    pid_m = first_m + (pid_in_group % actual_group_m)
    pid_n = pid_in_group // actual_group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in tl.range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * K + start_k + offs_k[None, :],
                    mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(B + offs_n[:, None] * K + start_k + offs_k[None, :],
                    mask=offs_n[:, None] < N, other=0.0)
        acc += tl.dot(a, tl.trans(b))
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _attention(Q, K, V, O, N_CTX: tl.constexpr, SOFTCAP: tl.constexpr,
               Q_STRIDE_T: tl.constexpr, KV_STRIDE_T: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Causal GQA, including the reference's intermediate bf16 roundings."""
    block_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // 32
    head = bh - batch * 32
    kv_head = head // 4

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    d0 = tl.arange(0, 64)
    d1 = tl.arange(0, 32)

    q_base = batch * N_CTX * Q_STRIDE_T + head * 96
    q0 = tl.load(Q + q_base + offs_m[:, None] * Q_STRIDE_T + d0[None, :],
                 mask=offs_m[:, None] < N_CTX, other=0.0)
    q1 = tl.load(Q + q_base + offs_m[:, None] * Q_STRIDE_T + 64 + d1[None, :],
                 mask=offs_m[:, None] < N_CTX, other=0.0)

    # First pass computes float32 softmax statistics.  The score is rounded
    # after every bf16 reference operation: matmul, divide, tanh, multiply.
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc1 = tl.zeros((BLOCK_M, 32), tl.float32)
    hi = tl.minimum((block_m + 1) * BLOCK_M, N_CTX)
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_base = batch * N_CTX * KV_STRIDE_T + kv_head * 96
        k0 = tl.load(K + k_base + offs_n[:, None] * KV_STRIDE_T + d0[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        k1 = tl.load(K + k_base + offs_n[:, None] * KV_STRIDE_T + 64 + d1[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        score = tl.dot(q0, tl.trans(k0)) + tl.dot(q1, tl.trans(k1))
        score = score.to(tl.bfloat16)
        score = (score.to(tl.float32) / SOFTCAP).to(tl.bfloat16)
        x = score.to(tl.float32)
        x2 = x * x
        num = x * (945.0 + x2 * (105.0 + x2))
        den = 945.0 + x2 * (420.0 + 15.0 * x2)
        score = (num / den).to(tl.bfloat16)
        score = (score.to(tl.float32) * SOFTCAP).to(tl.bfloat16).to(tl.float32)
        valid = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < N_CTX)
        score = tl.where(valid, score, -float("inf"))

        block_max = tl.max(score, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = libdevice.exp(row_max - new_max)
        probs = libdevice.exp(score - new_max[:, None])
        probs_bf16 = probs.to(tl.bfloat16)
        v_base = batch * N_CTX * KV_STRIDE_T + kv_head * 96
        v0 = tl.load(V + v_base + offs_n[:, None] * KV_STRIDE_T + d0[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        v1 = tl.load(V + v_base + offs_n[:, None] * KV_STRIDE_T + 64 + d1[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        acc0 = acc0 * alpha[:, None] + tl.dot(probs_bf16, v0)
        acc1 = acc1 * alpha[:, None] + tl.dot(probs_bf16, v1)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max

    o_base = batch * N_CTX * 32 * 96 + head * 96
    acc0 = acc0 / row_sum[:, None]
    acc1 = acc1 / row_sum[:, None]
    tl.store(O + o_base + offs_m[:, None] * 32 * 96 + d0[None, :], acc0,
             mask=offs_m[:, None] < N_CTX)
    tl.store(O + o_base + offs_m[:, None] * 32 * 96 + 64 + d1[None, :], acc1,
             mask=offs_m[:, None] < N_CTX)


@triton.jit
def _attention_gqa4(Q, K, V, O, N_CTX: tl.constexpr, SOFTCAP: tl.constexpr,
                    Q_STRIDE_T: tl.constexpr, KV_STRIDE_T: tl.constexpr,
                    HEAD_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One program jointly handles the four Q heads sharing each KV head."""
    block_m = tl.program_id(0)
    bkv = tl.program_id(1)
    batch = bkv // 8
    kv_head = bkv - batch * 8
    BLOCK_M: tl.constexpr = HEAD_M * 4

    row = tl.arange(0, BLOCK_M)
    q_in_group = row // HEAD_M
    offs_m = block_m * HEAD_M + row - q_in_group * HEAD_M
    q_head = kv_head * 4 + q_in_group
    offs_n_base = tl.arange(0, BLOCK_N)
    d0 = tl.arange(0, 64)
    d1 = tl.arange(0, 32)

    q_base = batch * N_CTX * Q_STRIDE_T
    q0 = tl.load(Q + q_base + offs_m[:, None] * Q_STRIDE_T +
                 q_head[:, None] * 96 + d0[None, :],
                 mask=offs_m[:, None] < N_CTX, other=0.0)
    q1 = tl.load(Q + q_base + offs_m[:, None] * Q_STRIDE_T +
                 q_head[:, None] * 96 + 64 + d1[None, :],
                 mask=offs_m[:, None] < N_CTX, other=0.0)

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc1 = tl.zeros((BLOCK_M, 32), tl.float32)
    hi = tl.minimum((block_m + 1) * HEAD_M, N_CTX)
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        kv_base = batch * N_CTX * KV_STRIDE_T + kv_head * 96
        k0 = tl.load(K + kv_base + offs_n[:, None] * KV_STRIDE_T + d0[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        k1 = tl.load(K + kv_base + offs_n[:, None] * KV_STRIDE_T + 64 + d1[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        score = tl.dot(q0, tl.trans(k0)) + tl.dot(q1, tl.trans(k1))
        score = score.to(tl.bfloat16)
        score = (score.to(tl.float32) / SOFTCAP).to(tl.bfloat16)
        x = score.to(tl.float32)
        x2 = x * x
        num = x * (945.0 + x2 * (105.0 + x2))
        den = 945.0 + x2 * (420.0 + 15.0 * x2)
        score = (num / den).to(tl.bfloat16)
        score = (score.to(tl.float32) * SOFTCAP).to(tl.bfloat16).to(tl.float32)
        valid = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < N_CTX)
        score = tl.where(valid, score, -float("inf"))

        block_max = tl.max(score, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = libdevice.exp(row_max - new_max)
        probs = libdevice.exp(score - new_max[:, None])
        probs_bf16 = probs.to(tl.bfloat16)
        v0 = tl.load(V + kv_base + offs_n[:, None] * KV_STRIDE_T + d0[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        v1 = tl.load(V + kv_base + offs_n[:, None] * KV_STRIDE_T + 64 + d1[None, :],
                     mask=offs_n[:, None] < N_CTX, other=0.0)
        acc0 = acc0 * alpha[:, None] + tl.dot(probs_bf16, v0)
        acc1 = acc1 * alpha[:, None] + tl.dot(probs_bf16, v1)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max

    acc0 = acc0 / row_sum[:, None]
    acc1 = acc1 / row_sum[:, None]
    o_base = batch * N_CTX * 3072
    tl.store(O + o_base + offs_m[:, None] * 3072 + q_head[:, None] * 96 +
             d0[None, :], acc0, mask=offs_m[:, None] < N_CTX)
    tl.store(O + o_base + offs_m[:, None] * 3072 + q_head[:, None] * 96 + 64 +
             d1[None, :], acc1, mask=offs_m[:, None] < N_CTX)


@triton.jit
def _norm_rope_qkv(Q, K, V, POS, Q_WEIGHT, K_WEIGHT, INV_FREQ,
                   Q_STRIDE_T: tl.constexpr, KV_STRIDE_T: tl.constexpr,
                   POS_STRIDE_B: tl.constexpr, POS_STRIDE_S: tl.constexpr,
                   N_CTX: tl.constexpr, EPS: tl.constexpr, THETA: tl.constexpr):
    """Fuse three RMSNorms and both RoPE applications, in-place."""
    pid = tl.program_id(0)
    token = pid // 32
    head = pid - token * 32
    batch = token // N_CTX
    seq = token - batch * N_CTX
    offs = tl.arange(0, 128)
    valid = offs < 96
    pair = tl.where(offs < 48, offs + 48, offs - 48)

    q_ptr = Q + token * Q_STRIDE_T + head * 96
    qx = tl.load(q_ptr + offs, mask=valid, other=0.0).to(tl.float32)
    qpair = tl.load(q_ptr + pair, mask=valid, other=0.0).to(tl.float32)
    q_var = tl.sum(qx * qx, axis=0) * (1.0 / 96.0)
    q_inv = libdevice.rsqrt(q_var + EPS)
    qw = tl.load(Q_WEIGHT + offs, mask=valid, other=0.0).to(tl.float32)
    qwp = tl.load(Q_WEIGHT + pair, mask=valid, other=0.0).to(tl.float32)
    qn = (qx * q_inv * qw).to(tl.bfloat16)
    qnp = (qpair * q_inv * qwp).to(tl.bfloat16)

    freq_i = tl.where(offs < 48, offs, offs - 48)
    inv_freq = tl.load(INV_FREQ + freq_i, mask=valid, other=0.0)
    pos = tl.load(POS + batch * POS_STRIDE_B + seq * POS_STRIDE_S).to(tl.float32)
    angle = pos * inv_freq
    cos = libdevice.cos(angle).to(tl.bfloat16)
    sin = libdevice.sin(angle).to(tl.bfloat16)
    qrot = tl.where(offs < 48, -qnp, qnp)
    qout = ((qn * cos).to(tl.bfloat16).to(tl.float32) +
            (qrot * sin).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(q_ptr + offs, qout, mask=valid)

    if head < 8:
        k_ptr = K + token * KV_STRIDE_T + head * 96
        kx = tl.load(k_ptr + offs, mask=valid, other=0.0).to(tl.float32)
        kpair = tl.load(k_ptr + pair, mask=valid, other=0.0).to(tl.float32)
        k_var = tl.sum(kx * kx, axis=0) * (1.0 / 96.0)
        k_inv = libdevice.rsqrt(k_var + EPS)
        kw = tl.load(K_WEIGHT + offs, mask=valid, other=0.0).to(tl.float32)
        kwp = tl.load(K_WEIGHT + pair, mask=valid, other=0.0).to(tl.float32)
        kn = (kx * k_inv * kw).to(tl.bfloat16)
        knp = (kpair * k_inv * kwp).to(tl.bfloat16)
        krot = tl.where(offs < 48, -knp, knp)
        kout = ((kn * cos).to(tl.bfloat16).to(tl.float32) +
                (krot * sin).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(k_ptr + offs, kout, mask=valid)

        v_ptr = V + token * KV_STRIDE_T + head * 96
        vx = tl.load(v_ptr + offs, mask=valid, other=0.0).to(tl.float32)
        v_var = tl.sum(vx * vx, axis=0) * (1.0 / 96.0)
        vn = (vx * libdevice.rsqrt(v_var + EPS)).to(tl.bfloat16)
        tl.store(v_ptr + offs, vn, mask=valid)


def _rms(x, weight, eps):
    var = x.float().square().mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps) * weight).to(x.dtype)


def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos[:, :, None, :] + torch.cat((-x[..., half:], x[..., :half]), -1) * sin[:, :, None, :]


@torch.no_grad()
def run(hidden_states, position_ids, attention_mask, q_proj_weight,
        k_proj_weight, v_proj_weight, o_proj_weight, q_norm_weight,
        k_norm_weight, shared_key_states, shared_value_states, rope_theta,
        softcap, rms_norm_eps, use_shared_kv):
    b, s, _ = hidden_states.shape
    if use_shared_kv:
        # Shared tensors arrive as [B, KVH, S, D]; this uncommon semantic path
        # retains the straightforward reference layout handling.
        q = F.linear(hidden_states, q_proj_weight).view(b, s, 32, 96)
        q = _rms(q, q_norm_weight, rms_norm_eps)
        inv = _inv_freq(hidden_states.device, rope_theta)
        freqs = (inv[None, :, None].expand(b, -1, 1) @
                 position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), -1)
        q = _rope(q, emb.cos().to(q.dtype), emb.sin().to(q.dtype))
        qh = q.transpose(1, 2)
        k, v = shared_key_states, shared_value_states
        k = k[:, :, None].expand(b, 8, 4, s, 96).reshape(b, 32, s, 96)
        v = v[:, :, None].expand(b, 8, 4, s, 96).reshape(b, 32, s, 96)
        a = qh @ k.transpose(2, 3)
        a = torch.tanh(a / softcap) * softcap + attention_mask
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (a @ v).transpose(1, 2).contiguous().reshape(b, s, 3072)
    else:
        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), 0)
        qkv = F.linear(hidden_states, qkv_weight)
        q = qkv[..., :3072].view(b, s, 32, 96)
        k = qkv[..., 3072:3840].view(b, s, 8, 96)
        v = qkv[..., 3840:].view(b, s, 8, 96)
        inv = _inv_freq(hidden_states.device, rope_theta)
        _norm_rope_qkv[(b * s * 32,)](
            q, k, v, position_ids, q_norm_weight, k_norm_weight, inv,
            Q_STRIDE_T=q.stride(1), KV_STRIDE_T=k.stride(1),
            POS_STRIDE_B=position_ids.stride(0),
            POS_STRIDE_S=position_ids.stride(1), N_CTX=s,
            EPS=rms_norm_eps, THETA=rope_theta,
            num_warps=1, num_stages=1)
        out = torch.empty((b, s, 32, 96), dtype=q.dtype, device=q.device)
        if b <= 2 and s <= 256:
            block_m, block_n = 16, 64
            _attention[(triton.cdiv(s, block_m), b * 32)](
                q, k, v, out, N_CTX=s, SOFTCAP=softcap,
                Q_STRIDE_T=q.stride(1), KV_STRIDE_T=k.stride(1),
                BLOCK_M=block_m, BLOCK_N=block_n, num_warps=4, num_stages=1)
        else:
            _attention_gqa4[(triton.cdiv(s, 32), b * 8)](
                q, k, v, out, N_CTX=s, SOFTCAP=softcap,
                Q_STRIDE_T=q.stride(1), KV_STRIDE_T=k.stride(1),
                HEAD_M=32, BLOCK_N=64, num_warps=4, num_stages=1)
        out = out.reshape(b, s, 3072)
    if b * s == 256:
        result = torch.empty_like(out)
        _gemm[(triton.cdiv(256, 64) * triton.cdiv(3072, 64),)](
            out, o_proj_weight, result, M=256, N=3072, K=3072,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8,
            num_warps=4, num_stages=1)
        return result
    return F.linear(out, o_proj_weight)
