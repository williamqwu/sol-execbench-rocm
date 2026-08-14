import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _pack_qkv_kernel(qkv, cu, cos, sin, packed, n_elements: tl.constexpr,
                     BLOCK: tl.constexpr):
    sh = tl.program_id(0)
    seq = sh // 16
    head = sh % 16
    end = tl.load(cu + seq)
    start = tl.where(seq == 0, 0, tl.load(cu + seq - 1))
    length = end - start
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pos = offs // 36
    d = offs % 36
    mask = pos < length
    token = start + pos
    within1 = head * 72 + d
    within2 = within1 + 36
    base1 = token * 3456 + within1
    base2 = token * 3456 + within2
    emb1 = token * 1152 + within1
    emb2 = token * 1152 + within2
    c1 = tl.load(cos + emb1, mask=mask)
    c2 = tl.load(cos + emb2, mask=mask)
    s1 = tl.load(sin + emb1, mask=mask)
    s2 = tl.load(sin + emb2, mask=mask)
    dst1 = start * 1152 + head * length * 72 + pos * 72 + d
    dst2 = dst1 + 36

    q1 = tl.load(qkv + base1, mask=mask)
    q2 = tl.load(qkv + base2, mask=mask)
    qo1 = ((q1 * c1).to(tl.bfloat16) +
           ((-q2) * s1).to(tl.bfloat16)).to(tl.bfloat16)
    qo2 = ((q2 * c2).to(tl.bfloat16) +
           (q1 * s2).to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(packed + dst1, qo1, mask=mask)
    tl.store(packed + dst2, qo2, mask=mask)

    k1 = tl.load(qkv + base1 + 1152, mask=mask)
    k2 = tl.load(qkv + base2 + 1152, mask=mask)
    ko1 = ((k1 * c1).to(tl.bfloat16) +
           ((-k2) * s1).to(tl.bfloat16)).to(tl.bfloat16)
    ko2 = ((k2 * c2).to(tl.bfloat16) +
           (k1 * s2).to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(packed + n_elements + dst1, ko1, mask=mask)
    tl.store(packed + n_elements + dst2, ko2, mask=mask)
    v1 = tl.load(qkv + base1 + 2304, mask=mask)
    v2 = tl.load(qkv + base2 + 2304, mask=mask)
    tl.store(packed + 2 * n_elements + dst1, v1, mask=mask)
    tl.store(packed + 2 * n_elements + dst2, v2, mask=mask)


def _pack_qkv(qkv, cu_seqlens, cos, sin, max_len):
    n_elements = cos.numel()
    packed = torch.empty((3, n_elements), device=qkv.device, dtype=qkv.dtype)
    grid = (cu_seqlens.numel() * 16, triton.cdiv(max_len * 36, 256))
    _pack_qkv_kernel[grid](qkv, cu_seqlens, cos, sin, packed, n_elements,
                           BLOCK=256, num_warps=2)
    return packed


@triton.jit
def _pack_padded_kernel(qkv, cu, cos, sin, packed, MAX_LEN: tl.constexpr,
                        BLOCK: tl.constexpr):
    sh = tl.program_id(0)
    seq = sh // 16
    head = sh % 16
    end = tl.load(cu + seq)
    start = tl.where(seq == 0, 0, tl.load(cu + seq - 1))
    length = end - start
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pos = offs // 36
    d = offs % 36
    in_range = pos < MAX_LEN
    valid = pos < length
    token = start + pos
    within1 = head * 72 + d
    within2 = within1 + 36
    base1 = token * 3456 + within1
    base2 = token * 3456 + within2
    emb1 = token * 1152 + within1
    emb2 = token * 1152 + within2
    c1 = tl.load(cos + emb1, mask=valid, other=0.0)
    c2 = tl.load(cos + emb2, mask=valid, other=0.0)
    s1 = tl.load(sin + emb1, mask=valid, other=0.0)
    s2 = tl.load(sin + emb2, mask=valid, other=0.0)
    dst1 = (seq * (16 * MAX_LEN * 72) + head * (MAX_LEN * 72) +
            pos * 72 + d)
    dst2 = dst1 + 36

    q1 = tl.load(qkv + base1, mask=valid, other=0.0)
    q2 = tl.load(qkv + base2, mask=valid, other=0.0)
    qo1 = ((q1 * c1).to(tl.bfloat16) +
           ((-q2) * s1).to(tl.bfloat16)).to(tl.bfloat16)
    qo2 = ((q2 * c2).to(tl.bfloat16) +
           (q1 * s2).to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(packed + dst1, qo1, mask=in_range)
    tl.store(packed + dst2, qo2, mask=in_range)

    plane = tl.num_programs(0) * MAX_LEN * 72
    k1 = tl.load(qkv + base1 + 1152, mask=valid, other=0.0)
    k2 = tl.load(qkv + base2 + 1152, mask=valid, other=0.0)
    ko1 = ((k1 * c1).to(tl.bfloat16) +
           ((-k2) * s1).to(tl.bfloat16)).to(tl.bfloat16)
    ko2 = ((k2 * c2).to(tl.bfloat16) +
           (k1 * s2).to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(packed + plane + dst1, ko1, mask=in_range)
    tl.store(packed + plane + dst2, ko2, mask=in_range)
    v1 = tl.load(qkv + base1 + 2304, mask=valid, other=0.0)
    v2 = tl.load(qkv + base2 + 2304, mask=valid, other=0.0)
    tl.store(packed + 2 * plane + dst1, v1, mask=in_range)
    tl.store(packed + 2 * plane + dst2, v2, mask=in_range)


@triton.jit
def _scale_mask_kernel(scores, cu, MAX_LEN: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    seq = row // (16 * MAX_LEN)
    end = tl.load(cu + seq)
    start = tl.where(seq == 0, 0, tl.load(cu + seq - 1))
    length = end - start
    cols = tl.arange(0, BLOCK)
    mask = cols < MAX_LEN
    vals = tl.load(scores + row * MAX_LEN + cols, mask=mask)
    vals = (vals.to(tl.float32) * 0.11785113019775793).to(tl.bfloat16)
    vals = tl.where(cols < length, vals, -float("inf"))
    tl.store(scores + row * MAX_LEN + cols, vals, mask=mask)


@triton.jit
def _unpack_kernel(heads, cu, attn, MAX_LEN: tl.constexpr,
                   BLOCK: tl.constexpr):
    sh = tl.program_id(0)
    seq = sh // 16
    head = sh % 16
    end = tl.load(cu + seq)
    start = tl.where(seq == 0, 0, tl.load(cu + seq - 1))
    length = end - start
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pos = offs // 72
    d = offs % 72
    mask = pos < length
    src = seq * (16 * MAX_LEN * 72) + head * (MAX_LEN * 72) + offs
    dst = (start + pos) * 1152 + head * 72 + d
    vals = tl.load(heads + src, mask=mask)
    tl.store(attn + dst, vals, mask=mask)


def _padded_attention(qkv, cu_seqlens, cos, sin, max_len):
    sequences = cu_seqlens.numel()
    packed = torch.empty((3, sequences, 16, max_len, 72),
                         device=qkv.device, dtype=qkv.dtype)
    grid = (sequences * 16, triton.cdiv(max_len * 36, 256))
    _pack_padded_kernel[grid](qkv, cu_seqlens, cos, sin, packed,
                              MAX_LEN=max_len, BLOCK=256, num_warps=2)
    q = packed[0].flatten(0, 1)
    k = packed[1].flatten(0, 1)
    v = packed[2].flatten(0, 1)
    scores = torch.bmm(q, k.transpose(1, 2))
    block = triton.next_power_of_2(max_len)
    _scale_mask_kernel[(sequences * 16 * max_len,)](
        scores, cu_seqlens, MAX_LEN=max_len, BLOCK=block, num_warps=1)
    probs = F.softmax(scores, dim=-1)
    heads = torch.bmm(probs, v).view(sequences, 16, max_len, 72)
    attn = torch.empty((qkv.shape[0], 16, 72), device=qkv.device,
                       dtype=qkv.dtype)
    unpack_grid = (sequences * 16, triton.cdiv(max_len * 72, 512))
    _unpack_kernel[unpack_grid](heads, cu_seqlens, attn, MAX_LEN=max_len,
                                BLOCK=512, num_warps=4)
    return attn


_cached_cu = None
_cached_cu_version = None
_cached_desc = None


def _sequence_desc(cu_seqlens):
    global _cached_cu, _cached_cu_version, _cached_desc
    version = cu_seqlens._version
    if cu_seqlens is _cached_cu and version == _cached_cu_version:
        return _cached_desc
    ends = cu_seqlens.tolist()
    starts = [0] + ends[:-1]
    runs = []
    max_len = 0
    for start, end in zip(starts, ends):
        length = end - start
        if length <= 0:
            continue
        max_len = max(max_len, length)
        if runs and runs[-1][2] == length and runs[-1][1] == start:
            runs[-1] = (runs[-1][0], end, length)
        else:
            runs.append((start, end, length))
    _cached_cu = cu_seqlens
    _cached_cu_version = version
    _cached_desc = (runs, max_len)
    return _cached_desc


@torch.no_grad()
def run(hidden_states, cu_seqlens, cos, sin, qkv_weight, qkv_bias,
        proj_weight, proj_bias):
    n = hidden_states.shape[0]
    # Fetch the tiny sequence descriptor before launching any compute, so this
    # synchronization cannot stall the projection below.  Equal-length runs
    # can be evaluated as a single strided-batched GEMM without changing the
    # arithmetic performed for any sequence.
    runs, max_len = _sequence_desc(cu_seqlens)

    if not runs:
        return torch.zeros_like(hidden_states)

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias).reshape(n, 3, 16, 72)
    if len(runs) > 1:
        attn = _padded_attention(qkv, cu_seqlens, cos, sin, max_len)
        return F.linear(attn.reshape(n, 1152), proj_weight, proj_bias)

    packed = _pack_qkv(qkv, cu_seqlens, cos, sin, max_len)

    attn = torch.empty((n, 16, 72), device=hidden_states.device,
                       dtype=hidden_states.dtype)
    scale = 72 ** -0.5
    for start, end, length in runs:
        count = (end - start) // length
        offset = start * 1152
        size = (end - start) * 1152
        qi = packed[0].narrow(0, offset, size).view(count, 16, length, 72)
        ki = packed[1].narrow(0, offset, size).view(count, 16, length, 72)
        vi = packed[2].narrow(0, offset, size).view(count, 16, length, 72)
        qi = qi.flatten(0, 1)
        ki = ki.flatten(0, 1)
        vi = vi.flatten(0, 1)
        scores = torch.bmm(qi, ki.transpose(1, 2))
        scores.mul_(scale)
        # The BF16 softmax kernel accumulates in FP32 and directly performs
        # the same final BF16 conversion as `.softmax(dtype=float32).to(bf16)`.
        probs = F.softmax(scores, dim=-1)
        out = torch.bmm(probs, vi).view(count, 16, length, 72)
        out = out.permute(0, 2, 1, 3)
        attn[start:end].view(count, length, 16, 72).copy_(out)

    return F.linear(attn.reshape(n, 1152), proj_weight, proj_bias)
