import torch
import triton
import triton.language as tl


_BLOCK_T = 32
_BLOCK_D = 64
_PARTIAL_CHUNK_T = 256


@triton.jit
def _logits_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    logits,
    sm_scale: tl.constexpr,
    num_kv_indices: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    head = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_mask = offs_t < num_kv_indices

    batch = tl.zeros((BLOCK_T,), dtype=tl.int32)
    for b in range(1, batch_size):
        start_b = tl.load(kv_indptr + b)
        batch = tl.where(offs_t >= start_b, b, batch)

    pages = tl.load(kv_indices + offs_t, mask=tok_mask, other=0).to(tl.int64)

    dim = tl.arange(0, BLOCK_D)
    score = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for base in range(0, 512, BLOCK_D):
        d = base + dim
        k = tl.load(
            ckv_cache + pages[:, None] * 512 + d[None, :],
            mask=tok_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            q_nope + ((batch[:, None] * 16 + head) * 512 + d[None, :]),
            mask=tok_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k * q, axis=1)

    kp = tl.load(
        kpe_cache + pages[:, None] * 64 + dim[None, :],
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    qp = tl.load(
        q_pe + ((batch[:, None] * 16 + head) * 64 + dim[None, :]),
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    score += tl.sum(kp * qp, axis=1)

    tl.store(logits + offs_t * 16 + head, score * sm_scale, mask=tok_mask)


@triton.jit
def _logits_seq_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    logits,
    sm_scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)

    beg = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    pos = beg + block * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_mask = pos < end
    pages = tl.load(kv_indices + pos, mask=tok_mask, other=0).to(tl.int64)

    dim = tl.arange(0, BLOCK_D)
    score = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for base in range(0, 512, BLOCK_D):
        d = base + dim
        k = tl.load(
            ckv_cache + pages[:, None] * 512 + d[None, :],
            mask=tok_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        q = tl.load(q_nope + (batch * 16 + head) * 512 + d, mask=dim < BLOCK_D).to(
            tl.float32
        )
        score += tl.sum(k * q[None, :], axis=1)

    kp = tl.load(
        kpe_cache + pages[:, None] * 64 + dim[None, :],
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    qp = tl.load(q_pe + (batch * 16 + head) * 64 + dim, mask=dim < BLOCK_D).to(tl.float32)
    score += tl.sum(kp * qp[None, :], axis=1)

    tl.store(logits + pos * 16 + head, score * sm_scale, mask=tok_mask)


@triton.jit
def _lse_kernel(
    kv_indptr,
    logits,
    lse,
    BLOCK_T: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, BLOCK_T)

    beg = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)

    m = -float("inf")
    denom = 0.0
    cur = beg
    while cur < end:
        pos = cur + offs
        vals = tl.load(logits + pos * 16 + head, mask=pos < end, other=-float("inf"))
        block_m = tl.max(vals, axis=0)
        new_m = tl.maximum(m, block_m)
        denom = denom * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m), axis=0)
        m = new_m
        cur += BLOCK_T

    out = (tl.log(denom) + m) * 1.4426950408889634
    out = tl.where(beg < end, out, -float("inf"))
    tl.store(lse + batch * 16 + head, out)


@triton.jit
def _output_kernel(
    ckv_cache,
    kv_indptr,
    kv_indices,
    logits,
    lse,
    output,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dim_block = tl.program_id(2)

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)

    beg = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    lse_nat = tl.load(lse + batch * 16 + head) * 0.6931471805599453

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    cur = beg
    while cur < end:
        pos = cur + offs_t
        tok_mask = pos < end
        pages = tl.load(kv_indices + pos, mask=tok_mask, other=0).to(tl.int64)
        logit = tl.load(logits + pos * 16 + head, mask=tok_mask, other=-float("inf"))
        weight = tl.exp(logit - lse_nat)
        vals = tl.load(
            ckv_cache + pages[:, None] * 512 + offs_d[None, :],
            mask=tok_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vals * weight[:, None], axis=0)
        cur += BLOCK_T

    acc = tl.where(beg < end, acc, 0.0)
    tl.store(output + (batch * 16 + head) * 512 + offs_d, acc)


@triton.jit
def _output_partial_kernel(
    ckv_cache,
    kv_indptr,
    kv_indices,
    chunk_batch,
    chunk_start,
    logits,
    lse,
    partials,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CHUNK_T: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    dim_block = tl.program_id(2)

    batch = tl.load(chunk_batch + chunk)
    start = tl.load(chunk_start + chunk)
    seq_end = tl.load(kv_indptr + batch + 1)
    end = tl.minimum(start + CHUNK_T, seq_end)

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    lse_nat = tl.load(lse + batch * 16 + head) * 0.6931471805599453

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    cur = start
    while cur < end:
        pos = cur + offs_t
        tok_mask = pos < end
        pages = tl.load(kv_indices + pos, mask=tok_mask, other=0).to(tl.int64)
        logit = tl.load(logits + pos * 16 + head, mask=tok_mask, other=-float("inf"))
        weight = tl.exp(logit - lse_nat)
        vals = tl.load(
            ckv_cache + pages[:, None] * 512 + offs_d[None, :],
            mask=tok_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vals * weight[:, None], axis=0)
        cur += BLOCK_T

    tl.store(partials + (chunk * 16 + head) * 512 + offs_d, acc)


@triton.jit
def _output_reduce_kernel(
    chunk_indptr,
    partials,
    output,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dim_block = tl.program_id(2)

    offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    chunk_beg = tl.load(chunk_indptr + batch)
    chunk_end = tl.load(chunk_indptr + batch + 1)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    cur = chunk_beg
    while cur < chunk_end:
        vals = tl.load(partials + (cur * 16 + head) * 512 + offs_d)
        acc += vals
        cur += 1

    tl.store(output + (batch * 16 + head) * 512 + offs_d, acc)


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _make_chunk_map(kv_indptr, batch_size, chunk_size, device):
    indptr = kv_indptr.detach().cpu().tolist()
    starts = []
    batches = []
    chunk_ptr = [0]
    for b in range(batch_size):
        beg = indptr[b]
        end = indptr[b + 1]
        for start in range(beg, end, chunk_size):
            starts.append(start)
            batches.append(b)
        chunk_ptr.append(len(starts))

    chunk_batch = torch.tensor(batches, dtype=torch.int32, device=device)
    chunk_start = torch.tensor(starts, dtype=torch.int32, device=device)
    chunk_indptr = torch.tensor(chunk_ptr, dtype=torch.int32, device=device)
    return chunk_batch, chunk_start, chunk_indptr


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size = q_nope.shape[0]
    num_kv_indices = kv_indices.shape[0]
    output = torch.empty((batch_size, 16, 512), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty((batch_size, 16), dtype=torch.float32, device=q_nope.device)

    if num_kv_indices == 0:
        output.zero_()
        lse.fill_(-float("inf"))
        return {"output": output, "lse": lse}

    logits = torch.empty((num_kv_indices, 16), dtype=torch.float32, device=q_nope.device)
    scale = _as_float(sm_scale)

    grid_logits = (triton.cdiv(num_kv_indices, _BLOCK_T), 16)
    _logits_kernel[grid_logits](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_indptr,
        kv_indices,
        logits,
        scale,
        num_kv_indices,
        batch_size,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        num_warps=8,
    )

    _lse_kernel[(batch_size, 16)](
        kv_indptr,
        logits,
        lse,
        BLOCK_T=_BLOCK_T,
        num_warps=1,
    )

    if batch_size >= 64 and num_kv_indices >= 9000:
        chunk_batch, chunk_start, chunk_indptr = _make_chunk_map(
            kv_indptr, batch_size, _PARTIAL_CHUNK_T, q_nope.device
        )
        num_chunks = chunk_start.shape[0]
        partials = torch.empty((num_chunks, 16, 512), dtype=torch.float32, device=q_nope.device)
        _output_partial_kernel[(num_chunks, 16, 8)](
            ckv_cache,
            kv_indptr,
            kv_indices,
            chunk_batch,
            chunk_start,
            logits,
            lse,
            partials,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            CHUNK_T=_PARTIAL_CHUNK_T,
            num_warps=8,
        )
        _output_reduce_kernel[(batch_size, 16, 8)](
            chunk_indptr,
            partials,
            output,
            BLOCK_D=_BLOCK_D,
            num_warps=1,
        )
    else:
        _output_kernel[(batch_size, 16, 8)](
            ckv_cache,
            kv_indptr,
            kv_indices,
            logits,
            lse,
            output,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            num_warps=8,
        )

    return {"output": output, "lse": lse}
