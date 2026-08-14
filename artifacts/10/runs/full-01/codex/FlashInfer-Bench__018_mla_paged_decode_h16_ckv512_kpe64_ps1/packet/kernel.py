import torch
import triton
import triton.language as tl


HEADS = 16
DIM = 512
ROPE_DIM = 64
_HEADS = tl.constexpr(16)
_DIM = tl.constexpr(512)
_ROPE_DIM = tl.constexpr(64)
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _mla_stage1(
    q_ptr,
    qr_ptr,
    kc_ptr,
    kr_ptr,
    indptr_ptr,
    indices_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    num_splits,
    SINGLE_PASS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    split = tl.program_id(1)

    begin = tl.load(indptr_ptr + batch)
    end = tl.load(indptr_ptr + batch + 1)
    length = end - begin
    split_len = (length + num_splits - 1) // num_splits
    split_begin = begin + split * split_len
    split_end = tl.minimum(split_begin + split_len, end)

    offs_h = tl.arange(0, _HEADS)
    offs_d = tl.arange(0, _DIM)
    offs_r = tl.arange(0, _ROPE_DIM)
    q = tl.load(
        q_ptr
        + batch * (_HEADS * _DIM)
        + offs_h[:, None] * _DIM
        + offs_d[None, :]
    )
    qr = tl.load(
        qr_ptr
        + batch * (_HEADS * _ROPE_DIM)
        + offs_h[:, None] * _ROPE_DIM
        + offs_r[None, :]
    )

    acc = tl.zeros((_HEADS, _DIM), dtype=tl.float32)
    denom = tl.zeros((_HEADS,), dtype=tl.float32)
    max_logit = tl.full((_HEADS,), -float("inf"), dtype=tl.float32)

    for token_start in range(split_begin, split_end, BLOCK_N):
        offs_n = token_start + tl.arange(0, BLOCK_N)
        token_mask = offs_n < split_end
        pages = tl.load(indices_ptr + offs_n, mask=token_mask, other=0)
        kc = tl.load(
            kc_ptr + pages[:, None] * _DIM + offs_d[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        kr = tl.load(
            kr_ptr + pages[:, None] * _ROPE_DIM + offs_r[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )

        logits = tl.dot(q, tl.trans(kc), out_dtype=tl.float32)
        logits += tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
        logits *= sm_scale * _LOG2E
        logits = tl.where(token_mask[None, :], logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(max_logit, block_max)
        alpha = tl.exp2(max_logit - new_max)
        probs = tl.exp2(logits - new_max[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(probs.to(tl.bfloat16), kc, out_dtype=tl.float32)
        denom = denom * alpha + tl.sum(probs, axis=1)
        max_logit = new_max

    valid = denom > 0.0
    normalized = tl.where(valid[:, None], acc / denom[:, None], 0.0)
    split_lse = tl.where(valid, max_logit + tl.log2(denom), -float("inf"))

    if SINGLE_PASS:
        tl.store(
            output_ptr
            + batch * (_HEADS * _DIM)
            + offs_h[:, None] * _DIM
            + offs_d[None, :],
            normalized,
        )
        tl.store(lse_ptr + batch * _HEADS + offs_h, split_lse)
    else:
        out_base = (batch * num_splits + split) * (_HEADS * _DIM)
        lse_base = (batch * num_splits + split) * _HEADS
        tl.store(
            partial_out_ptr
            + out_base
            + offs_h[:, None] * _DIM
            + offs_d[None, :],
            normalized,
        )
        tl.store(partial_lse_ptr + lse_base + offs_h, split_lse)


@triton.jit
def _mla_reduce(partial_out_ptr, partial_lse_ptr, output_ptr, lse_ptr, num_splits):
    batch = tl.program_id(0)
    offs_h = tl.arange(0, _HEADS)
    offs_d = tl.arange(0, _DIM)

    acc = tl.zeros((_HEADS, _DIM), dtype=tl.float32)
    denom = tl.zeros((_HEADS,), dtype=tl.float32)
    max_lse = tl.full((_HEADS,), -float("inf"), dtype=tl.float32)

    for split in range(0, num_splits):
        out_base = (batch * num_splits + split) * (_HEADS * _DIM)
        lse_base = (batch * num_splits + split) * _HEADS
        split_lse = tl.load(partial_lse_ptr + lse_base + offs_h)
        partial = tl.load(
            partial_out_ptr
            + out_base
            + offs_h[:, None] * _DIM
            + offs_d[None, :]
        )
        split_valid = split_lse != -float("inf")
        old_valid = max_lse != -float("inf")
        new_max = tl.maximum(max_lse, split_lse)
        alpha = tl.where(old_valid, tl.exp2(max_lse - new_max), 0.0)
        beta = tl.where(split_valid, tl.exp2(split_lse - new_max), 0.0)
        acc = acc * alpha[:, None] + partial * beta[:, None]
        denom = denom * alpha + beta
        max_lse = new_max

    valid = denom > 0.0
    result = tl.where(valid[:, None], acc / denom[:, None], 0.0)
    final_lse = tl.where(valid, max_lse + tl.log2(denom), -float("inf"))
    tl.store(
        output_ptr
        + batch * (_HEADS * _DIM)
        + offs_h[:, None] * _DIM
        + offs_d[None, :],
        result,
    )
    tl.store(lse_ptr + batch * _HEADS + offs_h, final_lse)


def _dispatch(batch_size, total_tokens):
    if batch_size == 1:
        if total_tokens <= 32:
            return 1, 32, 4
        if total_tokens <= 256:
            return 1, 64, 4
        if total_tokens <= 768:
            return 8, 64, 4
        return 16, 64, 4
    if batch_size == 16:
        if total_tokens <= 1024:
            return 1, 64, 4
        if total_tokens <= 8192:
            return 8, 32, 8
        return 8, 64, 8
    if 12000 < total_tokens < 20000:
        return 4, 32, 8
    if total_tokens >= 47000:
        return 8, 64, 4
    return 4, 64, 8


def _use_fp32_partials(batch_size, total_tokens):
    if batch_size == 16:
        return 10000 < total_tokens < 12000 or total_tokens >= 14000
    if batch_size == 64:
        return 12000 <= total_tokens < 44000
    return False


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size = q_nope.shape[0]
    total_tokens = kv_indices.shape[0]
    num_splits, block_n, num_warps = _dispatch(
        batch_size, total_tokens
    )
    fp32_partials = _use_fp32_partials(batch_size, total_tokens)
    output_elements = batch_size * HEADS * DIM
    lse_elements = batch_size * HEADS
    partial_out_elements = (
        0 if num_splits == 1 else batch_size * num_splits * HEADS * DIM
    )
    partial_lse_elements = (
        0 if num_splits == 1 else batch_size * num_splits * HEADS
    )
    partial_storage_elements = partial_out_elements * (2 if fp32_partials else 1)
    storage = torch.empty(
        output_elements
        + partial_storage_elements
        + 2 * (lse_elements + partial_lse_elements),
        dtype=torch.bfloat16,
        device=q_nope.device,
    )
    output = storage[:output_elements].view(batch_size, HEADS, DIM)
    partial_storage = storage[
        output_elements : output_elements + partial_storage_elements
    ]
    float_storage = storage[
        output_elements + partial_storage_elements :
    ].view(torch.float32)
    lse = float_storage[:lse_elements].view(batch_size, HEADS)
    if num_splits == 1:
        _mla_stage1[(batch_size, 1)](
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            kv_indptr,
            kv_indices,
            output,
            lse,
            output,
            lse,
            sm_scale,
            1,
            SINGLE_PASS=True,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        partial_out = (
            partial_storage.view(torch.float32)
            if fp32_partials
            else partial_storage
        ).view(
            batch_size, num_splits, HEADS, DIM
        )
        partial_lse = float_storage[lse_elements:].view(
            batch_size, num_splits, HEADS
        )
        _mla_stage1[(batch_size, num_splits)](
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            kv_indptr,
            kv_indices,
            partial_out,
            partial_lse,
            output,
            lse,
            sm_scale,
            num_splits,
            SINGLE_PASS=False,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )
        _mla_reduce[(batch_size,)](
            partial_out,
            partial_lse,
            output,
            lse,
            num_splits,
            num_warps=num_warps,
            num_stages=1,
        )
    return {"output": output, "lse": lse}
