import torch
import triton
import triton.language as tl


@triton.jit
def _init_outputs(
    output_ptr,
    lse_ptr,
    n_output: tl.constexpr,
    n_lse: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output_ptr + offsets, 0.0, mask=offsets < n_output)
    tl.store(lse_ptr + offsets, -float("inf"), mask=offsets < n_lse)


@triton.jit
def _paged_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    NUM_BATCH: tl.constexpr,
    SEARCH_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    # One program computes one query head for one valid causal KV prefix.
    # Prefix position p in a sequence corresponds to q = q_len - kv_len + p;
    # negative q positions are precisely the fully masked query rows.
    global_kv = tl.program_id(0)
    head = tl.program_id(1)

    batch_offsets = tl.arange(0, SEARCH_BLOCK)
    batch_starts = tl.load(
        kv_indptr_ptr + batch_offsets,
        mask=batch_offsets < NUM_BATCH,
        other=0,
    )
    candidates = tl.where(
        (batch_offsets < NUM_BATCH) & (batch_starts <= global_kv),
        batch_offsets,
        -1,
    )
    batch = tl.max(candidates, axis=0)

    kv_start = tl.load(kv_indptr_ptr + batch)
    kv_end = tl.load(kv_indptr_ptr + batch + 1)
    q_start = tl.load(qo_indptr_ptr + batch)
    q_end = tl.load(qo_indptr_ptr + batch + 1)
    prefix_end = global_kv - kv_start
    q_local = (q_end - q_start) - (kv_end - kv_start) + prefix_end

    # q_local < 0 has no causal keys and retains the initialized zero/-inf.
    if q_local >= 0:
        query = q_start + q_local
        kv_head = head // 4
        dim = tl.arange(0, 128)
        q = tl.load(q_ptr + (query * 32 + head) * 128 + dim).to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((128,), tl.float32)

        # Online softmax over all keys through this causal prefix.
        for key_base in range(0, prefix_end + 1, KV_BLOCK):
            key_local = key_base + tl.arange(0, KV_BLOCK)
            key_mask = key_local <= prefix_end
            page_index = tl.load(
                kv_indices_ptr + kv_start + key_local,
                mask=key_mask,
                other=0,
            )
            cache_offsets = (page_index[:, None] * 8 + kv_head) * 128 + dim[None, :]
            keys = tl.load(k_ptr + cache_offsets, mask=key_mask[:, None], other=0.0).to(tl.float32)
            logits = tl.sum(keys * q[None, :], axis=1) * sm_scale
            logits = tl.where(key_mask, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            weights = tl.exp(logits - new_max)
            values = tl.load(v_ptr + cache_offsets, mask=key_mask[:, None], other=0.0).to(tl.float32)
            accumulator = accumulator * old_scale + tl.sum(weights[:, None] * values, axis=0)
            running_sum = running_sum * old_scale + tl.sum(weights, axis=0)
            running_max = new_max

        result = accumulator / running_sum
        output_offsets = (query * 32 + head) * 128 + dim
        tl.store(output_ptr + output_offsets, result)
        tl.store(
            lse_ptr + query * 32 + head,
            running_max * 1.4426950408889634 + tl.log2(running_sum),
        )


@triton.jit
def _paged_attention_blocked(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    CHUNK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    USE_BF16: tl.constexpr,
):
    batch = tl.program_id(0)
    chunk_id = tl.program_id(1)
    head_group = tl.program_id(2)
    kv_head = head_group // (4 // GROUP)

    q_start = tl.load(qo_indptr_ptr + batch)
    q_end = tl.load(qo_indptr_ptr + batch + 1)
    kv_start = tl.load(kv_indptr_ptr + batch)
    kv_end = tl.load(kv_indptr_ptr + batch + 1)
    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    prefix_base = chunk_id * CHUNK

    if prefix_base < kv_len:
        # Flatten query positions and a subset of a GQA group into the MFMA M axis.
        rows = tl.arange(0, ROWS)
        query_in_chunk = rows // GROUP
        head_in_group = rows % GROUP
        prefixes = prefix_base + query_in_chunk
        q_local = q_len - kv_len + prefixes
        valid_rows = (prefixes < kv_len) & (q_local >= 0)
        queries = q_start + q_local
        heads = head_group * GROUP + head_in_group
        dim = tl.arange(0, 128)

        q_offsets = (queries[:, None] * 32 + heads[:, None]) * 128 + dim[None, :]
        q = tl.load(q_ptr + q_offsets, mask=valid_rows[:, None], other=0.0)
        if USE_BF16:
            q = q.to(tl.bfloat16)
        else:
            q = q.to(tl.float16)

        running_max = tl.full((ROWS,), -float("inf"), tl.float32)
        running_sum = tl.zeros((ROWS,), tl.float32)
        accumulator = tl.zeros((ROWS, 128), tl.float32)
        last_prefix = tl.minimum(prefix_base + CHUNK - 1, kv_len - 1)

        for key_base in range(0, last_prefix + 1, KV_BLOCK):
            key_local = key_base + tl.arange(0, KV_BLOCK)
            key_mask = key_local < kv_len
            page_index = tl.load(
                kv_indices_ptr + kv_start + key_local,
                mask=key_mask,
                other=0,
            )
            cache_offsets = (page_index[:, None] * 8 + kv_head) * 128 + dim[None, :]
            keys = tl.load(k_ptr + cache_offsets, mask=key_mask[:, None], other=0.0)
            if USE_BF16:
                keys = keys.to(tl.bfloat16)
            else:
                keys = keys.to(tl.float16)
            logits = tl.dot(q, tl.trans(keys)) * sm_scale
            attention_mask = (
                valid_rows[:, None]
                & key_mask[None, :]
                & (key_local[None, :] <= prefixes[:, None])
            )
            logits = tl.where(attention_mask, logits, -float("inf"))

            block_max = tl.max(logits, axis=1)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            weights = tl.exp(logits - new_max[:, None])
            accumulator *= old_scale[:, None]
            values = tl.load(v_ptr + cache_offsets, mask=key_mask[:, None], other=0.0)
            if USE_BF16:
                values = values.to(tl.bfloat16)
                dot_weights = weights.to(tl.bfloat16)
            else:
                values = values.to(tl.float16)
                dot_weights = weights.to(tl.float16)
            accumulator = tl.dot(dot_weights, values, accumulator)
            running_sum = running_sum * old_scale + tl.sum(weights, axis=1)
            running_max = new_max

        result = accumulator / running_sum[:, None]
        output_offsets = (queries[:, None] * 32 + heads[:, None]) * 128 + dim[None, :]
        tl.store(output_ptr + output_offsets, result, mask=valid_rows[:, None])
        tl.store(
            lse_ptr + queries * 32 + heads,
            running_max * 1.4426950408889634 + tl.log2(running_sum),
            mask=valid_rows,
        )


@triton.jit
def _fused_small_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    NUM_BATCH: tl.constexpr,
    SEARCH_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    kv_head = tl.program_id(1)

    batch_offsets = tl.arange(0, SEARCH_BLOCK)
    q_starts = tl.load(
        qo_indptr_ptr + batch_offsets,
        mask=batch_offsets < NUM_BATCH,
        other=0,
    )
    candidates = tl.where(
        (batch_offsets < NUM_BATCH) & (q_starts <= query),
        batch_offsets,
        -1,
    )
    batch = tl.max(candidates, axis=0)
    q_start = tl.load(qo_indptr_ptr + batch)
    q_end = tl.load(qo_indptr_ptr + batch + 1)
    kv_start = tl.load(kv_indptr_ptr + batch)
    kv_end = tl.load(kv_indptr_ptr + batch + 1)
    prefix = query - q_start + (kv_end - kv_start) - (q_end - q_start)

    rows = tl.arange(0, 16)
    valid_rows = rows < 4
    heads = kv_head * 4 + rows
    dim = tl.arange(0, 128)
    output_offsets = (query * 32 + heads[:, None]) * 128 + dim[None, :]
    tl.store(output_ptr + output_offsets, 0.0, mask=valid_rows[:, None])
    tl.store(lse_ptr + query * 32 + heads, -float("inf"), mask=valid_rows)

    if prefix >= 0:
        q = tl.load(q_ptr + output_offsets, mask=valid_rows[:, None], other=0.0).to(tl.bfloat16)
        running_max = tl.full((16,), -float("inf"), tl.float32)
        running_sum = tl.zeros((16,), tl.float32)
        accumulator = tl.zeros((16, 128), tl.float32)

        for key_base in range(0, prefix + 1, KV_BLOCK):
            key_local = key_base + tl.arange(0, KV_BLOCK)
            key_mask = key_local <= prefix
            page_index = tl.load(
                kv_indices_ptr + kv_start + key_local,
                mask=key_mask,
                other=0,
            )
            cache_offsets = (page_index[:, None] * 8 + kv_head) * 128 + dim[None, :]
            keys = tl.load(k_ptr + cache_offsets, mask=key_mask[:, None], other=0.0).to(tl.bfloat16)
            logits = tl.dot(q, tl.trans(keys)) * sm_scale
            attention_mask = valid_rows[:, None] & key_mask[None, :]
            logits = tl.where(attention_mask, logits, -float("inf"))
            block_max = tl.max(logits, axis=1)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            weights = tl.exp(logits - new_max[:, None])
            accumulator *= old_scale[:, None]
            values = tl.load(v_ptr + cache_offsets, mask=key_mask[:, None], other=0.0).to(tl.bfloat16)
            accumulator = tl.dot(weights.to(tl.bfloat16), values, accumulator)
            running_sum = running_sum * old_scale + tl.sum(weights, axis=1)
            running_max = new_max

        tl.store(
            output_ptr + output_offsets,
            accumulator / running_sum[:, None],
            mask=valid_rows[:, None],
        )
        tl.store(
            lse_ptr + query * 32 + heads,
            running_max * 1.4426950408889634 + tl.log2(running_sum),
            mask=valid_rows,
        )


@triton.jit
def _single_page_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    LAST_QUERY: tl.constexpr,
):
    query = tl.program_id(0)
    heads = tl.arange(0, 32)
    dim = tl.arange(0, 128)
    offsets = (query * 32 + heads[:, None]) * 128 + dim[None, :]

    if query == LAST_QUERY:
        page = tl.load(kv_indices_ptr)
        cache_offsets = (page * 8 + (heads[:, None] // 4)) * 128 + dim[None, :]
        q = tl.load(q_ptr + offsets).to(tl.float32)
        k = tl.load(k_ptr + cache_offsets).to(tl.float32)
        v = tl.load(v_ptr + cache_offsets)
        logits = tl.sum(q * k, axis=1) * sm_scale
        tl.store(output_ptr + offsets, v)
        tl.store(lse_ptr + query * 32 + heads, logits * 1.4426950408889634)
    else:
        tl.store(output_ptr + offsets, 0.0)
        tl.store(lse_ptr + query * 32 + heads, -float("inf"))


@triton.jit
def _few_page_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    TOTAL_QUERY: tl.constexpr,
    NUM_KV: tl.constexpr,
):
    query = tl.program_id(0)
    heads = tl.arange(0, 32)
    dim = tl.arange(0, 128)
    offsets = (query * 32 + heads[:, None]) * 128 + dim[None, :]
    prefix = query + NUM_KV - TOTAL_QUERY

    if prefix >= 0:
        q = tl.load(q_ptr + offsets).to(tl.float32)
        running_max = tl.full((32,), -float("inf"), tl.float32)
        running_sum = tl.zeros((32,), tl.float32)
        accumulator = tl.zeros((32, 128), tl.float32)

        for key_local in range(0, NUM_KV):
            page = tl.load(kv_indices_ptr + key_local)
            cache_offsets = (page * 8 + (heads[:, None] // 4)) * 128 + dim[None, :]
            key = tl.load(k_ptr + cache_offsets).to(tl.float32)
            logit = tl.sum(q * key, axis=1) * sm_scale
            logit = tl.where(key_local <= prefix, logit, -float("inf"))
            new_max = tl.maximum(running_max, logit)
            old_scale = tl.exp(running_max - new_max)
            weight = tl.exp(logit - new_max)
            value = tl.load(v_ptr + cache_offsets).to(tl.float32)
            accumulator = accumulator * old_scale[:, None] + weight[:, None] * value
            running_sum = running_sum * old_scale + weight
            running_max = new_max

        tl.store(output_ptr + offsets, accumulator / running_sum[:, None])
        tl.store(
            lse_ptr + query * 32 + heads,
            running_max * 1.4426950408889634 + tl.log2(running_sum),
        )
    else:
        tl.store(output_ptr + offsets, 0.0)
        tl.store(lse_ptr + query * 32 + heads, -float("inf"))


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q.shape[0]
    num_kv = kv_indices.numel()
    num_batch = qo_indptr.numel() - 1

    output = torch.empty_like(q)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

    if total_q == 1:
        search_block = triton.next_power_of_2(num_batch)
        _fused_small_attention[(total_q, 8)](
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            output,
            lse,
            sm_scale,
            NUM_BATCH=num_batch,
            SEARCH_BLOCK=search_block,
            KV_BLOCK=32,
            num_warps=4,
        )
        return output, lse

    if num_kv == 1 and num_batch == 1:
        _single_page_attention[(total_q,)](
            q,
            k_cache,
            v_cache,
            kv_indices,
            output,
            lse,
            sm_scale,
            LAST_QUERY=total_q - 1,
            num_warps=4,
        )
        return output, lse

    if num_batch == 1 and num_kv <= 4:
        _few_page_attention[(total_q,)](
            q,
            k_cache,
            v_cache,
            kv_indices,
            output,
            lse,
            sm_scale,
            TOTAL_QUERY=total_q,
            NUM_KV=num_kv,
            num_warps=8,
        )
        return output, lse

    n_output = output.numel()
    n_lse = lse.numel()
    _init_outputs[(triton.cdiv(n_output, 4096),)](
        output,
        lse,
        n_output=n_output,
        n_lse=n_lse,
        BLOCK=4096,
        num_warps=8,
    )

    if num_kv:
        if num_kv <= 256:
            max_chunks = triton.cdiv(num_kv, 16)
            _paged_attention_blocked[(num_batch, max_chunks, 8)](
                q,
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                output,
                lse,
                sm_scale,
                CHUNK=16,
                KV_BLOCK=16,
                GROUP=4,
                ROWS=64,
                USE_BF16=False,
                num_warps=8,
            )
        else:
            max_chunks = triton.cdiv(num_kv, 32)
            _paged_attention_blocked[(num_batch, max_chunks, 8)](
                q,
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                output,
                lse,
                sm_scale,
                CHUNK=32,
                KV_BLOCK=32,
                GROUP=4,
                ROWS=128,
                USE_BF16=True,
                num_warps=4,
            )

    return output, lse
