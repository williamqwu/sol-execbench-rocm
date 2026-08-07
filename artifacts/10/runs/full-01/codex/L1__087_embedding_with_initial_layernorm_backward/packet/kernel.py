import torch
import triton
import triton.language as tl


VOCAB = 65536
HIDDEN = 4096


@triton.jit
def _zero_output_meta_kernel(
    out, counts, leaders, norm_acc,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + off, 0.0, mask=off < n_elements)
    tl.store(counts + off, 0, mask=off < 65536)
    tl.store(leaders + off, 2147483647, mask=off < 65536)
    tl.store(norm_acc + off, 0.0, mask=off < 4096)


@triton.jit
def _histogram_kernel(ids, counts, leaders, n_tokens: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_tokens
    token = tl.load(ids + off, mask=mask, other=0).to(tl.int32)
    tl.atomic_add(counts + token, 1, mask=mask)
    tl.atomic_min(leaders + token, off, mask=mask)


@triton.jit
def _zero_collision_rows_kernel(
    ids, counts, leaders, collision_acc,
    n_tokens: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token = tl.load(ids + token_idx).to(tl.int32)
    count = tl.load(counts + token)
    leader = tl.load(leaders + token)
    h = tl.arange(0, BLOCK_H)
    active = (token_idx == leader) & (count > 1) & (h < 4096)
    tl.store(collision_acc + token_idx * 4096 + h, 0.0, mask=active)


@triton.jit
def _scatter_kernel(
    grad_output, ids, hidden_states, rstd, norm_weight,
    counts, leaders, collision_acc, out,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask_h = h < 4096
    base = token_idx * 4096 + h

    go = tl.load(grad_output + base, mask=mask_h, other=0.0).to(tl.float32)
    x = tl.load(hidden_states + base, mask=mask_h, other=0.0).to(tl.float32)
    weight = tl.load(norm_weight + h, mask=mask_h, other=0.0).to(tl.float32)
    rs = tl.load(rstd + token_idx).to(tl.float32)

    normalized = x * rs
    grad_normalized = go * weight
    dot = tl.sum(grad_normalized * normalized, axis=0)
    mean = dot * (1.0 / 4096)
    grad = rs * (grad_normalized - mean * normalized)

    token = tl.load(ids + token_idx).to(tl.int32)
    count = tl.load(counts + token)
    leader = tl.load(leaders + token)
    unique = count == 1
    tl.store(out + token * 4096 + h, grad, mask=mask_h & unique)
    tl.atomic_add(
        collision_acc + leader * 4096 + h,
        grad,
        mask=mask_h & (count > 1),
    )


@triton.jit
def _finish_collision_rows_kernel(
    ids, counts, leaders, collision_acc, out,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token = tl.load(ids + token_idx).to(tl.int32)
    count = tl.load(counts + token)
    leader = tl.load(leaders + token)
    h = tl.arange(0, BLOCK_H)
    active = (token_idx == leader) & (count > 1) & (h < 4096)
    value = tl.load(collision_acc + token_idx * 4096 + h, mask=active, other=0.0)
    tl.store(out + token * 4096 + h, value, mask=active)


@triton.jit
def _norm_weight_kernel(
    grad_output, hidden_states, rstd, norm_acc,
    n_tokens: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (n[:, None] < n_tokens) & (h[None, :] < 4096)
    off = n[:, None] * 4096 + h[None, :]
    go = tl.load(grad_output + off, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(hidden_states + off, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(rstd + n, mask=n < n_tokens, other=0.0).to(tl.float32)
    normalized = x * rs[:, None]
    partial = tl.sum(go * normalized, axis=0)
    tl.atomic_add(norm_acc + h, partial, mask=h < 4096)


@triton.jit
def _finish_norm_kernel(norm_acc, norm_out, BLOCK: tl.constexpr):
    h = tl.arange(0, BLOCK)
    value = tl.load(norm_acc + h, mask=h < 4096, other=0.0)
    tl.store(norm_out + h, value, mask=h < 4096)


@triton.jit
def _zero_output_state_kernel(
    out, leaders, next_idx, norm_acc,
    n_elements: tl.constexpr,
    n_next: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + off, 0.0, mask=off < n_elements)
    tl.store(leaders + off, -1, mask=off < 65536)
    tl.store(leaders + 65536 + off, 0, mask=off < n_next)
    tl.store(norm_acc + off, 0.0, mask=off < 4096)


@triton.jit
def _init_state_kernel(
    leaders, norm_acc,
    n_next: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(leaders + off, -1, mask=off < 65536)
    tl.store(leaders + 65536 + off, 0, mask=off < n_next)
    tl.store(norm_acc + off, 0.0, mask=off < 4096)


@triton.jit
def _build_lists_kernel(
    ids, leaders, next_idx,
    n_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = off < n_tokens
    token = tl.load(ids + off, mask=valid, other=0).to(tl.int32)
    observed = tl.load(leaders + token)
    pending = valid
    saved = tl.full((BLOCK,), -1, tl.int32)

    while tl.max(pending.to(tl.int32), axis=0) != 0:
        # atomic_cas has no mask argument on this Triton/ROCm build. Completed
        # and padding lanes perform a harmless 0 -> 0 CAS in their own slots.
        ptr = leaders + tl.where(pending, token, 65536 + off)
        expected = tl.where(pending, observed, 0)
        desired = tl.where(pending, off, 0)
        previous = tl.atomic_cas(ptr, expected, desired)
        success = pending & (previous == observed)
        saved = tl.where(success, observed, saved)
        observed = tl.where(pending & ~success, previous, observed)
        pending = pending & ~success

    tl.store(next_idx + off, saved, mask=valid)


@triton.jit
def _token_mean_kernel(
    grad_output, hidden_states, rstd, norm_weight, means,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < 4096
    off = token_idx * 4096 + h
    go = tl.load(grad_output + off, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(hidden_states + off, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(norm_weight + h, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(rstd + token_idx).to(tl.float32)
    normalized = x * rs
    grad_normalized = go * weight
    mean = tl.sum(grad_normalized * normalized, axis=0) * (1.0 / 4096)
    tl.store(means + token_idx, mean)


@triton.jit
def _linked_scatter_kernel(
    grad_output, ids, hidden_states, rstd, norm_weight, means,
    leaders, next_idx, out,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token = tl.load(ids + token_idx).to(tl.int32)
    head = tl.load(leaders + token)
    is_head = head == token_idx
    current = tl.where(is_head, token_idx, -1)
    h = tl.arange(0, BLOCK_H)
    mask_h = h < 4096
    total = tl.zeros((BLOCK_H,), dtype=tl.float32)

    while current >= 0:
        off = current * 4096 + h
        go = tl.load(grad_output + off, mask=mask_h, other=0.0).to(tl.float32)
        x = tl.load(hidden_states + off, mask=mask_h, other=0.0).to(tl.float32)
        weight = tl.load(norm_weight + h, mask=mask_h, other=0.0).to(tl.float32)
        rs = tl.load(rstd + current).to(tl.float32)
        mean = tl.load(means + current).to(tl.float32)
        normalized = x * rs
        grad_normalized = go * weight
        grad = rs * (grad_normalized - mean * normalized)
        total += grad
        current = tl.load(next_idx + current)

    tl.store(out + token * 4096 + h, total, mask=mask_h & is_head)


@triton.jit
def _means_and_norm_kernel(
    grad_output, hidden_states, rstd, norm_weight, means, norm_acc,
    n_tokens: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = n < n_tokens
    mask = valid_n[:, None] & (h[None, :] < 4096)
    off = n[:, None] * 4096 + h[None, :]
    go = tl.load(grad_output + off, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(hidden_states + off, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(rstd + n, mask=valid_n, other=0.0).to(tl.float32)
    weight = tl.load(norm_weight + h, mask=h < 4096, other=0.0).to(tl.float32)
    normalized = x * rs[:, None]
    value = go * normalized
    mean_partial = tl.sum(value * weight[None, :], axis=1) * (1.0 / 4096)
    norm_partial = tl.sum(value, axis=0)
    tl.atomic_add(means + n, mean_partial, mask=valid_n)
    tl.atomic_add(norm_acc + h, norm_partial, mask=h < 4096)


@triton.jit
def _norm_weight_loop_kernel(
    grad_output, hidden_states, rstd, norm_acc,
    n_tokens: tl.constexpr,
    CHUNK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    start_n = pid_n * CHUNK_N
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.range(0, CHUNK_N):
        n = start_n + k
        valid = n < n_tokens
        off = n * 4096 + h
        go = tl.load(grad_output + off, mask=valid & (h < 4096), other=0.0).to(tl.float32)
        x = tl.load(hidden_states + off, mask=valid & (h < 4096), other=0.0).to(tl.float32)
        rs = tl.load(rstd + n, mask=valid, other=0.0).to(tl.float32)
        normalized = x * rs
        acc += go * normalized
    tl.atomic_add(norm_acc + h, acc, mask=h < 4096)


@triton.jit
def _mean_norm_stream_kernel(
    grad_output, hidden_states, rstd, norm_weight, means, norm_acc,
    n_tokens: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = n < n_tokens
    rs = tl.load(rstd + n, mask=valid_n, other=0.0).to(tl.float32)
    mean_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h_start in tl.range(0, 4096, BLOCK_H):
        h = h_start + tl.arange(0, BLOCK_H)
        mask = valid_n[:, None] & (h[None, :] < 4096)
        off = n[:, None] * 4096 + h[None, :]
        go = tl.load(grad_output + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(hidden_states + off, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(norm_weight + h, mask=h < 4096, other=0.0).to(tl.float32)
        normalized = x * rs[:, None]
        value = go * normalized
        mean_acc += tl.sum(value * weight[None, :], axis=1) * (1.0 / 4096)
        norm_partial = tl.sum(value, axis=0)
        tl.atomic_add(norm_acc + h, norm_partial, mask=h < 4096)

    tl.store(means + n, mean_acc, mask=valid_n)


@triton.jit
def _linked_fused_scatter_kernel(
    grad_output, ids, hidden_states, rstd, norm_weight,
    leaders, next_idx, out,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token = tl.load(ids + token_idx).to(tl.int32)
    head = tl.load(leaders + token)
    is_head = head == token_idx
    current = tl.where(is_head, token_idx, -1)
    h = tl.arange(0, BLOCK_H)
    mask_h = h < 4096
    total = tl.zeros((BLOCK_H,), dtype=tl.float32)

    while current >= 0:
        off = current * 4096 + h
        go = tl.load(grad_output + off, mask=mask_h, other=0.0).to(tl.float32)
        x = tl.load(hidden_states + off, mask=mask_h, other=0.0).to(tl.float32)
        weight = tl.load(norm_weight + h, mask=mask_h, other=0.0).to(tl.float32)
        rs = tl.load(rstd + current).to(tl.float32)
        normalized = x * rs
        grad_normalized = go * weight
        mean = tl.sum(grad_normalized * normalized, axis=0) * (1.0 / 4096)
        grad = rs * (grad_normalized - mean * normalized)
        total += grad
        current = tl.load(next_idx + current)

    tl.store(out + token * 4096 + h, total, mask=mask_h & is_head)


@triton.jit
def _vocab_fused_scatter_kernel(
    grad_output, hidden_states, rstd, norm_weight,
    leaders, next_idx, out,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    current = tl.load(leaders + token)
    h = tl.arange(0, BLOCK_H)
    mask_h = h < 4096
    total = tl.zeros((BLOCK_H,), dtype=tl.float32)

    while current >= 0:
        off = current * 4096 + h
        go = tl.load(grad_output + off, mask=mask_h, other=0.0).to(tl.float32)
        x = tl.load(hidden_states + off, mask=mask_h, other=0.0).to(tl.float32)
        weight = tl.load(norm_weight + h, mask=mask_h, other=0.0).to(tl.float32)
        rs = tl.load(rstd + current).to(tl.float32)
        normalized = x * rs
        grad_normalized = go * weight
        mean = tl.sum(grad_normalized * normalized, axis=0) * (1.0 / 4096)
        grad = rs * (grad_normalized - mean * normalized)
        total += grad
        current = tl.load(next_idx + current)

    tl.store(out + token * 4096 + h, total, mask=mask_h)


@torch.no_grad()
def run(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight):
    n_tokens = input_ids.numel()
    device = grad_output.device
    dense_vocab_path = n_tokens >= 10000
    if dense_vocab_path:
        out = torch.empty((VOCAB, HIDDEN), dtype=torch.bfloat16, device=device)
    else:
        out = torch.zeros((VOCAB, HIDDEN), dtype=torch.bfloat16, device=device)
    grad_norm = torch.empty((HIDDEN,), dtype=torch.bfloat16, device=device)
    list_block = 256
    padded_tokens = triton.cdiv(n_tokens, list_block) * list_block
    leaders = torch.empty((VOCAB + padded_tokens,), dtype=torch.int32, device=device)
    next_idx = torch.empty((padded_tokens,), dtype=torch.int32, device=device)
    norm_acc = torch.empty((HIDDEN,), dtype=torch.float32, device=device)

    state_block = 65536
    _init_state_kernel[(triton.cdiv(VOCAB + padded_tokens, state_block),)](
        leaders, norm_acc, padded_tokens,
        BLOCK=state_block,
        num_warps=8,
    )
    _build_lists_kernel[(triton.cdiv(n_tokens, list_block),)](
        input_ids, leaders, next_idx, n_tokens,
        BLOCK=list_block, num_warps=4,
    )
    if dense_vocab_path:
        _vocab_fused_scatter_kernel[(VOCAB,)](
            grad_output, hidden_states_fp32, rstd, norm_weight,
            leaders, next_idx, out,
            BLOCK_H=HIDDEN, num_warps=8,
        )
    else:
        _linked_fused_scatter_kernel[(n_tokens,)](
            grad_output, input_ids, hidden_states_fp32, rstd, norm_weight,
            leaders, next_idx, out,
            BLOCK_H=HIDDEN, num_warps=8,
        )

    if n_tokens >= 10000:
        target_chunk = max(8, n_tokens // 32)
        chunk_n = min(1024, 1 << (target_chunk.bit_length() - 1))
        block_h = 256
        _norm_weight_loop_kernel[(triton.cdiv(HIDDEN, block_h), triton.cdiv(n_tokens, chunk_n))](
            grad_output, hidden_states_fp32, rstd, norm_acc, n_tokens,
            CHUNK_N=chunk_n, BLOCK_H=block_h,
            num_warps=4, num_stages=2,
        )
    else:
        block_n = min(512, max(32, triton.next_power_of_2(triton.cdiv(n_tokens, 8))))
        block_h = 128
        _norm_weight_kernel[(triton.cdiv(HIDDEN, block_h), triton.cdiv(n_tokens, block_n))](
            grad_output, hidden_states_fp32, rstd, norm_acc, n_tokens,
            BLOCK_N=block_n, BLOCK_H=block_h,
            num_warps=4 if block_n <= 128 else 8,
        )
    _finish_norm_kernel[(1,)](norm_acc, grad_norm, BLOCK=HIDDEN, num_warps=8)
    return out, grad_norm
