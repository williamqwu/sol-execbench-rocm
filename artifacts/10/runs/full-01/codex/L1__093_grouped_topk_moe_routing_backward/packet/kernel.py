import torch
import triton
import triton.language as tl
import aiter


# hipBLASLt exposes many valid kernels for these skinny projections.  PyTorch's
# heuristic is excellent for most sizes, while these measured choices avoid
# its performance cliffs.  The handle contains no problem data or outputs.
aiter.hipb_create_extension()

_GRAD_WEIGHT_SOLUTIONS = {
    1280: 435315,
    1344: 435348,
    2048: 435212,
    2131: 435212,
    2521: 435348,
    3011: 435212,
    3557: 435212,
    8192: 435245,
}

_GRAD_HIDDEN_SOLUTIONS = {
    1280: 436636,
    1312: 436636,
    1321: 436636,
    1376: 436636,
    1408: 436636,
    1440: 436636,
    1721: 436331,
    2048: 436331,
    2131: 436331,
    2521: 436636,
    3011: 436535,
    3072: 436535,
    3557: 436725,
    6144: 436535,
    8192: 437319,
}


@triton.jit
def _debug_stages(grad_ptr, normalized_ptr, denominator_ptr, out_ptr, scale):
    row = tl.program_id(0)
    k = tl.arange(0, 8)
    grad = tl.load(grad_ptr + row * 8 + k)
    grad_n = (grad.to(tl.float32) * scale).to(tl.bfloat16)
    normalized = tl.load(normalized_ptr + row * 8 + k)
    product = (grad_n.to(tl.float32) * normalized.to(tl.float32)).to(tl.bfloat16)
    grad_sum = tl.sum(product.to(tl.float32), axis=0).to(tl.bfloat16)
    denominator = tl.load(denominator_ptr + row)
    numerator = (
        grad_n.to(tl.float32) - grad_sum.to(tl.float32)
    ).to(tl.bfloat16)
    grad_u = tl.div_rn(
        numerator.to(tl.float32), denominator.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(out_ptr + row * 25 + k, grad_n)
    tl.store(out_ptr + row * 25 + 8 + k, product)
    tl.store(out_ptr + row * 25 + 16, grad_sum)
    tl.store(out_ptr + row * 25 + 17 + k, grad_u)


@triton.jit
def _make_grad_logits(
    grad_ptr,
    normalized_ptr,
    denominator_ptr,
    scores_ptr,
    indices_ptr,
    output_ptr,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, 8)

    # Each explicit conversion to bf16 is an observable rounding point in the
    # eager reference.  Triton otherwise keeps these expressions in fp32.
    grad = tl.load(grad_ptr + row * 8 + k)
    grad_n = (grad.to(tl.float32) * scale).to(tl.bfloat16)
    normalized = tl.load(normalized_ptr + row * 8 + k)
    product = (
        grad_n.to(tl.float32) * normalized.to(tl.float32)
    ).to(tl.bfloat16)
    grad_sum = tl.sum(product.to(tl.float32), axis=0).to(tl.bfloat16)
    denominator = tl.load(denominator_ptr + row)
    numerator = (
        grad_n.to(tl.float32) - grad_sum.to(tl.float32)
    ).to(tl.bfloat16)
    grad_u = tl.div_rn(
        numerator.to(tl.float32),
        denominator.to(tl.float32),
    ).to(tl.bfloat16)

    columns = tl.arange(0, BLOCK)
    valid = columns < 160
    indices = tl.load(indices_ptr + row * 8 + k)
    selected = columns[:, None] == indices[None, :]
    grad_score = tl.sum(
        tl.where(selected, grad_u[None, :].to(tl.float32), 0.0), axis=1
    ).to(tl.bfloat16)

    score = tl.load(scores_ptr + row * 160 + columns, mask=valid, other=0.0)
    one_minus_score = (1.0 - score.to(tl.float32)).to(tl.bfloat16)
    first_product = (
        grad_score.to(tl.float32) * score.to(tl.float32)
    ).to(tl.bfloat16)
    result = (
        first_product.to(tl.float32) * one_minus_score.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(output_ptr + row * 160 + columns, result, mask=valid)


@triton.jit
def _make_grad_logits_batched(
    grad_ptr,
    normalized_ptr,
    denominator_ptr,
    scores_ptr,
    indices_ptr,
    output_ptr,
    scale,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, 8)
    row_mask = rows < n_rows
    grad = tl.load(
        grad_ptr + rows[:, None] * 8 + k[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    grad_n = (grad.to(tl.float32) * scale).to(tl.bfloat16)
    normalized = tl.load(
        normalized_ptr + rows[:, None] * 8 + k[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    product = (
        grad_n.to(tl.float32) * normalized.to(tl.float32)
    ).to(tl.bfloat16)
    grad_sum = tl.sum(product.to(tl.float32), axis=1).to(tl.bfloat16)
    denominator = tl.load(denominator_ptr + rows, mask=row_mask, other=1.0)
    numerator = (
        grad_n.to(tl.float32) - grad_sum[:, None].to(tl.float32)
    ).to(tl.bfloat16)
    grad_u = tl.div_rn(
        numerator.to(tl.float32), denominator[:, None].to(tl.float32)
    ).to(tl.bfloat16)

    columns = tl.arange(0, BLOCK_E)
    valid = row_mask[:, None] & (columns[None, :] < 160)
    grad_score = tl.zeros((BLOCK_M, BLOCK_E), tl.float32)
    for ki in tl.static_range(0, 8):
        index = tl.load(
            indices_ptr + rows * 8 + ki, mask=row_mask, other=0
        )
        grad_ki = tl.sum(
            tl.where(
                k[None, :] == ki, grad_u.to(tl.float32), 0.0
            ),
            axis=1,
        )
        grad_score += tl.where(
            columns[None, :] == index[:, None],
            grad_ki[:, None],
            0.0,
        )
    grad_score = grad_score.to(tl.bfloat16)
    score = tl.load(
        scores_ptr + rows[:, None] * 160 + columns[None, :],
        mask=valid,
        other=0.0,
    )
    one_minus_score = (1.0 - score.to(tl.float32)).to(tl.bfloat16)
    first_product = (
        grad_score.to(tl.float32) * score.to(tl.float32)
    ).to(tl.bfloat16)
    result = (
        first_product.to(tl.float32) * one_minus_score.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + rows[:, None] * 160 + columns[None, :], result, mask=valid
    )


@triton.jit
def _make_grad_logits_sparse_store(
    grad_ptr,
    normalized_ptr,
    denominator_ptr,
    scores_ptr,
    indices_ptr,
    output_ptr,
    scale,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, 8)
    row_mask = rows < n_rows
    mask8 = row_mask[:, None]
    offsets8 = rows[:, None] * 8 + k[None, :]

    grad = tl.load(grad_ptr + offsets8, mask=mask8, other=0.0)
    grad_n = (grad.to(tl.float32) * scale).to(tl.bfloat16)
    normalized = tl.load(normalized_ptr + offsets8, mask=mask8, other=0.0)
    product = (
        grad_n.to(tl.float32) * normalized.to(tl.float32)
    ).to(tl.bfloat16)
    grad_sum = tl.sum(product.to(tl.float32), axis=1).to(tl.bfloat16)
    denominator = tl.load(denominator_ptr + rows, mask=row_mask, other=1.0)
    numerator = (
        grad_n.to(tl.float32) - grad_sum[:, None].to(tl.float32)
    ).to(tl.bfloat16)
    grad_u = tl.div_rn(
        numerator.to(tl.float32), denominator[:, None].to(tl.float32)
    ).to(tl.bfloat16)

    indices = tl.load(indices_ptr + offsets8, mask=mask8, other=0)
    selected_scores = tl.load(
        scores_ptr + rows[:, None] * 160 + indices,
        mask=mask8,
        other=0.0,
    )
    one_minus_score = (
        1.0 - selected_scores.to(tl.float32)
    ).to(tl.bfloat16)
    first_product = (
        grad_u.to(tl.float32) * selected_scores.to(tl.float32)
    ).to(tl.bfloat16)
    selected_result = (
        first_product.to(tl.float32) * one_minus_score.to(tl.float32)
    ).to(tl.bfloat16)

    columns = tl.arange(0, BLOCK_E)
    dense_mask = row_mask[:, None] & (columns[None, :] < 160)
    tl.store(
        output_ptr + rows[:, None] * 160 + columns[None, :],
        0.0,
        mask=dense_mask,
    )
    # All rows owned by this program are private.  The barrier makes the
    # zero-fill globally visible before the eight sparse overwrites.
    tl.debug_barrier()
    tl.store(
        output_ptr + rows[:, None] * 160 + indices,
        selected_result,
        mask=mask8,
    )


@triton.jit
def _matmul_hidden(
    grad_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pid = tl.cdiv(5120, BLOCK_N)
    pid_m = pid // n_pid
    pid_n = pid % n_pid
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, 160, BLOCK_K):
        kk = k_start + rk
        a = tl.load(
            grad_ptr + rows[:, None] * 160 + kk[None, :],
            mask=(rows[:, None] < n_rows) & (kk[None, :] < 160),
            other=0.0,
        )
        b = tl.load(
            weight_ptr + kk[:, None] * 5120 + cols[None, :],
            mask=(kk[:, None] < 160) & (cols[None, :] < 5120),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        output_ptr + rows[:, None] * 5120 + cols[None, :],
        acc,
        mask=(rows[:, None] < n_rows) & (cols[None, :] < 5120),
    )


@triton.jit
def _matmul_weight(
    grad_ptr,
    hidden_ptr,
    output_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pid = tl.cdiv(5120, BLOCK_N)
    pid_m = pid // n_pid
    pid_n = pid % n_pid
    experts = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, n_rows, BLOCK_K):
        kk = k_start + rk
        a = tl.load(
            grad_ptr + kk[:, None] * 160 + experts[None, :],
            mask=(kk[:, None] < n_rows) & (experts[None, :] < 160),
            other=0.0,
        ).T
        b = tl.load(
            hidden_ptr + kk[:, None] * 5120 + cols[None, :],
            mask=(kk[:, None] < n_rows) & (cols[None, :] < 5120),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        output_ptr + experts[:, None] * 5120 + cols[None, :],
        acc,
        mask=(experts[:, None] < 160) & (cols[None, :] < 5120),
    )


@triton.jit
def _sparse_hidden(
    grad_ptr,
    indices_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pid = tl.cdiv(5120, BLOCK_N)
    pid_m = pid // n_pid
    pid_n = pid % n_pid
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in tl.static_range(0, 8):
        expert = tl.load(
            indices_ptr + rows * 8 + k, mask=row_mask, other=0
        )
        grad = tl.load(
            grad_ptr + rows * 160 + expert, mask=row_mask, other=0.0
        )
        w = tl.load(
            weight_ptr + expert[:, None] * 5120 + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < 5120),
            other=0.0,
        )
        acc += grad[:, None].to(tl.float32) * w.to(tl.float32)
    tl.store(
        output_ptr + rows[:, None] * 5120 + cols[None, :],
        acc,
        mask=row_mask[:, None] & (cols[None, :] < 5120),
    )


@torch.no_grad()
def run(
    grad_topk_weights,
    hidden_states,
    weight,
    scores,
    topk_indices,
    topk_weights,
    topk_weights_normalized,
    denominator,
    routed_scaling_factor,
):
    n = hidden_states.shape[0]
    grad_logits = torch.empty_like(scores)
    _make_grad_logits_sparse_store[(triton.cdiv(n, 8),)](
        grad_topk_weights,
        topk_weights_normalized,
        denominator,
        scores,
        topk_indices,
        grad_logits,
        routed_scaling_factor,
        n,
        BLOCK_M=8,
        BLOCK_E=256,
        num_warps=4,
    )
    hidden_solution = _GRAD_HIDDEN_SOLUTIONS.get(n)
    if hidden_solution is None:
        grad_hidden = grad_logits @ weight
    else:
        grad_hidden = aiter.hipb_mm(
            grad_logits, weight, hidden_solution
        )
    solution = _GRAD_WEIGHT_SOLUTIONS.get(n)
    if solution is None:
        grad_weight = grad_logits.T @ hidden_states
    else:
        grad_weight = aiter.hipb_mm(
            grad_logits.T, hidden_states, solution
        )
    return grad_hidden, grad_weight
