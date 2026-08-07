import torch
import triton
import triton.language as tl


_AUXILIARY_STREAMS = {}
_PARALLEL_OUTPUT_THRESHOLD = 5_000_000
_PARALLEL_QKV_BIAS_THRESHOLD = 5_000_000
_PARALLEL_PROJECTION_MAX_TOKENS = 1 << 30
_PARALLEL_EDIT_MAX_TOKENS = 8192
_DROPOUT_FUSION_THRESHOLD = 5_000_000
_DIRECT_EDIT_MAX_BATCH = 4
_DIRECT_EDIT_MAX_SEQ = 1024


def _get_auxiliary_streams():
    device = torch.cuda.current_device()
    streams = _AUXILIARY_STREAMS.get(device)
    if streams is None:
        streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
        _AUXILIARY_STREAMS[device] = streams
    return streams


@triton.jit
def _dropout_mask_and_product(
    grad_ptr,
    dropout_mask_ptr,
    probs_ptr,
    product_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=valid)
    dropout = tl.load(dropout_mask_ptr + offsets, mask=valid)
    probs = tl.load(probs_ptr + offsets, mask=valid)
    grad = grad * dropout
    tl.store(grad_ptr + offsets, grad, mask=valid)
    tl.store(product_ptr + offsets, grad * probs, mask=valid)


@triton.jit
def _finish_softmax_bwd(
    grad_ptr,
    probs_ptr,
    row_sum_ptr,
    attention_mask_ptr,
    edit_mask_ptr,
    cross_product_ptr,
    within_product_ptr,
    n_elements: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < n_elements
    row = offset // SEQ
    col = offset - row * SEQ
    grad = tl.load(grad_ptr + offset, mask=valid)
    probs = tl.load(probs_ptr + offset, mask=valid)
    row_sum = tl.load(row_sum_ptr + row, mask=valid)
    batch = row // (16 * SEQ)
    valid_key = tl.load(
        attention_mask_ptr + batch * SEQ + col, mask=valid, other=0.0
    )
    result = probs * (grad - row_sum)
    result = tl.where(valid_key == 0.0, 0.0, result)
    tl.store(grad_ptr + offset, result, mask=valid)
    query_position = row - (row // SEQ) * SEQ
    edit_q = tl.load(edit_mask_ptr + batch * SEQ + query_position, mask=valid)
    edit_k = tl.load(edit_mask_ptr + batch * SEQ + col, mask=valid)
    cross = edit_q * (1.0 - edit_k) + (1.0 - edit_q) * edit_k
    within = edit_q * edit_k
    tl.store(cross_product_ptr + offset, result * cross, mask=valid)
    tl.store(within_product_ptr + offset, result * within, mask=valid)


@triton.jit
def _pack_qkv_bwd(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    token = offsets // 3072
    component_offset = offsets - token * 3072
    component = component_offset // 1024
    hidden = component_offset - component * 1024
    batch = token // SEQ
    position = token - batch * SEQ
    head = hidden // 64
    lane = hidden - head * 64
    source_offset = ((batch * 16 + head) * SEQ + position) * 64 + lane
    q = tl.load(query_ptr + source_offset, mask=valid & (component == 0), other=0.0)
    k = tl.load(key_ptr + source_offset, mask=valid & (component == 1), other=0.0)
    v = tl.load(value_ptr + source_offset, mask=valid & (component == 2), other=0.0)
    result = tl.where(component == 0, q, tl.where(component == 1, k, v))
    tl.store(output_ptr + offsets, result, mask=valid)


@triton.jit
def _sum_edit_region(
    scores_ptr,
    output_ptr,
    BATCH: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_elements: tl.constexpr = 16 * 1024 * 1024
    valid = offsets < output_elements
    head = offsets // (1024 * 1024)
    position = offsets - head * (1024 * 1024)
    row = position // 1024
    col = position - row * 1024
    inside = valid & (row < SEQ) & (col < SEQ)
    total = tl.zeros((BLOCK,), tl.float32)
    for batch in tl.static_range(0, BATCH):
        source = ((batch * 16 + head) * SEQ + row) * SEQ + col
        total += tl.load(scores_ptr + source, mask=inside, other=0.0)
    tl.store(output_ptr + offsets, total, mask=valid)


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    edit_region_mask,
    qkv_weight,
    qkv_bias,
    out_weight,
    out_bias,
    edit_region_bias,
    within_edit_bias,
    cross_edit_bias,
    attention_mask,
    query,
    key,
    value,
    attention_scores,
    attention_probs,
    attention_probs_dropped,
    attention_output,
    dropout_mask,
    scale,
    dropout_p,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 16
    head_dim = 64

    grad_attention_output = torch.matmul(grad_output, out_weight)
    grad_attention_output = grad_attention_output.reshape(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)

    grad_attention_probs_dropped = torch.matmul(
        grad_attention_output, value.transpose(-2, -1)
    )
    grad_value = torch.matmul(
        attention_probs_dropped.transpose(-2, -1), grad_attention_output
    )

    score_elements = batch_size * num_heads * seq_len * seq_len
    parallel_projection = (
        score_elements >= _PARALLEL_OUTPUT_THRESHOLD
        and batch_size * seq_len <= _PARALLEL_PROJECTION_MAX_TOKENS
    )
    parallel_edit = (
        score_elements >= _PARALLEL_OUTPUT_THRESHOLD
        and batch_size * seq_len <= _PARALLEL_EDIT_MAX_TOKENS
    )
    parallel_qkv_bias = (
        score_elements >= _PARALLEL_QKV_BIAS_THRESHOLD
        and not (batch_size >= 64 and seq_len > 128)
    )
    if parallel_projection or parallel_edit or parallel_qkv_bias:
        main_stream = torch.cuda.current_stream()
        output_stream, edit_stream = _get_auxiliary_streams()
    if parallel_projection:
        output_stream.wait_stream(main_stream)
        with torch.cuda.stream(output_stream):
            grad_out_weight = torch.matmul(
                grad_output.reshape(-1, hidden_size).t(),
                attention_output.reshape(-1, hidden_size),
            )
            grad_out_bias = grad_output.sum(dim=(0, 1))
    else:
        grad_out_weight = torch.matmul(
            grad_output.reshape(-1, hidden_size).t(),
            attention_output.reshape(-1, hidden_size),
        )
        grad_out_bias = grad_output.sum(dim=(0, 1))

    softmax_product = torch.empty_like(grad_attention_probs_dropped)
    if dropout_p > 0:
        if score_elements >= _DROPOUT_FUSION_THRESHOLD:
            grad_attention_probs_dropped.div_(1 - dropout_p)
            _dropout_mask_and_product[(triton.cdiv(score_elements, 256),)](
                grad_attention_probs_dropped,
                dropout_mask,
                attention_probs,
                softmax_product,
                n_elements=score_elements,
                BLOCK=256,
                num_warps=4,
            )
        else:
            grad_attention_probs_dropped.mul_(dropout_mask)
            grad_attention_probs_dropped.div_(1 - dropout_p)
            torch.mul(
                grad_attention_probs_dropped,
                attention_probs,
                out=softmax_product,
            )
    else:
        torch.mul(
            grad_attention_probs_dropped, attention_probs, out=softmax_product
        )

    sum_grad = softmax_product.sum(dim=-1, keepdim=True)
    edit_products = torch.empty(
        (2,) + tuple(grad_attention_probs_dropped.shape),
        device=grad_attention_probs_dropped.device,
        dtype=grad_attention_probs_dropped.dtype,
    )
    _finish_softmax_bwd[(triton.cdiv(score_elements, 1024),)](
        grad_attention_probs_dropped,
        attention_probs,
        sum_grad,
        attention_mask,
        edit_region_mask,
        edit_products[0],
        edit_products[1],
        n_elements=score_elements,
        SEQ=seq_len,
        BLOCK=1024,
        num_warps=4,
    )
    grad_attention_scores = grad_attention_probs_dropped

    if parallel_edit:
        edit_stream.wait_stream(main_stream)
        with torch.cuda.stream(edit_stream):
            grad_cross_edit_bias = edit_products[0].sum(
                dim=(0, 2, 3), keepdim=True
            ).reshape(num_heads, 1, 1)
            grad_within_edit_bias = edit_products[1].sum(
                dim=(0, 2, 3), keepdim=True
            ).reshape(num_heads, 1, 1)
            if (
                batch_size == 1
                and seq_len == edit_region_bias.shape[-1]
            ):
                grad_edit_region_bias = grad_attention_scores[0]
            elif (
                batch_size <= _DIRECT_EDIT_MAX_BATCH
                and seq_len <= _DIRECT_EDIT_MAX_SEQ
            ):
                grad_edit_region_bias = torch.empty_like(edit_region_bias)
                _sum_edit_region[(triton.cdiv(16 * 1024 * 1024, 1024),)](
                    grad_attention_scores,
                    grad_edit_region_bias,
                    BATCH=batch_size,
                    SEQ=seq_len,
                    BLOCK=1024,
                    num_warps=4,
                )
            else:
                grad_edit_region_bias = torch.zeros_like(edit_region_bias)
                if seq_len <= 1024:
                    grad_edit_region_bias[:, :seq_len, :seq_len] = (
                        grad_attention_scores.sum(dim=0)
                    )
        grad_attention_scores_scaled = grad_attention_scores * scale
    else:
        grad_cross_edit_bias = edit_products[0].sum(
            dim=(0, 2, 3), keepdim=True
        ).reshape(num_heads, 1, 1)
        grad_within_edit_bias = edit_products[1].sum(
            dim=(0, 2, 3), keepdim=True
        ).reshape(num_heads, 1, 1)
        if (
            batch_size <= _DIRECT_EDIT_MAX_BATCH
            and seq_len <= _DIRECT_EDIT_MAX_SEQ
        ):
            grad_edit_region_bias = torch.empty_like(edit_region_bias)
            _sum_edit_region[(triton.cdiv(16 * 1024 * 1024, 1024),)](
                grad_attention_scores,
                grad_edit_region_bias,
                BATCH=batch_size,
                SEQ=seq_len,
                BLOCK=1024,
                num_warps=4,
            )
        else:
            grad_edit_region_bias = torch.zeros_like(edit_region_bias)
            if seq_len <= 1024:
                grad_edit_region_bias[:, :seq_len, :seq_len] = (
                    grad_attention_scores.sum(dim=0)
                )
        grad_attention_scores.mul_(scale)
        grad_attention_scores_scaled = grad_attention_scores

    grad_query = torch.matmul(grad_attention_scores_scaled, key)
    grad_key = torch.matmul(
        grad_attention_scores_scaled.transpose(-2, -1), query
    )

    grad_qkv = torch.empty(
        (batch_size, seq_len, 3 * hidden_size),
        device=grad_query.device,
        dtype=grad_query.dtype,
    )
    qkv_elements = batch_size * seq_len * 3 * hidden_size
    _pack_qkv_bwd[(triton.cdiv(qkv_elements, 1024),)](
        grad_query,
        grad_key,
        grad_value,
        grad_qkv,
        n_elements=qkv_elements,
        SEQ=seq_len,
        BLOCK=1024,
        num_warps=4,
    )

    if parallel_qkv_bias:
        edit_stream.wait_stream(main_stream)
        with torch.cuda.stream(edit_stream):
            grad_qkv_bias = grad_qkv.sum(dim=(0, 1))

    grad_hidden_states = torch.matmul(grad_qkv, qkv_weight)
    grad_qkv_weight = torch.matmul(
        grad_qkv.reshape(-1, 3 * hidden_size).t(),
        hidden_states.reshape(-1, hidden_size),
    )
    if parallel_projection:
        main_stream.wait_stream(output_stream)
    if parallel_edit or parallel_qkv_bias:
        main_stream.wait_stream(edit_stream)
    if not parallel_qkv_bias:
        grad_qkv_bias = grad_qkv.sum(dim=(0, 1))

    return (
        grad_hidden_states,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_edit_region_bias,
        grad_within_edit_bias,
        grad_cross_edit_bias,
    )
