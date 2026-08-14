import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_masks_kernel(
    full_ptr,
    swa_ptr,
    n_elements: tl.constexpr,
    seq_length: tl.constexpr,
    source_length: tl.constexpr,
    past_length: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < n_elements
    position = offsets % (seq_length * source_length)
    target = position // source_length
    source = position - target * source_length
    full_value = source > target + past_length
    tl.store(full_ptr + offsets, full_value, mask=active)
    tl.store(swa_ptr + offsets, 0, mask=active)


@triton.jit
def _prepare_masks_by_row_kernel(
    full_ptr,
    swa_ptr,
    n_rows: tl.constexpr,
    seq_length: tl.constexpr,
    source_length: tl.constexpr,
    past_length: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    COL_CHUNKS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    first_row = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM
    lane = tl.arange(0, BLOCK_COLS)

    for row_delta in tl.static_range(0, ROWS_PER_PROGRAM):
        row = first_row + row_delta
        target = (row % seq_length).to(tl.int32)
        row_start = row * source_length
        causal_limit = target + past_length
        for col_chunk in tl.static_range(0, COL_CHUNKS):
            source = col_chunk * BLOCK_COLS + lane
            active = (row < n_rows) & (source < source_length)
            offsets = row_start + source
            tl.store(full_ptr + offsets, source > causal_limit, mask=active)
            tl.store(swa_ptr + offsets, 0, mask=active)


@triton.jit
def _prepare_masks_packed_kernel(
    full_ptr,
    swa_ptr,
    n_rows: tl.constexpr,
    seq_length: tl.constexpr,
    source_length: tl.constexpr,
    past_length: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
    WORD_CHUNKS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    N_WORDS: tl.constexpr,
    TAIL: tl.constexpr,
):
    first_row = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM
    lane = tl.arange(0, BLOCK_WORDS)

    for row_delta in tl.static_range(0, ROWS_PER_PROGRAM):
        row = first_row + row_delta
        target = (row % seq_length).to(tl.int32)
        row_start = row * source_length
        causal_limit = target + past_length
        full_words = (full_ptr + row_start).to(tl.pointer_type(tl.uint32))
        swa_words = (swa_ptr + row_start).to(tl.pointer_type(tl.uint32))
        for word_chunk in tl.static_range(0, WORD_CHUNKS):
            word = word_chunk * BLOCK_WORDS + lane
            source = word * 4
            active = (row < n_rows) & (word < N_WORDS)
            nfalse = tl.maximum(0, tl.minimum(4, causal_limit - source + 1))
            shift = tl.minimum(nfalse, 3) * 8
            packed = tl.full(source.shape, 0x01010101, tl.uint32) << shift
            packed = tl.where(nfalse == 4, 0, packed)
            tl.store(full_words + word, packed, mask=active)
            tl.store(swa_words + word, 0, mask=active)
        if TAIL:
            source = N_WORDS * 4 + lane
            active = (row < n_rows) & (lane < TAIL)
            tl.store(full_ptr + row_start + source, source > causal_limit, mask=active)
            tl.store(swa_ptr + row_start + source, 0, mask=active)


@triton.jit
def _prepare_masks_packed64_kernel(
    full_ptr,
    swa_ptr,
    n_rows: tl.constexpr,
    seq_length: tl.constexpr,
    source_length: tl.constexpr,
    past_length: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
    WORD_CHUNKS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    N_WORDS: tl.constexpr,
    TAIL: tl.constexpr,
):
    first_row = tl.program_id(0).to(tl.int64) * ROWS_PER_PROGRAM
    lane = tl.arange(0, BLOCK_WORDS)

    for row_delta in tl.static_range(0, ROWS_PER_PROGRAM):
        row = first_row + row_delta
        target = (row % seq_length).to(tl.int32)
        row_start = row * source_length
        causal_limit = target + past_length
        full_words = (full_ptr + row_start).to(tl.pointer_type(tl.uint64))
        swa_words = (swa_ptr + row_start).to(tl.pointer_type(tl.uint64))
        for word_chunk in tl.static_range(0, WORD_CHUNKS):
            word = word_chunk * BLOCK_WORDS + lane
            source = word * 8
            active = (row < n_rows) & (word < N_WORDS)
            nfalse = tl.maximum(0, tl.minimum(8, causal_limit - source + 1))
            shift = tl.minimum(nfalse, 7) * 8
            packed = tl.full(source.shape, 0x0101010101010101, tl.uint64) << shift
            packed = tl.where(nfalse == 8, 0, packed)
            tl.store(full_words + word, packed, mask=active)
            tl.store(swa_words + word, 0, mask=active)
        if TAIL:
            source = N_WORDS * 8 + lane
            active = (row < n_rows) & (lane < TAIL)
            tl.store(full_ptr + row_start + source, source > causal_limit, mask=active)
            tl.store(swa_ptr + row_start + source, 0, mask=active)


@triton.jit
def _prepare_masks_flat_packed64_kernel(
    full_ptr,
    swa_ptr,
    n_words: tl.constexpr,
    seq_length: tl.constexpr,
    words_per_row: tl.constexpr,
    past_length: tl.constexpr,
    BLOCK: tl.constexpr,
):
    word = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = word < n_words
    row = word // words_per_row
    source = (word - row * words_per_row) * 8
    target = row % seq_length
    causal_limit = target + past_length
    nfalse = tl.maximum(0, tl.minimum(8, causal_limit - source + 1))
    shift = tl.minimum(nfalse, 7) * 8
    packed = tl.full(source.shape, 0x0101010101010101, tl.uint64) << shift
    packed = tl.where(nfalse == 8, 0, packed)
    full_words = full_ptr.to(tl.pointer_type(tl.uint64))
    swa_words = swa_ptr.to(tl.pointer_type(tl.uint64))
    tl.store(full_words + word, packed, mask=active, cache_modifier=".cs")
    tl.store(swa_words + word, 0, mask=active, cache_modifier=".cs")


@triton.jit
def _prepare_masks_flat_coarse_kernel(
    full_ptr,
    swa_ptr,
    n_words: tl.constexpr,
    seq_length: tl.constexpr,
    words_per_row: tl.constexpr,
    past_length: tl.constexpr,
    BLOCK: tl.constexpr,
):
    word = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = word < n_words
    row = word // words_per_row
    source = (word - row * words_per_row) * 8
    target = row % seq_length
    packed = tl.where(
        source > target + past_length,
        tl.full(source.shape, 0x0101010101010101, tl.uint64),
        0,
    )
    full_words = full_ptr.to(tl.pointer_type(tl.uint64))
    swa_words = swa_ptr.to(tl.pointer_type(tl.uint64))
    tl.store(full_words + word, packed, mask=active)
    tl.store(swa_words + word, 0, mask=active)


@triton.jit
def _fix_mask_boundary_kernel(
    full_ptr,
    n_rows: tl.constexpr,
    seq_length: tl.constexpr,
    words_per_row: tl.constexpr,
    past_length: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = row < n_rows
    target = row % seq_length
    causal_limit = target + past_length
    boundary_word = causal_limit // 8
    nfalse = causal_limit - boundary_word * 8 + 1
    shift = tl.minimum(nfalse, 7) * 8
    packed = tl.full(row.shape, 0x0101010101010101, tl.uint64) << shift
    packed = tl.where(nfalse == 8, 0, packed)
    full_words = full_ptr.to(tl.pointer_type(tl.uint64))
    tl.store(full_words + row * words_per_row + boundary_word, packed, mask=active)


@torch.no_grad()
def run(batch_size_scalar, seq_length_scalar, past_key_values_length_scalar):
    batch_size = int(batch_size_scalar)
    seq_length = int(seq_length_scalar)
    past_length = int(past_key_values_length_scalar)
    source_length = seq_length + past_length

    shape = (batch_size, 64, seq_length, source_length)
    full_attention_mask = torch.empty(shape, dtype=torch.bool, device="cuda")
    sliding_window_attention_mask = torch.empty_like(full_attention_mask)
    n_elements = full_attention_mask.numel()

    if source_length % 8 == 0:
        n_words = n_elements // 8
        if n_words < 100_000:
            block = 256
        elif n_elements < 100_000_000:
            block = 512
        else:
            block = 2048
        _prepare_masks_flat_packed64_kernel[(triton.cdiv(n_words, block),)](
            full_attention_mask,
            sliding_window_attention_mask,
            n_words,
            seq_length,
            source_length // 8,
            past_length,
            BLOCK=block,
            num_warps=8 if block >= 512 else 4,
            num_stages=1,
        )
    elif n_elements >= 100_000_000:
        if source_length <= 512:
            rows_per_program = 32
        elif source_length <= 1024:
            rows_per_program = 16
        elif source_length <= 2048:
            rows_per_program = 8
        else:
            rows_per_program = 4 if source_length < 4096 else 2
        n_rows = batch_size * 64 * seq_length
        if source_length <= 1024 and source_length % 8:
            n_words = source_length // 4
            block_words = 256
            word_chunks = triton.cdiv(n_words, block_words)
            _prepare_masks_packed_kernel[(triton.cdiv(n_rows, rows_per_program),)](
                full_attention_mask,
                sliding_window_attention_mask,
                n_rows,
                seq_length,
                source_length,
                past_length,
                BLOCK_WORDS=block_words,
                WORD_CHUNKS=word_chunks,
                ROWS_PER_PROGRAM=rows_per_program,
                N_WORDS=n_words,
                TAIL=source_length - n_words * 4,
                num_warps=4,
                num_stages=1,
            )
        else:
            n_words = source_length // 8
            if source_length <= 512:
                block_words = 64
            elif source_length <= 1024:
                block_words = 128
            elif source_length <= 2048:
                block_words = 256
            else:
                block_words = 512
            word_chunks = triton.cdiv(n_words, block_words)
            _prepare_masks_packed64_kernel[(triton.cdiv(n_rows, rows_per_program),)](
                full_attention_mask,
                sliding_window_attention_mask,
                n_rows,
                seq_length,
                source_length,
                past_length,
                BLOCK_WORDS=block_words,
                WORD_CHUNKS=word_chunks,
                ROWS_PER_PROGRAM=rows_per_program,
                N_WORDS=n_words,
                TAIL=source_length - n_words * 8,
                num_warps=max(1, block_words // 64),
                num_stages=1,
            )
    else:
        block = 1024
        _prepare_masks_kernel[(triton.cdiv(n_elements, block),)](
            full_attention_mask,
            sliding_window_attention_mask,
            n_elements,
            seq_length,
            source_length,
            past_length,
            BLOCK=block,
            num_warps=8,
        )
    return full_attention_mask, sliding_window_attention_mask
