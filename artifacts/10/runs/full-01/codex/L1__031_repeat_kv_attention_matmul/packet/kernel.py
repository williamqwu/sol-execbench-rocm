import torch
import triton
import triton.language as tl


@triton.jit
def _attention_scores_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    head = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    head_offset = head * seq_len * 128
    q_base = query + head_offset
    k_base = key + head_offset

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 128, BLOCK_K):
        k_idx = k_start + offs_k
        q = tl.load(
            q_base + offs_m[:, None] * 128 + k_idx[None, :],
            mask=offs_m[:, None] < seq_len,
            other=0.0,
        )
        k = tl.load(
            k_base + offs_n[None, :] * 128 + k_idx[:, None],
            mask=offs_n[None, :] < seq_len,
            other=0.0,
        )
        acc += tl.dot(q, k)

    out_offset = head * seq_len * out_stride
    out_ptrs = output + out_offset + offs_m[:, None] * out_stride + offs_n[None, :]
    tl.store(
        out_ptrs,
        acc * 0.08838834764831845,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
    )


@triton.jit
def _locality_attention_scores_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    flat_pid = tl.program_id(0)
    num_m = tl.cdiv(seq_len, BLOCK_M)
    num_n = tl.cdiv(seq_len, BLOCK_N)
    tiles_per_head = num_m * num_n
    head = flat_pid // tiles_per_head
    tile_pid = flat_pid - head * tiles_per_head

    tiles_per_group = GROUP_M * num_n
    group_id = tile_pid // tiles_per_group
    first_m = group_id * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    in_group = tile_pid - group_id * tiles_per_group
    tile_m = first_m + (in_group % group_m)
    tile_n = in_group // group_m

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    head_offset = head * seq_len * 128

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 128, BLOCK_K):
        k_idx = k_start + offs_k
        q = tl.load(
            query + head_offset + offs_m[:, None] * 128 + k_idx[None, :],
            mask=offs_m[:, None] < seq_len,
            other=0.0,
        )
        k = tl.load(
            key + head_offset + offs_n[None, :] * 128 + k_idx[:, None],
            mask=offs_n[None, :] < seq_len,
            other=0.0,
        )
        acc += tl.dot(q, k)

    out_ptrs = (
        output
        + head * seq_len * out_stride
        + offs_m[:, None] * out_stride
        + offs_n[None, :]
    )
    tl.store(
        out_ptrs,
        acc * 0.08838834764831845,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
    )


@triton.jit
def _persistent_query_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    head = tl.program_id(0)
    tile_m = tl.program_id(1)

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    head_offset = head * seq_len * 128
    q = tl.load(
        query + head_offset + offs_m[:, None] * 128 + offs_k[None, :],
        mask=offs_m[:, None] < seq_len,
        other=0.0,
    )

    for n_start in tl.range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        k = tl.load(
            key + head_offset + offs_k[:, None] + offs_n[None, :] * 128,
            mask=offs_n[None, :] < seq_len,
            other=0.0,
        )
        acc = tl.dot(q, k)
        out_ptrs = (
            output
            + head * seq_len * out_stride
            + offs_m[:, None] * out_stride
            + offs_n[None, :]
        )
        tl.store(
            out_ptrs,
            acc * 0.08838834764831845,
            mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
        )


@triton.jit
def _persistent_key_kernel(
    query,
    key,
    output,
    seq_len: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    head = tl.program_id(0)
    tile_n = tl.program_id(1)

    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    head_offset = head * seq_len * 128
    k = tl.load(
        key + head_offset + offs_k[:, None] + offs_n[None, :] * 128,
        mask=offs_n[None, :] < seq_len,
        other=0.0,
    )

    for m_start in tl.range(0, seq_len, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        q = tl.load(
            query + head_offset + offs_m[:, None] * 128 + offs_k[None, :],
            mask=offs_m[:, None] < seq_len,
            other=0.0,
        )
        acc = tl.dot(q, k)
        out_ptrs = (
            output
            + head * seq_len * out_stride
            + offs_m[:, None] * out_stride
            + offs_n[None, :]
        )
        tl.store(
            out_ptrs,
            acc * 0.08838834764831845,
            mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len),
        )


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_len, _ = query.shape
    # A short padded row makes stores coalesced for odd sequence lengths.  The
    # returned tensor is a normal strided view with exactly the requested shape.
    if seq_len >= 1500 and seq_len % 128 != 0:
        out_stride = triton.cdiv(seq_len, 128) * 128
    else:
        out_stride = triton.cdiv(seq_len, 16) * 16 if seq_len > 256 else seq_len
    output_storage = torch.empty(
        (batch_size, num_heads, seq_len, out_stride),
        device=query.device,
        dtype=torch.bfloat16,
    )
    output = output_storage[..., :seq_len]

    # More, smaller tiles keep all CUs occupied for the shortest single-batch cases.
    if seq_len <= 128 and batch_size * num_heads >= 256:
        block_m = 128
        block_n = 128
        num_warps = 8
    elif seq_len <= 256:
        block_m = 64
        block_n = 64
        num_warps = 4
    else:
        block_m = 128
        block_n = 128
        num_warps = 8

    if seq_len == 1024 and batch_size >= 4:
        block_n = 256

    block_k = 64 if seq_len > 256 and seq_len % 128 != 0 else 32
    use_locality_order = seq_len >= 2048
    if use_locality_order:
        grid = (
            batch_size
            * num_heads
            * triton.cdiv(seq_len, block_m)
            * triton.cdiv(seq_len, block_n),
        )
        _locality_attention_scores_kernel[grid](
            query,
            key,
            output_storage,
            seq_len=seq_len,
            out_stride=out_stride,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=4 if seq_len < 2048 else 1,
            num_warps=num_warps,
        )
    else:
        grid = (
            batch_size * num_heads,
            triton.cdiv(seq_len, block_m),
            triton.cdiv(seq_len, block_n),
        )
        _attention_scores_kernel[grid](
            query,
            key,
            output_storage,
            seq_len=seq_len,
            out_stride=out_stride,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
    return output
