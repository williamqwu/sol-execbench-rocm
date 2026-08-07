import torch
import triton
import triton.language as tl
import aiter


@triton.jit
def _ragged_gqa_fwd(
    Q,
    K,
    V,
    QO_INDPTR,
    KV_INDPTR,
    Out,
    Lse,
    sm_scale,
    BATCH: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # A program owns BLOCK_M query positions and all four query heads that
    # share one KV head.  Packed tile ids avoid padding every sequence to the
    # longest sequence in a ragged batch.
    tile_id = tl.program_id(0)
    head_group = tl.program_id(1)
    groups_per_kv = 4 // GROUP
    kv_head = head_group // groups_per_kv
    sub_group = head_group % groups_per_kv

    tile_prefix = 0
    q_start = 0
    q_len = 0
    kv_start = 0
    kv_len = 0
    local_tile = 0
    live_tile = False
    for b in range(BATCH):
        q0 = tl.load(QO_INDPTR + b)
        q1 = tl.load(QO_INDPTR + b + 1)
        k0 = tl.load(KV_INDPTR + b)
        k1 = tl.load(KV_INDPTR + b + 1)
        ntiles = tl.cdiv(q1 - q0, BLOCK_M)
        owns = (tile_id >= tile_prefix) & (tile_id < tile_prefix + ntiles)
        q_start = tl.where(owns, q0, q_start)
        q_len = tl.where(owns, q1 - q0, q_len)
        kv_start = tl.where(owns, k0, kv_start)
        kv_len = tl.where(owns, k1 - k0, kv_len)
        local_tile = tl.where(owns, tile_id - tile_prefix, local_tile)
        live_tile = live_tile | owns
        tile_prefix += ntiles

    # Flatten [query position, 4 GQA heads] into the row dimension of MFMA.
    offs_m = tl.arange(0, BLOCK_M * GROUP)
    tok_in_tile = offs_m // GROUP
    q_group_head = offs_m % GROUP
    local_q = local_tile * BLOCK_M + tok_in_tile
    global_q = q_start + local_q
    q_head = kv_head * 4 + sub_group * GROUP + q_group_head
    valid_q = live_tile & (local_q < q_len)
    offs_d = tl.arange(0, 128)

    q_offsets = (
        (global_q[:, None] * 32 + q_head[:, None]) * 128
        + offs_d[None, :]
    )
    q = tl.load(Q + q_offsets, mask=valid_q[:, None], other=0.0)

    m_i = tl.full((BLOCK_M * GROUP,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M * GROUP,), tl.float32)
    acc = tl.zeros((BLOCK_M * GROUP, 128), tl.float32)

    # Bottom-right causal alignment: q may attend through q + (K-Q).
    delta = kv_len - q_len
    last_key = tl.minimum(kv_len, local_tile * BLOCK_M + BLOCK_M + delta)
    for start_n in tl.range(0, last_key, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_k = live_tile & (offs_n < kv_len)
        k_offsets = (
            ((kv_start + offs_n)[:, None] * 8 + kv_head) * 128
            + offs_d[None, :]
        )
        k = tl.load(K + k_offsets, mask=valid_k[:, None], other=0.0)
        v = tl.load(V + k_offsets, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk *= sm_scale * 1.4426950408889634
        causal = offs_n[None, :] <= (local_q[:, None] + delta)
        keep = valid_q[:, None] & valid_k[None, :] & causal
        qk = tl.where(keep, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(l_i == 0.0, 0.0, tl.exp2(m_i - m_new))
        p = tl.where(keep, tl.exp2(qk - m_new[:, None]), 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.bfloat16), v, out_dtype=tl.float32
        )
        m_i = m_new
        l_i = l_new

    nonempty = l_i > 0.0
    out = acc / tl.where(nonempty, l_i, 1.0)[:, None]
    out_offsets = q_offsets
    tl.store(Out + out_offsets, out, mask=valid_q[:, None])
    lse = tl.where(nonempty, m_i + tl.log2(l_i), -float("inf"))
    tl.store(Lse + global_q * 32 + q_head, lse, mask=valid_q)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    # The tuned assembly path has a persistent packed-tile scheduler that is
    # especially effective once attention becomes compute bound.  Its LSE is
    # natural-log and head-major, so convert it to the requested base-2 layout.
    if q.shape[0] > 2048:
        output, native_lse = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            q.shape[0],
            k.shape[0],
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=True,
            return_lse=True,
        )
        return output, native_lse.transpose(0, 1) * 1.4426950408889634

    output = torch.empty_like(q)
    lse = torch.empty((q.shape[0], 32), dtype=torch.float32, device=q.device)
    batch = qo_indptr.shape[0] - 1
    # Four query positions make exactly one 16-row MFMA tile after folding in
    # the GQA group.  It wins for launch-bound inputs; longer inputs benefit
    # strongly from a 128-row (32 position) tile and a narrower key tile.
    if q.shape[0] <= 128:
        block_m = 4
        block_n = 64
    else:
        block_m = 32
        block_n = 32
    # sum(ceil(length_i / block_m)) <= ceil(total / block_m) + batch - 1
    max_tiles = triton.cdiv(q.shape[0], block_m) + batch - 1
    group = 4
    _ragged_gqa_fwd[(max_tiles, 32 // group)](
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        output,
        lse,
        sm_scale,
        BATCH=batch,
        GROUP=group,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output, lse
