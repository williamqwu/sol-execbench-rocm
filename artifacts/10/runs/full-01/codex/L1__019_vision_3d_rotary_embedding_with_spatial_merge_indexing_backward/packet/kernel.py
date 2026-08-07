import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


HEADS = tl.constexpr(16)
HEAD_DIM = tl.constexpr(128)
HALF_DIM = tl.constexpr(64)


@triton.jit
def _rounded_add(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _backward_kernel(
    grad_q_ptr, grad_k_ptr, q_ptr, k_ptr, embeddings_ptr,
    out_q_ptr, out_k_ptr, out_embeddings_ptr,
):
    token = tl.program_id(0)
    d0 = tl.arange(0, HALF_DIM)
    d1 = d0 + HALF_DIM

    emb_base = token * HEAD_DIM
    e0 = tl.load(embeddings_ptr + emb_base + d0)
    e1 = tl.load(embeddings_ptr + emb_base + d1)
    c0 = libdevice.cos(e0)
    s0 = libdevice.sin(e0)
    c1 = libdevice.cos(e1)
    s1 = libdevice.sin(e1)

    gc0 = tl.zeros((HALF_DIM,), tl.float32)
    gs0 = tl.zeros((HALF_DIM,), tl.float32)
    gc1 = tl.zeros((HALF_DIM,), tl.float32)
    gs1 = tl.zeros((HALF_DIM,), tl.float32)
    token_base = token * HEADS * HEAD_DIM

    for rem in tl.static_range(4):
        pc0 = tl.zeros((HALF_DIM,), tl.float32)
        ps0 = tl.zeros((HALF_DIM,), tl.float32)
        pc1 = tl.zeros((HALF_DIM,), tl.float32)
        ps1 = tl.zeros((HALF_DIM,), tl.float32)
        for group in tl.static_range(4):
            h = rem + group * 4
            base = token_base + h * HEAD_DIM
            gq0 = tl.load(grad_q_ptr + base + d0, cache_modifier=".cg")
            gq1 = tl.load(grad_q_ptr + base + d1, cache_modifier=".cg")
            gk0 = tl.load(grad_k_ptr + base + d0, cache_modifier=".cg")
            gk1 = tl.load(grad_k_ptr + base + d1, cache_modifier=".cg")
            q0 = tl.load(q_ptr + base + d0, cache_modifier=".cg")
            q1 = tl.load(q_ptr + base + d1, cache_modifier=".cg")
            k0 = tl.load(k_ptr + base + d0, cache_modifier=".cg")
            k1 = tl.load(k_ptr + base + d1, cache_modifier=".cg")

            tl.store(out_q_ptr + base + d0, gq0 * c0 + gq1 * s1, cache_modifier=".wt")
            tl.store(out_q_ptr + base + d1, gq1 * c1 - gq0 * s0, cache_modifier=".wt")
            tl.store(out_k_ptr + base + d0, gk0 * c0 + gk1 * s1, cache_modifier=".wt")
            tl.store(out_k_ptr + base + d1, gk1 * c1 - gk0 * s0, cache_modifier=".wt")

            pc0 += _rounded_add(gq0 * q0, gk0 * k0)
            pc1 += _rounded_add(gq1 * q1, gk1 * k1)
            ps0 += _rounded_add(gq0 * (-q1), gk0 * (-k1))
            ps1 += _rounded_add(gq1 * q0, gk1 * k0)

        gc0 = _rounded_add(gc0, pc0)
        gs0 = _rounded_add(gs0, ps0)
        gc1 = _rounded_add(gc1, pc1)
        gs1 = _rounded_add(gs1, ps1)

    out_e0 = _rounded_add(gc0 * (-s0), gs0 * c0)
    out_e1 = _rounded_add(gc1 * (-s1), gs1 * c1)
    tl.store(out_embeddings_ptr + emb_base + d0, out_e0, cache_modifier=".wt")
    tl.store(out_embeddings_ptr + emb_base + d1, out_e1, cache_modifier=".wt")


@triton.jit
def _backward_parallel_kernel(
    grad_q_ptr, grad_k_ptr, q_ptr, k_ptr, embeddings_ptr,
    out_q_ptr, out_k_ptr, out_embeddings_ptr,
):
    token = tl.program_id(0)
    rem = tl.arange(0, 4)[:, None]
    d0 = tl.arange(0, HALF_DIM)[None, :]
    d1 = d0 + HALF_DIM

    emb_base = token * HEAD_DIM
    e0 = tl.load(embeddings_ptr + emb_base + d0)
    e1 = tl.load(embeddings_ptr + emb_base + d1)
    c0 = libdevice.cos(e0)
    s0 = libdevice.sin(e0)
    c1 = libdevice.cos(e1)
    s1 = libdevice.sin(e1)

    lc0 = tl.zeros((4, HALF_DIM), tl.float32)
    ls0 = tl.zeros((4, HALF_DIM), tl.float32)
    lc1 = tl.zeros((4, HALF_DIM), tl.float32)
    ls1 = tl.zeros((4, HALF_DIM), tl.float32)
    rc0 = tl.zeros((4, HALF_DIM), tl.float32)
    rs0 = tl.zeros((4, HALF_DIM), tl.float32)
    rc1 = tl.zeros((4, HALF_DIM), tl.float32)
    rs1 = tl.zeros((4, HALF_DIM), tl.float32)
    token_base = token * HEADS * HEAD_DIM
    left_rem = tl.where(rem == 0, 0, tl.where(rem == 1, 3, 2))
    left_mask = rem < 3
    right_mask = rem == 0

    for group in tl.static_range(4):
        head = left_rem + group * 4
        base = token_base + head * HEAD_DIM
        gq0 = tl.load(grad_q_ptr + base + d0, mask=left_mask, other=0.0, cache_modifier=".cg")
        gq1 = tl.load(grad_q_ptr + base + d1, mask=left_mask, other=0.0, cache_modifier=".cg")
        gk0 = tl.load(grad_k_ptr + base + d0, mask=left_mask, other=0.0, cache_modifier=".cg")
        gk1 = tl.load(grad_k_ptr + base + d1, mask=left_mask, other=0.0, cache_modifier=".cg")
        q0 = tl.load(q_ptr + base + d0, mask=left_mask, other=0.0, cache_modifier=".cg")
        q1 = tl.load(q_ptr + base + d1, mask=left_mask, other=0.0, cache_modifier=".cg")
        k0 = tl.load(k_ptr + base + d0, mask=left_mask, other=0.0, cache_modifier=".cg")
        k1 = tl.load(k_ptr + base + d1, mask=left_mask, other=0.0, cache_modifier=".cg")

        tl.store(out_q_ptr + base + d0, gq0 * c0 + gq1 * s1, mask=left_mask, cache_modifier=".wt")
        tl.store(out_q_ptr + base + d1, gq1 * c1 - gq0 * s0, mask=left_mask, cache_modifier=".wt")
        tl.store(out_k_ptr + base + d0, gk0 * c0 + gk1 * s1, mask=left_mask, cache_modifier=".wt")
        tl.store(out_k_ptr + base + d1, gk1 * c1 - gk0 * s0, mask=left_mask, cache_modifier=".wt")

        lc0 = _rounded_add(lc0, _rounded_add(gq0 * q0, gk0 * k0))
        lc1 = _rounded_add(lc1, _rounded_add(gq1 * q1, gk1 * k1))
        ls0 = _rounded_add(ls0, _rounded_add(gq0 * (-q1), gk0 * (-k1)))
        ls1 = _rounded_add(ls1, _rounded_add(gq1 * q0, gk1 * k0))

        head = 1 + group * 4 + rem * 0
        base = token_base + head * HEAD_DIM
        gq0 = tl.load(grad_q_ptr + base + d0, mask=right_mask, other=0.0, cache_modifier=".cg")
        gq1 = tl.load(grad_q_ptr + base + d1, mask=right_mask, other=0.0, cache_modifier=".cg")
        gk0 = tl.load(grad_k_ptr + base + d0, mask=right_mask, other=0.0, cache_modifier=".cg")
        gk1 = tl.load(grad_k_ptr + base + d1, mask=right_mask, other=0.0, cache_modifier=".cg")
        q0 = tl.load(q_ptr + base + d0, mask=right_mask, other=0.0, cache_modifier=".cg")
        q1 = tl.load(q_ptr + base + d1, mask=right_mask, other=0.0, cache_modifier=".cg")
        k0 = tl.load(k_ptr + base + d0, mask=right_mask, other=0.0, cache_modifier=".cg")
        k1 = tl.load(k_ptr + base + d1, mask=right_mask, other=0.0, cache_modifier=".cg")

        tl.store(out_q_ptr + base + d0, gq0 * c0 + gq1 * s1, mask=right_mask, cache_modifier=".wt")
        tl.store(out_q_ptr + base + d1, gq1 * c1 - gq0 * s0, mask=right_mask, cache_modifier=".wt")
        tl.store(out_k_ptr + base + d0, gk0 * c0 + gk1 * s1, mask=right_mask, cache_modifier=".wt")
        tl.store(out_k_ptr + base + d1, gk1 * c1 - gk0 * s0, mask=right_mask, cache_modifier=".wt")

        rc0 = _rounded_add(rc0, _rounded_add(gq0 * q0, gk0 * k0))
        rc1 = _rounded_add(rc1, _rounded_add(gq1 * q1, gk1 * k1))
        rs0 = _rounded_add(rs0, _rounded_add(gq0 * (-q1), gk0 * (-k1)))
        rs1 = _rounded_add(rs1, _rounded_add(gq1 * q0, gk1 * k0))

    gc0 = tl.sum(_rounded_add(lc0, rc0), axis=0)
    gs0 = tl.sum(_rounded_add(ls0, rs0), axis=0)
    gc1 = tl.sum(_rounded_add(lc1, rc1), axis=0)
    gs1 = tl.sum(_rounded_add(ls1, rs1), axis=0)
    out_e0 = _rounded_add(gc0 * (-s0), gs0 * c0)
    out_e1 = _rounded_add(gc1 * (-s1), gs1 * c1)
    tl.store(out_embeddings_ptr + emb_base + d0, out_e0, cache_modifier=".wt")
    tl.store(out_embeddings_ptr + emb_base + d1, out_e1, cache_modifier=".wt")


@triton.jit
def _backward_two_wave_kernel(
    grad_q_ptr, grad_k_ptr, q_ptr, k_ptr, embeddings_ptr,
    out_q_ptr, out_k_ptr, out_embeddings_ptr,
):
    token = tl.program_id(0)
    lane_group = tl.arange(0, 2)[:, None]
    d0 = tl.arange(0, HALF_DIM)[None, :]
    d1 = d0 + HALF_DIM

    emb_base = token * HEAD_DIM
    e0 = tl.load(embeddings_ptr + emb_base + d0)
    e1 = tl.load(embeddings_ptr + emb_base + d1)
    c0 = libdevice.cos(e0)
    s0 = libdevice.sin(e0)
    c1 = libdevice.cos(e1)
    s1 = libdevice.sin(e1)

    gc0 = tl.zeros((2, HALF_DIM), tl.float32)
    gs0 = tl.zeros((2, HALF_DIM), tl.float32)
    gc1 = tl.zeros((2, HALF_DIM), tl.float32)
    gs1 = tl.zeros((2, HALF_DIM), tl.float32)
    token_base = token * HEADS * HEAD_DIM

    # Stage 0 computes A0 and A3 in parallel.  Stages 1 and 2 are
    # active only in lane group 0, yielding ((A0 + A1) + A2, A3).
    for stage in tl.static_range(3):
        pc0 = tl.zeros((2, HALF_DIM), tl.float32)
        ps0 = tl.zeros((2, HALF_DIM), tl.float32)
        pc1 = tl.zeros((2, HALF_DIM), tl.float32)
        ps1 = tl.zeros((2, HALF_DIM), tl.float32)
        for group in tl.static_range(4):
            if stage == 0:
                head = tl.where(lane_group == 0, 0, 3) + group * 4
                active = lane_group < 2
            else:
                head = stage + group * 4 + lane_group * 0
                active = lane_group == 0
            base = token_base + head * HEAD_DIM
            gq0 = tl.load(grad_q_ptr + base + d0, mask=active, other=0.0)
            gq1 = tl.load(grad_q_ptr + base + d1, mask=active, other=0.0)
            gk0 = tl.load(grad_k_ptr + base + d0, mask=active, other=0.0)
            gk1 = tl.load(grad_k_ptr + base + d1, mask=active, other=0.0)
            q0 = tl.load(q_ptr + base + d0, mask=active, other=0.0)
            q1 = tl.load(q_ptr + base + d1, mask=active, other=0.0)
            k0 = tl.load(k_ptr + base + d0, mask=active, other=0.0)
            k1 = tl.load(k_ptr + base + d1, mask=active, other=0.0)

            oq0 = _rounded_add(gq0 * c0, gq1 * s1)
            oq1 = _rounded_add(gq1 * c1, -(gq0 * s0))
            ok0 = _rounded_add(gk0 * c0, gk1 * s1)
            ok1 = _rounded_add(gk1 * c1, -(gk0 * s0))
            tl.store(out_q_ptr + base + d0, oq0, mask=active)
            tl.store(out_q_ptr + base + d1, oq1, mask=active)
            tl.store(out_k_ptr + base + d0, ok0, mask=active)
            tl.store(out_k_ptr + base + d1, ok1, mask=active)

            pc0 += _rounded_add(gq0 * q0, gk0 * k0)
            pc1 += _rounded_add(gq1 * q1, gk1 * k1)
            ps0 += _rounded_add(gq0 * (-q1), gk0 * (-k1))
            ps1 += _rounded_add(gq1 * q0, gk1 * k0)

        gc0 = _rounded_add(gc0, pc0)
        gs0 = _rounded_add(gs0, ps0)
        gc1 = _rounded_add(gc1, pc1)
        gs1 = _rounded_add(gs1, ps1)

    gc0 = tl.sum(gc0, axis=0)
    gs0 = tl.sum(gs0, axis=0)
    gc1 = tl.sum(gc1, axis=0)
    gs1 = tl.sum(gs1, axis=0)
    out_e0 = _rounded_add(gc0 * (-s0), gs0 * c0)
    out_e1 = _rounded_add(gc1 * (-s1), gs1 * c1)
    tl.store(out_embeddings_ptr + emb_base + d0, out_e0)
    tl.store(out_embeddings_ptr + emb_base + d1, out_e1)


@torch.no_grad()
def run(grad_q_embed, grad_k_embed, q, k, embeddings):
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    out_embeddings = torch.empty_like(embeddings)
    seq_len = q.shape[0]
    args = (
        grad_q_embed, grad_k_embed, q, k, embeddings,
        out_q, out_k, out_embeddings,
    )
    if seq_len <= 1024:
        waves = 1 if seq_len <= 256 else (2 if seq_len <= 512 else 4)
        _backward_parallel_kernel[(seq_len,)](
            *args, num_warps=4, waves_per_eu=waves,
        )
    else:
        if seq_len <= 3200:
            waves = 5
        elif seq_len <= 3900:
            waves = 8
        else:
            waves = 2
        _backward_kernel[(seq_len,)](
            *args, num_warps=1, waves_per_eu=waves,
        )
    return out_q, out_k, out_embeddings
