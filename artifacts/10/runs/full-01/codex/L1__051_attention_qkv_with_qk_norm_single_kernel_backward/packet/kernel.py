import torch
import triton
import triton.language as tl


DIRECT_NORM_LIMIT = 1025
PARTIAL_FUSION_LIMIT = 4097


@triton.jit
def _pack_grads(
    grad_q,
    grad_k,
    grad_v,
    q_weight,
    k_weight,
    q_normed,
    k_normed,
    q_rstd,
    k_rstd,
    packed,
    M: tl.constexpr,
    S: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, 256)[None, :]
    rq = 4 * M
    nq = tl.cdiv(rq, BLOCK_R)

    if pid < nq:
        # The source is [B, H, S, D], while the packed GEMM operand is
        # [B, S, H*D | K*D | V*D].
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
        rmask = r < rq
        b = r // (4 * S)
        hs = r - b * (4 * S)
        h = hs // S
        s = hs - h * S
        dst = (b * S + s) * 1536 + h * 256 + d

        g = tl.load(grad_q + r * 256 + d, mask=rmask).to(tl.float32)
        n = tl.load(q_normed + r * 256 + d, mask=rmask).to(tl.float32)
        scale = 1.0 + tl.load(q_weight + d).to(tl.float32)
        # This conversion is a required intermediate rounding point in the
        # reference implementation.
        gs = (g * scale).to(tl.bfloat16).to(tl.float32)
        mean = tl.sum(gs * n, axis=1)[:, None] * 0.00390625
        rs = tl.load(q_rstd + r, mask=rmask)
        tl.store(packed + dst, rs * (gs - mean * n), mask=rmask)

    else:
        r = (pid - nq) * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
        rmask = r < M
        dbase = r * 1536
        g = tl.load(grad_k + r * 256 + d, mask=rmask).to(tl.float32)
        n = tl.load(k_normed + r * 256 + d, mask=rmask).to(tl.float32)
        scale = 1.0 + tl.load(k_weight + d).to(tl.float32)
        gs = (g * scale).to(tl.bfloat16).to(tl.float32)
        mean = tl.sum(gs * n, axis=1)[:, None] * 0.00390625
        rs = tl.load(k_rstd + r, mask=rmask)
        tl.store(packed + dbase + 1024 + d, rs * (gs - mean * n), mask=rmask)
        v = tl.load(grad_v + r * 256 + d, mask=rmask)
        tl.store(packed + dbase + 1280 + d, v, mask=rmask)


@triton.jit
def _norm_weight_grad_partials(
    grad_q,
    grad_k,
    q_normed,
    k_normed,
    partials,
    RQ: tl.constexpr,
    RK: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rb = tl.program_id(0)
    db = tl.program_id(1)
    rows = rb * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    ds = db * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]

    if rb < NQ:
        mask = rows < RQ
        offs = rows * 256 + ds
        g = tl.load(grad_q + offs, mask=mask).to(tl.float32)
        n = tl.load(q_normed + offs, mask=mask).to(tl.float32)
        part = tl.sum(g * n, axis=0)
        dout = db * BLOCK_D + tl.arange(0, BLOCK_D)
        tl.store(partials + dout * NQ + rb, part)
    else:
        kb = rb - NQ
        rows = kb * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
        mask = rows < RK
        offs = rows * 256 + ds
        g = tl.load(grad_k + offs, mask=mask).to(tl.float32)
        n = tl.load(k_normed + offs, mask=mask).to(tl.float32)
        part = tl.sum(g * n, axis=0)
        dout = db * BLOCK_D + tl.arange(0, BLOCK_D)
        tl.store(partials + 256 * NQ + dout * NK + kb, part)


@triton.jit
def _finish_norm_weight_grads(
    partials,
    outputs,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_N)
    if pid < 256:
        vals = tl.load(partials + pid * NQ + chunks, mask=chunks < NQ)
    else:
        d = pid - 256
        vals = tl.load(partials + 256 * NQ + d * NK + chunks, mask=chunks < NK)
    tl.store(outputs + pid, tl.sum(vals, axis=0))


@triton.jit
def _direct_norm_weight_grads(
    grad_q,
    grad_k,
    q_normed,
    k_normed,
    outputs,
    RQ: tl.constexpr,
    RK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    if pid < 256:
        g = tl.load(grad_q + rows * 256 + pid, mask=rows < RQ).to(tl.float32)
        n = tl.load(q_normed + rows * 256 + pid, mask=rows < RQ).to(tl.float32)
        value = tl.sum(g * n, axis=0)
    else:
        d = pid - 256
        g = tl.load(grad_k + rows * 256 + d, mask=rows < RK).to(tl.float32)
        n = tl.load(k_normed + rows * 256 + d, mask=rows < RK).to(tl.float32)
        value = tl.sum(g * n, axis=0)
    tl.store(outputs + pid, value)


@triton.jit
def _direct_norm_weight_grads_tiled(
    grad_q,
    grad_k,
    q_normed,
    k_normed,
    outputs,
    RQ: tl.constexpr,
    RK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    nd = tl.cdiv(256, BLOCK_D)
    ds0 = (pid % nd) * BLOCK_D
    ds = ds0 + tl.arange(0, BLOCK_D)[None, :]
    acc = tl.zeros((BLOCK_D,), tl.float32)
    if pid < nd:
        for r0 in range(0, RQ, BLOCK_R):
            rows = r0 + tl.arange(0, BLOCK_R)[:, None]
            mask = rows < RQ
            offs = rows * 256 + ds
            g = tl.load(grad_q + offs, mask=mask).to(tl.float32)
            n = tl.load(q_normed + offs, mask=mask).to(tl.float32)
            acc += tl.sum(g * n, axis=0)
    else:
        for r0 in range(0, RK, BLOCK_R):
            rows = r0 + tl.arange(0, BLOCK_R)[:, None]
            mask = rows < RK
            offs = rows * 256 + ds
            g = tl.load(grad_k + offs, mask=mask).to(tl.float32)
            n = tl.load(k_normed + offs, mask=mask).to(tl.float32)
            acc += tl.sum(g * n, axis=0)
    tl.store(outputs + ds0 + tl.arange(0, BLOCK_D) + (pid >= nd) * 256, acc)


@triton.jit
def _hidden_gemm(
    packed,
    q_weight,
    k_weight,
    v_weight,
    output,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    im = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    jn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, 1024, BLOCK_K):
        a = tl.load(
            packed + im[:, None] * 1536 + k0 + kk[None, :],
            mask=im[:, None] < M,
        )
        b = tl.load(
            q_weight + (k0 + kk[:, None]) * 640 + jn[None, :],
            mask=jn[None, :] < 640,
        )
        acc += tl.dot(a, b)

    for k0 in range(0, 256, BLOCK_K):
        a = tl.load(
            packed + im[:, None] * 1536 + 1024 + k0 + kk[None, :],
            mask=im[:, None] < M,
        )
        b = tl.load(
            k_weight + (k0 + kk[:, None]) * 640 + jn[None, :],
            mask=jn[None, :] < 640,
        )
        acc += tl.dot(a, b)

    for k0 in range(0, 256, BLOCK_K):
        a = tl.load(
            packed + im[:, None] * 1536 + 1280 + k0 + kk[None, :],
            mask=im[:, None] < M,
        )
        b = tl.load(
            v_weight + (k0 + kk[:, None]) * 640 + jn[None, :],
            mask=jn[None, :] < 640,
        )
        acc += tl.dot(a, b)

    tl.store(
        output + im[:, None] * 640 + jn[None, :],
        acc,
        mask=(im[:, None] < M) & (jn[None, :] < 640),
    )


@triton.jit
def _hidden_gemm_and_finish_norm(
    packed,
    q_weight,
    k_weight,
    v_weight,
    partials,
    norm_outputs,
    output,
    M: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    if pm < NUM_M_BLOCKS:
        im = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, 1024, BLOCK_K):
            a = tl.load(
                packed + im[:, None] * 1536 + k0 + kk[None, :],
                mask=im[:, None] < M,
            )
            b = tl.load(
                q_weight + (k0 + kk[:, None]) * 640 + jn[None, :],
                mask=jn[None, :] < 640,
            )
            acc += tl.dot(a, b)
        for k0 in range(0, 256, BLOCK_K):
            a = tl.load(
                packed + im[:, None] * 1536 + 1024 + k0 + kk[None, :],
                mask=im[:, None] < M,
            )
            b = tl.load(
                k_weight + (k0 + kk[:, None]) * 640 + jn[None, :],
                mask=jn[None, :] < 640,
            )
            acc += tl.dot(a, b)
        for k0 in range(0, 256, BLOCK_K):
            a = tl.load(
                packed + im[:, None] * 1536 + 1280 + k0 + kk[None, :],
                mask=im[:, None] < M,
            )
            b = tl.load(
                v_weight + (k0 + kk[:, None]) * 640 + jn[None, :],
                mask=jn[None, :] < 640,
            )
            acc += tl.dot(a, b)
        tl.store(
            output + im[:, None] * 640 + jn[None, :],
            acc,
            mask=(im[:, None] < M) & (jn[None, :] < 640),
        )
    else:
        pid = (pm - NUM_M_BLOCKS) * NUM_N_BLOCKS + pn
        if pid < 512:
            chunks = tl.arange(0, BLOCK_CHUNKS)
            if pid < 256:
                vals = tl.load(partials + pid * NQ + chunks, mask=chunks < NQ)
            else:
                d = pid - 256
                vals = tl.load(partials + 256 * NQ + d * NK + chunks, mask=chunks < NK)
            tl.store(norm_outputs + pid, tl.sum(vals, axis=0))


@triton.jit
def _weight_gemm(
    packed,
    hidden,
    output,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    im = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    jn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ki = k0 + kk
        a = tl.load(
            packed + im[:, None] + ki[None, :] * 1536,
            mask=(im[:, None] < 1536) & (ki[None, :] < K),
        )
        b = tl.load(
            hidden + ki[:, None] * 640 + jn[None, :],
            mask=(ki[:, None] < K) & (jn[None, :] < 640),
        )
        acc += tl.dot(a, b)

    tl.store(
        output + im[:, None] * 640 + jn[None, :],
        acc,
        mask=(im[:, None] < 1536) & (jn[None, :] < 640),
    )


@triton.jit
def _weight_gemm_and_norm(
    packed,
    hidden,
    grad_q,
    grad_k,
    q_normed,
    k_normed,
    weight_output,
    norm_output,
    K: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NORM_BLOCK_R: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)

    if pm < NUM_M_BLOCKS:
        im = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ki = k0 + kk
            a = tl.load(
                packed + im[:, None] + ki[None, :] * 1536,
                mask=(im[:, None] < 1536) & (ki[None, :] < K),
            )
            b = tl.load(
                hidden + ki[:, None] * 640 + jn[None, :],
                mask=(ki[:, None] < K) & (jn[None, :] < 640),
            )
            acc += tl.dot(a, b)
        tl.store(
            weight_output + im[:, None] * 640 + jn[None, :],
            acc,
            mask=(im[:, None] < 1536) & (jn[None, :] < 640),
        )
    else:
        pid = (pm - NUM_M_BLOCKS) * NUM_N_BLOCKS + pn
        if pid < 512:
            rows = tl.arange(0, NORM_BLOCK_R)
            if pid < 256:
                g = tl.load(grad_q + rows * 256 + pid, mask=rows < RQ).to(tl.float32)
                n = tl.load(q_normed + rows * 256 + pid, mask=rows < RQ).to(tl.float32)
                value = tl.sum(g * n, axis=0)
            else:
                d = pid - 256
                g = tl.load(grad_k + rows * 256 + d, mask=rows < RK).to(tl.float32)
                n = tl.load(k_normed + rows * 256 + d, mask=rows < RK).to(tl.float32)
                value = tl.sum(g * n, axis=0)
            tl.store(norm_output + pid, value)


@triton.jit
def _weight_gemm_and_norm_partials(
    packed,
    hidden,
    grad_q,
    grad_k,
    q_normed,
    k_normed,
    weight_output,
    partials,
    K: tl.constexpr,
    RQ: tl.constexpr,
    RK: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REDUCE_BLOCK_R: tl.constexpr,
    REDUCE_BLOCK_D: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    if pm < NUM_M_BLOCKS:
        im = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        jn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ki = k0 + kk
            a = tl.load(
                packed + im[:, None] + ki[None, :] * 1536,
                mask=(im[:, None] < 1536) & (ki[None, :] < K),
            )
            b = tl.load(
                hidden + ki[:, None] * 640 + jn[None, :],
                mask=(ki[:, None] < K) & (jn[None, :] < 640),
            )
            acc += tl.dot(a, b)
        tl.store(
            weight_output + im[:, None] * 640 + jn[None, :],
            acc,
            mask=(im[:, None] < 1536) & (jn[None, :] < 640),
        )
    else:
        flat = (pm - NUM_M_BLOCKS) * NUM_N_BLOCKS + pn
        rb = flat // 8
        db = flat - rb * 8
        if rb < NQ + NK:
            ds = db * REDUCE_BLOCK_D + tl.arange(0, REDUCE_BLOCK_D)[None, :]
            if rb < NQ:
                rows = rb * REDUCE_BLOCK_R + tl.arange(0, REDUCE_BLOCK_R)[:, None]
                mask = rows < RQ
                offs = rows * 256 + ds
                g = tl.load(grad_q + offs, mask=mask).to(tl.float32)
                n = tl.load(q_normed + offs, mask=mask).to(tl.float32)
                part = tl.sum(g * n, axis=0)
                dout = db * REDUCE_BLOCK_D + tl.arange(0, REDUCE_BLOCK_D)
                tl.store(partials + dout * NQ + rb, part)
            else:
                kb = rb - NQ
                rows = kb * REDUCE_BLOCK_R + tl.arange(0, REDUCE_BLOCK_R)[:, None]
                mask = rows < RK
                offs = rows * 256 + ds
                g = tl.load(grad_k + offs, mask=mask).to(tl.float32)
                n = tl.load(k_normed + offs, mask=mask).to(tl.float32)
                part = tl.sum(g * n, axis=0)
                dout = db * REDUCE_BLOCK_D + tl.arange(0, REDUCE_BLOCK_D)
                tl.store(partials + 256 * NQ + dout * NK + kb, part)


@torch.no_grad()
def run(
    grad_query,
    grad_key,
    grad_value,
    hidden_states,
    q_weight,
    k_weight,
    v_weight,
    q_norm_weight,
    k_norm_weight,
    query_transposed,
    key_transposed,
    q_rstd,
    k_rstd,
    q_normed,
    k_normed,
    rms_norm_eps,
):
    batch, seq_len, _ = hidden_states.shape
    m = batch * seq_len

    packed = torch.empty((m, 1536), device=hidden_states.device, dtype=torch.bfloat16)
    norm_grads = torch.empty((512,), device=hidden_states.device, dtype=torch.float32)

    pack_block_r = 16
    _pack_grads[(triton.cdiv(4 * m, pack_block_r) + triton.cdiv(m, pack_block_r),)](
        grad_query,
        grad_key,
        grad_value,
        q_norm_weight,
        k_norm_weight,
        q_normed,
        k_normed,
        q_rstd,
        k_rstd,
        packed,
        M=m,
        S=seq_len,
        BLOCK_R=pack_block_r,
        num_warps=8,
    )

    rq = 4 * m
    rk = m
    fuse_weight_and_norm = m < DIRECT_NORM_LIMIT
    fuse_weight_and_partials = (not fuse_weight_and_norm) and m < PARTIAL_FUSION_LIMIT
    if not fuse_weight_and_norm:
        block_r = 256
        block_d = 32
        nq = triton.cdiv(rq, block_r)
        nk = triton.cdiv(rk, block_r)
        partials = torch.empty((256 * (nq + nk),), device=hidden_states.device, dtype=torch.float32)
        if fuse_weight_and_partials:
            grad_weights = torch.empty((1536, 640), device=hidden_states.device, dtype=torch.bfloat16)
            weight_block_m = 64
            weight_block_n = 64
            weight_m_blocks = triton.cdiv(1536, weight_block_m)
            weight_n_blocks = triton.cdiv(640, weight_block_n)
            reduce_programs = (nq + nk) * 8
            extra_m_blocks = triton.cdiv(reduce_programs, weight_n_blocks)
            _weight_gemm_and_norm_partials[(weight_m_blocks + extra_m_blocks, weight_n_blocks)](
                packed,
                hidden_states,
                grad_query,
                grad_key,
                q_normed,
                k_normed,
                grad_weights,
                partials,
                K=m,
                RQ=rq,
                RK=rk,
                NQ=nq,
                NK=nk,
                BLOCK_M=weight_block_m,
                BLOCK_N=weight_block_n,
                BLOCK_K=64,
                REDUCE_BLOCK_R=block_r,
                REDUCE_BLOCK_D=block_d,
                NUM_M_BLOCKS=weight_m_blocks,
                NUM_N_BLOCKS=weight_n_blocks,
                num_warps=4,
                num_stages=3 if m < 1800 else 4,
            )
        else:
            _norm_weight_grad_partials[(nq + nk, 8)](
                grad_query,
                grad_key,
                q_normed,
                k_normed,
                partials,
                RQ=rq,
                RK=rk,
                NQ=nq,
                NK=nk,
                BLOCK_R=block_r,
                BLOCK_D=block_d,
                num_warps=4,
            )

    grad_hidden = torch.empty((m, 640), device=hidden_states.device, dtype=torch.bfloat16)
    if m <= 192:
        gemm_block_m, gemm_block_n, gemm_warps = 16, 32, 4
    elif m <= 1024:
        gemm_block_m, gemm_block_n, gemm_warps = 16, 64, 4
    elif m < 1800:
        gemm_block_m, gemm_block_n, gemm_warps = 128, 64, 4
    elif m <= 4096:
        gemm_block_m, gemm_block_n, gemm_warps = 64, 64, 4
    else:
        gemm_block_m, gemm_block_n, gemm_warps = 128, 128, 8
    hidden_m_blocks = triton.cdiv(m, gemm_block_m)
    hidden_n_blocks = triton.cdiv(640, gemm_block_n)
    if fuse_weight_and_norm:
        _hidden_gemm[(hidden_m_blocks, hidden_n_blocks)](
            packed,
            q_weight,
            k_weight,
            v_weight,
            grad_hidden,
            M=m,
            BLOCK_M=gemm_block_m,
            BLOCK_N=gemm_block_n,
            BLOCK_K=64,
            num_warps=gemm_warps,
        )
    else:
        hidden_extra_blocks = triton.cdiv(512, hidden_n_blocks)
        _hidden_gemm_and_finish_norm[(hidden_m_blocks + hidden_extra_blocks, hidden_n_blocks)](
            packed,
            q_weight,
            k_weight,
            v_weight,
            partials,
            norm_grads,
            grad_hidden,
            M=m,
            NQ=nq,
            NK=nk,
            NUM_M_BLOCKS=hidden_m_blocks,
            NUM_N_BLOCKS=hidden_n_blocks,
            BLOCK_M=gemm_block_m,
            BLOCK_N=gemm_block_n,
            BLOCK_K=64,
            BLOCK_CHUNKS=triton.next_power_of_2(nq),
            num_warps=gemm_warps,
            num_stages=2,
        )
    if fuse_weight_and_norm:
        grad_weights = torch.empty((1536, 640), device=hidden_states.device, dtype=torch.bfloat16)
        weight_block_m = 64
        weight_block_n = 64
        weight_m_blocks = triton.cdiv(1536, weight_block_m)
        weight_n_blocks = triton.cdiv(640, weight_block_n)
        extra_m_blocks = triton.cdiv(512, weight_n_blocks)
        _weight_gemm_and_norm[(weight_m_blocks + extra_m_blocks, weight_n_blocks)](
            packed,
            hidden_states,
            grad_query,
            grad_key,
            q_normed,
            k_normed,
            grad_weights,
            norm_grads,
            K=m,
            RQ=rq,
            RK=rk,
            BLOCK_M=weight_block_m,
            BLOCK_N=weight_block_n,
            BLOCK_K=64,
            NORM_BLOCK_R=triton.next_power_of_2(rq),
            NUM_M_BLOCKS=weight_m_blocks,
            NUM_N_BLOCKS=weight_n_blocks,
            num_warps=4,
        )
    elif not fuse_weight_and_partials:
        grad_weights = torch.empty((1536, 640), device=hidden_states.device, dtype=torch.bfloat16)
        if m <= 8192:
            weight_block_m, weight_block_n = 64, 64
        else:
            weight_block_m, weight_block_n = 128, 32
        _weight_gemm[(triton.cdiv(1536, weight_block_m), triton.cdiv(640, weight_block_n))](
            packed,
            hidden_states,
            grad_weights,
            K=m,
            BLOCK_M=weight_block_m,
            BLOCK_N=weight_block_n,
            BLOCK_K=64,
            num_warps=4,
            num_stages=4,
        )

    return (
        grad_hidden.reshape(batch, seq_len, 640),
        grad_weights[:1024],
        grad_weights[1024:1280],
        grad_weights[1280:],
        norm_grads[:256],
        norm_grads[256:],
    )
