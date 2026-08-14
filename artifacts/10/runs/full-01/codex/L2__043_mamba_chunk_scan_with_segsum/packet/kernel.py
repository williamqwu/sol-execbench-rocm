import torch
import triton
import triton.language as tl


H = 16
DSTATE = 256
CHUNK = 256
KH = tl.constexpr(16)
KS = tl.constexpr(256)
KC = tl.constexpr(256)


@triton.jit
def _chunk_contrib_kernel(
    X, A, B, partial, chunk_decay,
    n_tokens, n_chunks,
    BLOCK_D: tl.constexpr, UNROLL: tl.constexpr,
):
    # One program owns BLOCK_D rows of one (batch, head, chunk) state.
    p = tl.program_id(0)
    c = tl.program_id(1)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt

    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = tl.arange(0, KS)[None, :]
    start = c * KC
    length = tl.minimum(n_tokens - start, KC)

    state = tl.zeros((BLOCK_D, KS), tl.float32)
    total = 0.0
    for k in tl.range(0, length, loop_unroll_factor=UNROLL):
        t = start + k
        av = tl.load(A + (b * KH + h) * n_tokens + t).to(tl.float32)
        total += av
        xv = tl.load(X + ((b * n_tokens + t) * KH + h) * 64 + dd,
                     ).to(tl.float32)
        bv = tl.load(B + (b * n_tokens + t) * KS + ss,
                     ).to(tl.float32)
        state = state * tl.exp(av) + xv * bv

    po = ((((b * n_chunks + c) * KH + h) * 64 + dd) * KS + ss)
    tl.store(partial + po, state)
    # Exactly one d-tile publishes this chunk's scalar decay.
    tl.store(chunk_decay + (b * KH + h) * n_chunks + c, tl.exp(total), mask=dt == 0)


@triton.jit
def _propagate_kernel(
    partial, chunk_decay, initial, final,
    n_chunks,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt
    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = tl.arange(0, KS)[None, :]

    io = ((b * KH + h) * 64 + dd) * KS + ss
    state = tl.load(initial + io).to(tl.float32)
    for c in tl.range(0, n_chunks, loop_unroll_factor=1):
        po = ((((b * n_chunks + c) * KH + h) * 64 + dd) * KS + ss)
        contribution = tl.load(partial + po)
        # Reuse the scratch buffer for the state entering each chunk.
        tl.store(partial + po, state)
        decay = tl.load(chunk_decay + (b * KH + h) * n_chunks + c)
        state = state * decay + contribution
    tl.store(final + io, state)


@triton.jit
def _output_kernel(
    X, A, B, C, D, states_in, Out,
    n_tokens, n_chunks,
    BLOCK_D: tl.constexpr, UNROLL: tl.constexpr,
):
    p = tl.program_id(0)
    c = tl.program_id(1)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt
    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = tl.arange(0, KS)[None, :]

    po = ((((b * n_chunks + c) * KH + h) * 64 + dd) * KS + ss)
    state = tl.load(states_in + po)
    dh = tl.load(D + h).to(tl.float32)
    start = c * KC
    length = tl.minimum(n_tokens - start, KC)
    for k in tl.range(0, length, loop_unroll_factor=UNROLL):
        t = start + k
        av = tl.load(A + (b * KH + h) * n_tokens + t,
                     ).to(tl.float32)
        xv = tl.load(X + ((b * n_tokens + t) * KH + h) * 64 + dd,
                     ).to(tl.float32)
        bv = tl.load(B + (b * n_tokens + t) * KS + ss,
                     ).to(tl.float32)
        cv = tl.load(C + (b * n_tokens + t) * KS + ss,
                     ).to(tl.float32)
        state = state * tl.exp(av) + xv * bv
        y = tl.sum(state * cv, axis=1)[:, None] + dh * xv
        oo = (b * n_tokens + t) * (KH * 64) + h * 64 + dd
        tl.store(Out + oo, y)


@triton.jit
def _chunk_contrib_split_kernel(
    X, A, B, partial, chunk_decay,
    n_tokens, n_chunks,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, UNROLL: tl.constexpr,
):
    p = tl.program_id(0)
    c = tl.program_id(1)
    st = tl.program_id(2)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt
    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = st * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
    start = c * KC
    length = tl.minimum(n_tokens - start, KC)

    state = tl.zeros((BLOCK_D, BLOCK_S), tl.float32)
    total = 0.0
    for k in tl.range(0, length, loop_unroll_factor=UNROLL):
        t = start + k
        av = tl.load(A + (b * KH + h) * n_tokens + t,
                     ).to(tl.float32)
        total += av
        xv = tl.load(X + ((b * n_tokens + t) * KH + h) * 64 + dd,
                     ).to(tl.float32)
        bv = tl.load(B + (b * n_tokens + t) * KS + ss,
                     ).to(tl.float32)
        state = state * tl.exp(av) + xv * bv

    po = ((((b * n_chunks + c) * KH + h) * 64 + dd) * KS + ss)
    tl.store(partial + po, state)
    tl.store(chunk_decay + (b * KH + h) * n_chunks + c, tl.exp(total),
             mask=(dt == 0) & (st == 0))


@triton.jit
def _propagate_split_kernel(
    partial, chunk_decay, initial, final,
    n_chunks,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    p = tl.program_id(0)
    st = tl.program_id(1)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt
    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = st * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
    io = ((b * KH + h) * 64 + dd) * KS + ss
    state = tl.load(initial + io).to(tl.float32)
    for c in tl.range(0, n_chunks, loop_unroll_factor=1):
        po = ((((b * n_chunks + c) * KH + h) * 64 + dd) * KS + ss)
        contribution = tl.load(partial + po)
        tl.store(partial + po, state)
        decay = tl.load(chunk_decay + (b * KH + h) * n_chunks + c)
        state = state * decay + contribution
    tl.store(final + io, state)


@triton.jit
def _output_split_kernel(
    X, A, B, C, states_in, Parts,
    n_tokens, n_chunks,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, N_ST: tl.constexpr,
    UNROLL: tl.constexpr,
):
    p = tl.program_id(0)
    c = tl.program_id(1)
    st = tl.program_id(2)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt
    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = st * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]

    po = ((((b * n_chunks + c) * KH + h) * 64 + dd) * KS + ss)
    state = tl.load(states_in + po)
    start = c * KC
    length = tl.minimum(n_tokens - start, KC)
    for k in tl.range(0, length, loop_unroll_factor=UNROLL):
        t = start + k
        av = tl.load(A + (b * KH + h) * n_tokens + t,
                     ).to(tl.float32)
        xv = tl.load(X + ((b * n_tokens + t) * KH + h) * 64 + dd,
                     ).to(tl.float32)
        bv = tl.load(B + (b * n_tokens + t) * KS + ss,
                     ).to(tl.float32)
        cv = tl.load(C + (b * n_tokens + t) * KS + ss,
                     ).to(tl.float32)
        state = state * tl.exp(av) + xv * bv
        y = tl.sum(state * cv, axis=1)[:, None]
        oo = ((b * n_tokens + t) * (KH * 64) + h * 64 + dd) * N_ST + st
        tl.store(Parts + oo, y)


@triton.jit
def _finish_split_kernel(X, D, Parts, Out, n_elements: tl.constexpr,
                         N_ST: tl.constexpr):
    o = tl.program_id(0) * 256 + tl.arange(0, 256)[:, None]
    mask = o < n_elements
    st = tl.arange(0, N_ST)[None, :]
    vals = tl.load(Parts + o * N_ST + st, mask=mask)
    value = tl.sum(vals, axis=1)[:, None]
    h = (o // 64) % KH
    xv = tl.load(X + o, mask=mask).to(tl.float32)
    dh = tl.load(D + h, mask=mask).to(tl.float32)
    y = value + dh * xv
    tl.store(Out + o, y, mask=mask)


@triton.jit
def _scan_single_kernel(
    X, A, B, C, D, initial, Out, final, n_tokens,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt
    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = tl.arange(0, KS)[None, :]
    io = ((b * KH + h) * 64 + dd) * KS + ss
    state = tl.load(initial + io).to(tl.float32)
    dh = tl.load(D + h).to(tl.float32)
    for t in tl.range(0, n_tokens, loop_unroll_factor=4):
        av = tl.load(A + (b * KH + h) * n_tokens + t).to(tl.float32)
        xv = tl.load(X + ((b * n_tokens + t) * KH + h) * 64 + dd).to(tl.float32)
        bv = tl.load(B + (b * n_tokens + t) * KS + ss).to(tl.float32)
        cv = tl.load(C + (b * n_tokens + t) * KS + ss).to(tl.float32)
        state = state * tl.exp(av) + xv * bv
        y = tl.sum(state * cv, axis=1)[:, None] + dh * xv
        oo = (b * n_tokens + t) * (KH * 64) + h * 64 + dd
        tl.store(Out + oo, y)
    tl.store(final + io, state)


@triton.jit
def _scan_single_split_kernel(
    X, A, B, C, initial, Parts, final, n_tokens,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, N_ST: tl.constexpr,
    UNROLL: tl.constexpr,
):
    p = tl.program_id(0)
    st = tl.program_id(1)
    n_dt = 64 // BLOCK_D
    b = p // (KH * n_dt)
    r = p - b * KH * n_dt
    h = r // n_dt
    dt = r - h * n_dt
    dd = dt * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    ss = st * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
    io = ((b * KH + h) * 64 + dd) * KS + ss
    state = tl.load(initial + io).to(tl.float32)
    for t in tl.range(0, n_tokens, loop_unroll_factor=UNROLL):
        av = tl.load(A + (b * KH + h) * n_tokens + t).to(tl.float32)
        xv = tl.load(X + ((b * n_tokens + t) * KH + h) * 64 + dd).to(tl.float32)
        bv = tl.load(B + (b * n_tokens + t) * KS + ss).to(tl.float32)
        cv = tl.load(C + (b * n_tokens + t) * KS + ss).to(tl.float32)
        state = state * tl.exp(av) + xv * bv
        y = tl.sum(state * cv, axis=1)[:, None]
        oo = ((b * n_tokens + t) * (KH * 64) + h * 64 + dd) * N_ST + st
        tl.store(Parts + oo, y)
    tl.store(final + io, state)


@torch.no_grad()
def run(hidden_states, A, B, C, D, initial_states):
    batch, n_tokens, _, _ = hidden_states.shape
    n_chunks = (n_tokens + CHUNK - 1) // CHUNK
    out = torch.empty(
        (batch, n_tokens, H * 64), device=hidden_states.device, dtype=torch.bfloat16
    )
    final = torch.empty_like(initial_states)

    # Once batch parallelism is high enough, a direct scan is cheaper than
    # constructing and revisiting chunk boundaries.  Low-batch long sequences
    # still take the parallel chunk path below.
    direct_scan = (
        n_chunks == 1
        or (batch == 2 and n_tokens < 384)
        or (batch == 4 and n_tokens < 640)
        or (batch >= 8 and n_tokens <= 1024)
    )
    if direct_scan:
        if batch <= 4:
            block_s, n_st, block_d, n_warps = 64, 4, (4 if batch == 1 else 16), 4
        elif batch <= 8:
            block_s, n_st, block_d, n_warps = 128, 2, 16, 4
        else:
            block_s, n_st, block_d, n_warps = 128, 2, 64, 8
        if n_tokens == 256:
            unroll = 1 if (batch == 1 or batch >= 32) else 2
        else:
            unroll = 2 if batch >= 16 else 4
        parts = torch.empty(
            (batch, n_tokens, H * 64, n_st),
            device=hidden_states.device, dtype=torch.float32,
        )
        _scan_single_split_kernel[(batch * H * (64 // block_d), n_st)](
            hidden_states, A, B, C, initial_states, parts, final, n_tokens,
            BLOCK_D=block_d, BLOCK_S=block_s, N_ST=n_st, UNROLL=unroll,
            num_warps=n_warps,
        )
        n_elements = batch * n_tokens * H * 64
        _finish_split_kernel[(triton.cdiv(n_elements, 256),)](
            hidden_states, D, parts, out,
            n_elements=n_elements, N_ST=n_st, num_warps=4,
        )
        return out, final

    # Float scratch is intentional: every corresponding reference
    # intermediate remains float32 until the two returned tensors.
    partial = torch.empty(
        (batch, n_chunks, H, 64, DSTATE),
        device=hidden_states.device, dtype=torch.float32,
    )
    chunk_decay = torch.empty(
        (batch, H, n_chunks), device=hidden_states.device, dtype=torch.float32
    )
    parallel_chunks = batch * n_chunks

    # Small grids benefit from state-size partitioning: 64-wide reductions
    # stay within a wave.  Larger grids use one 256-wide state tile to avoid
    # extra programs and partial-output traffic.
    if parallel_chunks <= 16:
        if parallel_chunks <= 4:
            block_s, n_st, block_d, n_warps = 64, 4, 16, 4
        elif parallel_chunks <= 12:
            block_s, n_st, block_d, n_warps = 128, 2, 16, 4
        else:
            block_s, n_st, block_d, n_warps = 128, 2, 64, 8
        n_dt = 64 // block_d
        parts = torch.empty(
            (batch, n_tokens, H * 64, n_st),
            device=hidden_states.device, dtype=torch.float32,
        )
        grid = (batch * H * n_dt, n_chunks, n_st)
        _chunk_contrib_split_kernel[grid](
            hidden_states, A, B, partial, chunk_decay, n_tokens, n_chunks,
            BLOCK_D=block_d, BLOCK_S=block_s, UNROLL=4, num_warps=n_warps,
        )
        _propagate_split_kernel[(batch * H * n_dt, n_st)](
            partial, chunk_decay, initial_states, final, n_chunks,
            BLOCK_D=block_d, BLOCK_S=block_s, num_warps=n_warps,
        )
        _output_split_kernel[grid](
            hidden_states, A, B, C, partial, parts, n_tokens, n_chunks,
            BLOCK_D=block_d, BLOCK_S=block_s, N_ST=n_st, UNROLL=4,
            num_warps=n_warps,
        )
        n_elements = batch * n_tokens * H * 64
        _finish_split_kernel[(triton.cdiv(n_elements, 256),)](
            hidden_states, D, parts, out,
            n_elements=n_elements, N_ST=n_st, num_warps=4,
        )
    else:
        block_d, n_warps = 32, 8
        n_dt = 64 // block_d
        grid = (batch * H * n_dt, n_chunks)
        _chunk_contrib_kernel[grid](
            hidden_states, A, B, partial, chunk_decay, n_tokens, n_chunks,
            BLOCK_D=block_d, UNROLL=2, num_warps=n_warps,
        )
        _propagate_kernel[(batch * H * n_dt,)](
            partial, chunk_decay, initial_states, final, n_chunks,
            BLOCK_D=block_d, num_warps=n_warps,
        )
        _output_kernel[grid](
            hidden_states, A, B, C, D, partial, out, n_tokens, n_chunks,
            BLOCK_D=block_d, UNROLL=2, num_warps=n_warps,
        )
    return out, final
