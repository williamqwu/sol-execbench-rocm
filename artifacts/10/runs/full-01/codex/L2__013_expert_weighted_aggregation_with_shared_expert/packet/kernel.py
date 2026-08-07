import torch
import triton
import triton.language as tl


HIDDEN = 2048
INTER = 512
N_EXPERTS = 512
TOP_K = 10
_SHARED_STREAM = None


@triton.jit
def _clear_counts(counts, n_experts: tl.constexpr):
    off = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(counts + off, 0, mask=off < n_experts)


@triton.jit
def _make_route_table(selected, counts, route_table, n_routes: tl.constexpr,
                      n_tokens: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = route < n_routes
    expert = tl.load(selected + route, mask=mask, other=0).to(tl.int32)
    pos = tl.atomic_add(counts + expert, 1, mask=mask)
    tl.store(route_table + expert * n_tokens + pos, route, mask=mask)


@triton.jit
def _expert_gate_up(
    x, route_table, counts, gate_w, up_w, mid,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr, inter: tl.constexpr, top_k: tl.constexpr,
    count_low: tl.constexpr, count_high: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    n0 = tl.program_id(0) * BN
    expert = tl.program_id(1)
    ns = n0 + tl.arange(0, BN)
    count = tl.load(counts + expert)
    count = tl.where((count > count_low) & (count <= count_high), count, 0)
    m0 = 0
    while m0 < count:
        ms = m0 + tl.arange(0, BM)
        mmask = ms < count
        route = tl.load(route_table + expert * n_tokens + ms,
                        mask=mmask, other=0).to(tl.int32)
        token = route // top_k
        acc_gate = tl.zeros((BM, BN), tl.float32)
        acc_up = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, hidden, BK):
            ks = k0 + tl.arange(0, BK)
            xv = tl.load(x + token[:, None] * hidden + ks[None, :],
                         mask=mmask[:, None], other=0.0)
            gw = tl.load(gate_w + expert * (inter * hidden)
                         + ks[:, None] + ns[None, :] * hidden)
            uw = tl.load(up_w + expert * (inter * hidden)
                         + ks[:, None] + ns[None, :] * hidden)
            acc_gate += tl.dot(xv, gw)
            acc_up += tl.dot(xv, uw)
        gate = acc_gate.to(tl.bfloat16)
        up = acc_up.to(tl.bfloat16)
        denom = (1.0 + tl.exp(-gate.to(tl.float32))).to(tl.bfloat16)
        silu = (gate / denom).to(tl.bfloat16)
        value = (silu * up).to(tl.bfloat16)
        tl.store(mid + route[:, None] * inter + ns[None, :], value,
                 mask=mmask[:, None])
        m0 += BM


@triton.jit
def _expert_gate_up_joined(
    x, route_table, counts, gate_w, up_w, mid,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr, inter: tl.constexpr, top_k: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    n0 = tl.program_id(0) * BN
    expert = tl.program_id(1)
    # A single dot produces interleaved gate/up column pairs.
    ns = n0 + tl.arange(0, BN)
    count = tl.load(counts + expert)
    m0 = 0
    while m0 < count:
        ms = m0 + tl.arange(0, BM)
        mmask = ms < count
        route = tl.load(route_table + expert * n_tokens + ms,
                        mask=mmask, other=0).to(tl.int32)
        token = route // top_k
        acc = tl.zeros((BM, 2 * BN), tl.float32)
        for k0 in range(0, hidden, BK):
            ks = k0 + tl.arange(0, BK)
            xv = tl.load(x + token[:, None] * hidden + ks[None, :],
                         mask=mmask[:, None], other=0.0)
            gp = tl.load(gate_w + expert * (inter * hidden)
                         + ks[:, None] + ns[None, :] * hidden)
            up = tl.load(up_w + expert * (inter * hidden)
                         + ks[:, None] + ns[None, :] * hidden)
            w = tl.interleave(gp, up)
            acc += tl.dot(xv, w)
        paired = tl.reshape(acc, (BM, BN, 2))
        acc_gate, acc_up = tl.split(paired)
        gate = acc_gate.to(tl.bfloat16)
        upv = acc_up.to(tl.bfloat16)
        denom = (1.0 + tl.exp(-gate.to(tl.float32))).to(tl.bfloat16)
        silu = (gate / denom).to(tl.bfloat16)
        value = (silu * upv).to(tl.bfloat16)
        out_ns = n0 + tl.arange(0, BN)
        tl.store(mid + route[:, None] * inter + out_ns[None, :], value,
                 mask=mmask[:, None])
        m0 += BM


@triton.jit
def _expert_gate_up_pair(
    x, route_table, counts, gate_w, up_w, mid,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr, inter: tl.constexpr, top_k: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    n0 = tl.program_id(0) * BN
    expert = tl.program_id(1)
    ns = n0 + tl.arange(0, BN)
    count = tl.load(counts + expert)
    m0 = 0
    while m0 < count:
        rows = tl.arange(0, BM)
        ms0 = m0 + rows
        ms1 = m0 + BM + rows
        mask0 = ms0 < count
        mask1 = ms1 < count
        route0 = tl.load(route_table + expert * n_tokens + ms0,
                         mask=mask0, other=0).to(tl.int32)
        route1 = tl.load(route_table + expert * n_tokens + ms1,
                         mask=mask1, other=0).to(tl.int32)
        token0 = route0 // top_k
        token1 = route1 // top_k
        ag0 = tl.zeros((BM, BN), tl.float32)
        au0 = tl.zeros((BM, BN), tl.float32)
        ag1 = tl.zeros((BM, BN), tl.float32)
        au1 = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, hidden, BK):
            ks = k0 + tl.arange(0, BK)
            gw = tl.load(gate_w + expert * (inter * hidden)
                         + ks[:, None] + ns[None, :] * hidden)
            uw = tl.load(up_w + expert * (inter * hidden)
                         + ks[:, None] + ns[None, :] * hidden)
            xv0 = tl.load(x + token0[:, None] * hidden + ks[None, :],
                          mask=mask0[:, None], other=0.0)
            ag0 += tl.dot(xv0, gw)
            au0 += tl.dot(xv0, uw)
            if m0 + BM < count:
                xv1 = tl.load(x + token1[:, None] * hidden + ks[None, :],
                              mask=mask1[:, None], other=0.0)
                ag1 += tl.dot(xv1, gw)
                au1 += tl.dot(xv1, uw)
        gate0 = ag0.to(tl.bfloat16)
        up0 = au0.to(tl.bfloat16)
        den0 = (1.0 + tl.exp(-gate0.to(tl.float32))).to(tl.bfloat16)
        val0 = ((gate0 / den0).to(tl.bfloat16) * up0).to(tl.bfloat16)
        tl.store(mid + route0[:, None] * inter + ns[None, :], val0,
                 mask=mask0[:, None])
        if m0 + BM < count:
            gate1 = ag1.to(tl.bfloat16)
            up1 = au1.to(tl.bfloat16)
            den1 = (1.0 + tl.exp(-gate1.to(tl.float32))).to(tl.bfloat16)
            val1 = ((gate1 / den1).to(tl.bfloat16) * up1).to(tl.bfloat16)
            tl.store(mid + route1[:, None] * inter + ns[None, :], val1,
                     mask=mask1[:, None])
        m0 += 2 * BM


@triton.jit
def _expert_down(
    mid, routing_weights, route_table, counts, down_w, route_out,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr, inter: tl.constexpr, top_k: tl.constexpr,
    count_low: tl.constexpr, count_high: tl.constexpr,
    atomic_reduce: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    n0 = tl.program_id(0) * BN
    expert = tl.program_id(1)
    ns = n0 + tl.arange(0, BN)
    count = tl.load(counts + expert)
    count = tl.where((count > count_low) & (count <= count_high), count, 0)
    m0 = 0
    while m0 < count:
        ms = m0 + tl.arange(0, BM)
        mmask = ms < count
        route = tl.load(route_table + expert * n_tokens + ms,
                        mask=mmask, other=0).to(tl.int32)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, inter, BK):
            ks = k0 + tl.arange(0, BK)
            a = tl.load(mid + route[:, None] * inter + ks[None, :],
                        mask=mmask[:, None], other=0.0)
            b = tl.load(down_w + expert * (hidden * inter)
                        + ks[:, None] + ns[None, :] * inter)
            acc += tl.dot(a, b)
        value = acc.to(tl.bfloat16)
        weight = tl.load(routing_weights + route, mask=mmask, other=0.0)
        value = (value * weight[:, None]).to(tl.bfloat16)
        if atomic_reduce:
            token = route // top_k
            tl.atomic_add(route_out + token[:, None] * hidden + ns[None, :],
                          value, mask=mmask[:, None])
        else:
            tl.store(route_out + route[:, None] * hidden + ns[None, :], value,
                     mask=mmask[:, None])
        m0 += BM


@triton.jit
def _expert_down_pair(
    mid, routing_weights, route_table, counts, down_w, route_out,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr, inter: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    n0 = tl.program_id(0) * BN
    expert = tl.program_id(1)
    ns = n0 + tl.arange(0, BN)
    count = tl.load(counts + expert)
    m0 = 0
    while m0 < count:
        rows = tl.arange(0, BM)
        ms0 = m0 + rows
        ms1 = m0 + BM + rows
        mask0 = ms0 < count
        mask1 = ms1 < count
        route0 = tl.load(route_table + expert * n_tokens + ms0,
                         mask=mask0, other=0).to(tl.int32)
        route1 = tl.load(route_table + expert * n_tokens + ms1,
                         mask=mask1, other=0).to(tl.int32)
        acc0 = tl.zeros((BM, BN), tl.float32)
        acc1 = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, inter, BK):
            ks = k0 + tl.arange(0, BK)
            w = tl.load(down_w + expert * (hidden * inter)
                        + ks[:, None] + ns[None, :] * inter)
            a0 = tl.load(mid + route0[:, None] * inter + ks[None, :],
                         mask=mask0[:, None], other=0.0)
            acc0 += tl.dot(a0, w)
            if m0 + BM < count:
                a1 = tl.load(mid + route1[:, None] * inter + ks[None, :],
                             mask=mask1[:, None], other=0.0)
                acc1 += tl.dot(a1, w)
        val0 = acc0.to(tl.bfloat16)
        rw0 = tl.load(routing_weights + route0, mask=mask0, other=0.0)
        val0 = (val0 * rw0[:, None]).to(tl.bfloat16)
        tl.store(route_out + route0[:, None] * hidden + ns[None, :], val0,
                 mask=mask0[:, None])
        if m0 + BM < count:
            val1 = acc1.to(tl.bfloat16)
            rw1 = tl.load(routing_weights + route1, mask=mask1, other=0.0)
            val1 = (val1 * rw1[:, None]).to(tl.bfloat16)
            tl.store(route_out + route1[:, None] * hidden + ns[None, :], val1,
                     mask=mask1[:, None])
        m0 += 2 * BM


@triton.jit
def _shared_gate_up(
    x, gate_w, up_w, shared_gate_w, mid, shared_weight,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr, inter: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m0 = tl.program_id(0) * BM
    n0 = tl.program_id(1) * BN
    ms = m0 + tl.arange(0, BM)
    ns = n0 + tl.arange(0, BN)
    mmask = ms < n_tokens
    acc_gate = tl.zeros((BM, BN), tl.float32)
    acc_up = tl.zeros((BM, BN), tl.float32)
    gate_logit = tl.zeros((BM,), tl.float32)
    for k0 in range(0, hidden, BK):
        ks = k0 + tl.arange(0, BK)
        xv = tl.load(x + ms[:, None] * hidden + ks[None, :],
                     mask=mmask[:, None], other=0.0)
        gw = tl.load(gate_w + ns[None, :] * hidden + ks[:, None])
        uw = tl.load(up_w + ns[None, :] * hidden + ks[:, None])
        acc_gate += tl.dot(xv, gw)
        acc_up += tl.dot(xv, uw)
        if n0 == 0:
            sw = tl.load(shared_gate_w + ks)
            gate_logit += tl.sum(xv.to(tl.float32) * sw[None, :].to(tl.float32), axis=1)
    gate = acc_gate.to(tl.bfloat16)
    up = acc_up.to(tl.bfloat16)
    denom = (1.0 + tl.exp(-gate.to(tl.float32))).to(tl.bfloat16)
    silu = (gate / denom).to(tl.bfloat16)
    value = (silu * up).to(tl.bfloat16)
    tl.store(mid + ms[:, None] * inter + ns[None, :], value,
             mask=mmask[:, None])
    if n0 == 0:
        logit = gate_logit.to(tl.bfloat16).to(tl.float32)
        sigmoid = (1.0 / (1.0 + tl.exp(-logit))).to(tl.bfloat16)
        tl.store(shared_weight + ms, sigmoid, mask=mmask)


@triton.jit
def _shared_down_and_reduce(
    shared_mid, shared_weight, shared_down_w, route_out, out,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr, inter: tl.constexpr, top_k: tl.constexpr,
    routed_is_reduced: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m0 = tl.program_id(0) * BM
    n0 = tl.program_id(1) * BN
    ms = m0 + tl.arange(0, BM)
    ns = n0 + tl.arange(0, BN)
    mmask = ms < n_tokens
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, inter, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(shared_mid + ms[:, None] * inter + ks[None, :],
                    mask=mmask[:, None], other=0.0)
        b = tl.load(shared_down_w + ns[None, :] * inter + ks[:, None])
        acc += tl.dot(a, b)
    shared = acc.to(tl.bfloat16)
    sw = tl.load(shared_weight + ms, mask=mmask, other=0.0)
    shared = (shared * sw[:, None]).to(tl.bfloat16)

    if routed_is_reduced:
        routed = tl.load(route_out + ms[:, None] * hidden + ns[None, :],
                         mask=mmask[:, None], other=0.0)
    else:
        routed = tl.zeros((BM, BN), tl.bfloat16)
        for slot in range(0, top_k):
            route = ms * top_k + slot
            value = tl.load(route_out + route[:, None] * hidden + ns[None, :],
                            mask=mmask[:, None], other=0.0)
            routed = (routed + value).to(tl.bfloat16)
    result = (routed + shared).to(tl.bfloat16)
    tl.store(out + ms[:, None] * hidden + ns[None, :], result,
             mask=mmask[:, None])


@torch.no_grad()
def run(
    hidden_states,
    routing_weights,
    selected_experts,
    expert_gate_proj_weights,
    expert_up_proj_weights,
    expert_down_proj_weights,
    shared_expert_gate_proj_weight,
    shared_expert_up_proj_weight,
    shared_expert_down_proj_weight,
    shared_expert_gate_weight,
):
    global _SHARED_STREAM
    n_tokens = hidden_states.shape[0]
    n_routes = n_tokens * TOP_K
    device = hidden_states.device

    counts = torch.empty((N_EXPERTS,), dtype=torch.int32, device=device)
    route_table = torch.empty((N_EXPERTS, n_tokens), dtype=torch.int32, device=device)
    expert_mid = torch.empty((n_routes, INTER), dtype=torch.bfloat16, device=device)
    route_out = torch.empty((n_routes, HIDDEN), dtype=torch.bfloat16, device=device)
    shared_mid = torch.empty((n_tokens, INTER), dtype=torch.bfloat16, device=device)
    shared_weight = torch.empty((n_tokens,), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(hidden_states)

    current_stream = torch.cuda.current_stream(device)
    if _SHARED_STREAM is None:
        _SHARED_STREAM = torch.cuda.Stream(device=device)
    _SHARED_STREAM.wait_stream(current_stream)
    with torch.cuda.stream(_SHARED_STREAM):
        _shared_gate_up[(triton.cdiv(n_tokens, 64), INTER // 64)](
            hidden_states, shared_expert_gate_proj_weight,
            shared_expert_up_proj_weight, shared_expert_gate_weight,
            shared_mid, shared_weight, n_tokens,
            HIDDEN, INTER,
            BM=64, BN=64, BK=64, num_warps=4, num_stages=2,
        )

    _clear_counts[(2,)](counts, N_EXPERTS, num_warps=1)
    _make_route_table[(triton.cdiv(n_routes, 256),)](
        selected_experts, counts, route_table, n_routes, n_tokens, BLOCK=256,
        num_warps=4,
    )
    if n_tokens > 4096:
        _expert_gate_up[(INTER // 64, N_EXPERTS)](
            hidden_states, route_table, counts, expert_gate_proj_weights,
            expert_up_proj_weights, expert_mid, n_tokens,
            HIDDEN, INTER, TOP_K,
            -1, n_tokens,
            BM=256, BN=64, BK=64, num_warps=4, num_stages=1,
        )
        _expert_down[(HIDDEN // 128, N_EXPERTS)](
            expert_mid, routing_weights, route_table, counts,
            expert_down_proj_weights, route_out, n_tokens,
            HIDDEN, INTER, TOP_K,
            -1, n_tokens, False,
            BM=256, BN=128, BK=32, num_warps=8, num_stages=2,
        )
    else:
        _expert_gate_up[(INTER // 64, N_EXPERTS)](
            hidden_states, route_table, counts, expert_gate_proj_weights,
            expert_up_proj_weights, expert_mid, n_tokens,
            HIDDEN, INTER, TOP_K,
            -1, n_tokens,
            BM=128, BN=64, BK=64, num_warps=4, num_stages=2,
        )
        _expert_down[(HIDDEN // 128, N_EXPERTS)](
            expert_mid, routing_weights, route_table, counts,
            expert_down_proj_weights, route_out, n_tokens,
            HIDDEN, INTER, TOP_K,
            -1, n_tokens, False,
            BM=128, BN=128, BK=64, num_warps=4, num_stages=1,
        )
    current_stream.wait_stream(_SHARED_STREAM)
    _shared_down_and_reduce[(triton.cdiv(n_tokens, 128), HIDDEN // 128)](
        shared_mid, shared_weight, shared_expert_down_proj_weight,
        route_out, output, n_tokens,
        HIDDEN, INTER, TOP_K, False,
        BM=128, BN=128, BK=32, num_warps=4, num_stages=2,
    )
    return output
