import torch, triton
import triton.language as tl


@triton.jit
def _build(idx_ptr, cnt_ptr, slots_ptr, ovf_ptr, novf_ptr, M, C: tl.constexpr,
           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < M
    t = tl.load(idx_ptr + offs, mask=m, other=0).to(tl.int32)
    p = tl.atomic_add(cnt_ptr + t, 1, mask=m)
    ok = m & (p < C)
    tl.store(slots_ptr + t * C + p, offs.to(tl.int32), mask=ok)
    bad = m & (p >= C)
    if tl.sum(bad.to(tl.int32)) > 0:
        q = tl.atomic_add(novf_ptr + tl.zeros([BLOCK], tl.int32), 1, mask=bad)
        tl.store(ovf_ptr + 2 * q, t, mask=bad)
        tl.store(ovf_ptr + 2 * q + 1, offs.to(tl.int32), mask=bad)


@triton.jit
def _gather(final_ptr, exp_ptr, cnt_ptr, slots_ptr, ovf_ptr, novf_ptr, out_ptr,
            H: tl.constexpr, C: tl.constexpr, NH: tl.constexpr,
            BLOCK_H: tl.constexpr, KB: tl.constexpr):
    t = tl.program_id(0)
    ph = tl.program_id(1)
    h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    base = t.to(tl.int64) * H + h
    acc = tl.load(final_ptr + base, mask=hm, other=0.0).to(tl.float32)
    cnt = tl.load(cnt_ptr + t)
    n = tl.minimum(cnt, C)
    ks0 = tl.arange(0, KB)
    for k0 in range(0, n, KB):
        ks = k0 + ks0
        km = ks < n
        rows = tl.load(slots_ptr + t * C + ks, mask=km, other=0)
        ptrs = exp_ptr + rows[:, None].to(tl.int64) * H + h[None, :]
        v = tl.load(ptrs, mask=km[:, None] & hm[None, :], other=0.0)
        acc += tl.sum(v.to(tl.float32), axis=0)
    nov = tl.load(novf_ptr)
    if nov > 0:
        for j0 in range(0, nov, KB):
            js = j0 + ks0
            jm = js < nov
            ot = tl.load(ovf_ptr + 2 * js, mask=jm, other=-1)
            orow = tl.load(ovf_ptr + 2 * js + 1, mask=jm, other=0)
            sel = jm & (ot == t)
            ptrs = exp_ptr + orow[:, None].to(tl.int64) * H + h[None, :]
            v = tl.load(ptrs, mask=sel[:, None] & hm[None, :], other=0.0)
            acc += tl.sum(v.to(tl.float32), axis=0)
    tl.store(out_ptr + base, acc.to(tl.bfloat16), mask=hm)


CAP = 32


def run_impl(final_hidden_states, expert_outputs, token_indices,
             BLOCK_H=1024, KB=8, C=CAP, BB=1024, nw=4):
    N, H = final_hidden_states.shape
    M = token_indices.shape[0]
    dev = final_hidden_states.device
    buf = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    cnt = buf[:N]
    novf = buf[N:]
    slots = torch.empty(N * C, dtype=torch.int32, device=dev)
    ovf = torch.empty(2 * M, dtype=torch.int32, device=dev)
    out = torch.empty_like(final_hidden_states)
    _build[(triton.cdiv(M, BB),)](token_indices, cnt, slots, ovf, novf, M,
                                  C=C, BLOCK=BB, num_warps=4)
    NH = triton.cdiv(H, BLOCK_H)
    _gather[(N, NH)](final_hidden_states, expert_outputs, cnt, slots, ovf, novf,
                     out, H=H, C=C, NH=NH, BLOCK_H=BLOCK_H, KB=KB, num_warps=nw)
    return out
