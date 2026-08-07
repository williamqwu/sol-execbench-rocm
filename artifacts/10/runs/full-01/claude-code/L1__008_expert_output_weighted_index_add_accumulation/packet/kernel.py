import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Semantics (from reference.py):
#     out = final_hidden_states.clone()
#     out.index_add_(0, token_indices, expert_outputs)
#
# i.e. out[token_indices[i]] += expert_outputs[i], bf16 in / bf16 out.
#
# Two strategies, chosen by problem size:
#
#  * small M  -> direct bf16 atomic scatter-add. Only 2 kernel launches, which
#                is what matters when the problem is launch-bound.
#  * large M  -> build a CSR (row -> list of contributing source rows) and do a
#                deterministic gather, accumulating in fp32. Costs 4 extra
#                launches but turns a scattered atomic RMW stream into pure
#                streaming reads, which is far closer to the bandwidth bound.
#
# The switch is on shape only (M), never on tensor values.
# ---------------------------------------------------------------------------


@triton.jit
def _atomic_scatter(src_ptr, idx_ptr, out_ptr, N,
                    H: tl.constexpr, BH: tl.constexpr, RPB: tl.constexpr):
    pid = tl.program_id(0)
    hb = tl.program_id(1)
    hoffs = hb * BH + tl.arange(0, BH)
    for r in range(RPB):
        j = pid * RPB + r
        if j < N:
            m = tl.load(idx_ptr + j)
            v = tl.load(src_ptr + j.to(tl.int64) * H + hoffs)
            tl.atomic_add(out_ptr + m * H + hoffs, v, sem="relaxed")


@triton.jit
def _count(idx_ptr, cnt_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    i = tl.load(idx_ptr + offs, mask=mask, other=0)
    tl.atomic_add(cnt_ptr + 1 + i, 1, mask=mask, sem="relaxed")


@triton.jit
def _scan(cnt_ptr, off_ptr, M1, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(cnt_ptr + offs, mask=offs < M1, other=0)
    tl.store(off_ptr + offs, tl.cumsum(v, 0), mask=offs < M1)


@triton.jit
def _scatter_rev(idx_ptr, cnt_ptr, off_ptr, perm_ptr, N, BLOCK: tl.constexpr):
    # Consumes cnt[1+i] as a countdown cursor, so no separate cursor buffer or
    # extra clone launch is needed. Leaves cnt back at zero.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    i = tl.load(idx_ptr + offs, mask=mask, other=0)
    k = tl.atomic_add(cnt_ptr + 1 + i, -1, mask=mask, sem="relaxed")
    end = tl.load(off_ptr + i + 1, mask=mask, other=0)
    tl.store(perm_ptr + end - k, offs.to(tl.int32), mask=mask)


@triton.jit
def _gather(fin_ptr, src_ptr, perm_ptr, off_ptr, out_ptr,
            M, H: tl.constexpr, BH: tl.constexpr, RM: tl.constexpr):
    m0 = tl.program_id(0) * RM
    hb = tl.program_id(1)
    hoffs = hb * BH + tl.arange(0, BH)
    for rr in range(RM):
        m = m0 + rr
        if m < M:
            base = m.to(tl.int64) * H + hoffs
            acc = tl.load(fin_ptr + base).to(tl.float32)
            s = tl.load(off_ptr + m)
            e = tl.load(off_ptr + m + 1)
            for k in range(s, e):
                j = tl.load(perm_ptr + k).to(tl.int64)
                acc += tl.load(src_ptr + j * H + hoffs).to(tl.float32)
            tl.store(out_ptr + base, acc.to(tl.bfloat16))


# Tuned on MI355X (gfx950).
_ATOMIC_MAX_M = 1200
_CB = 1024


def _run_atomic(f, s, i, BH=512, RPB=4, nw=4):
    out = f.clone()
    N, H = s.shape
    _atomic_scatter[(triton.cdiv(N, RPB), H // BH)](
        s, i, out, N, H, BH, RPB, num_warps=nw)
    return out


def _run_csr(f, s, i, BH=1024, nw=2, RM=1):
    M, H = f.shape
    N = s.shape[0]
    dev = f.device
    cnt = torch.zeros(M + 1, dtype=torch.int32, device=dev)
    g = (triton.cdiv(N, _CB),)
    _count[g](i, cnt, N, _CB)
    M1 = M + 1
    off = torch.empty(M1, dtype=torch.int32, device=dev)
    _scan[(1,)](cnt, off, M1, triton.next_power_of_2(M1), num_warps=8)
    perm = torch.empty(N, dtype=torch.int32, device=dev)
    _scatter_rev[g](i, cnt, off, perm, N, _CB)
    out = torch.empty_like(f)
    _gather[(triton.cdiv(M, RM), H // BH)](
        f, s, perm, off, out, M, H, BH, RM, num_warps=nw)
    return out


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    M, H = final_hidden_states.shape
    if M == 0 or expert_outputs.shape[0] == 0:
        return final_hidden_states.clone()
    # The scan is a single block, so it needs M+1 to fit one tile.
    if M <= _ATOMIC_MAX_M or (M + 1) > 65536 or (H % 512) != 0:
        return _run_atomic(final_hidden_states, expert_outputs, token_indices)
    return _run_csr(final_hidden_states, expert_outputs, token_indices)
