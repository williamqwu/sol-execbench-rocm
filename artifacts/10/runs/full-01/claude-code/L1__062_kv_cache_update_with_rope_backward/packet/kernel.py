import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: everything that touches the "new" rows (cache_position rows).
#
# For each (batch, block-of-new-positions) it handles ALL kv heads so that the
# grad_cos / grad_sin reduction over heads stays in registers.
#
# It also writes the zeros into the two cache-gradient outputs for those rows,
# so kernel 2 can skip them entirely (saves reading + rewriting them).
# ---------------------------------------------------------------------------
@triton.jit
def _rope_bwd_new(
    GKC, GVC, KEY, COS, SIN, CP,
    GKS, GVS, GCOS, GSIN, KO, VO,
    B, N, S,
    H: tl.constexpr,
    HD: tl.constexpr,      # half dim (64)
    BN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    n = pid_n * BN + tl.arange(0, BN)
    nmask = n < N

    d1 = tl.arange(0, HD)
    d2 = HD + tl.arange(0, HD)

    D = 2 * HD

    pos = tl.load(CP + n, mask=nmask, other=0).to(tl.int32)

    # cos / sin are shared across heads -> load once
    cbase = pid_b * N * D + n[:, None] * D
    c1 = tl.load(COS + cbase + d1[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
    c2 = tl.load(COS + cbase + d2[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cbase + d1[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cbase + d2[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)

    acc_cos1 = tl.zeros([BN, HD], dtype=tl.float32)
    acc_cos2 = tl.zeros([BN, HD], dtype=tl.float32)
    acc_sin1 = tl.zeros([BN, HD], dtype=tl.float32)
    acc_sin2 = tl.zeros([BN, HD], dtype=tl.float32)

    zero = tl.zeros([BN, HD], dtype=tl.bfloat16)

    for h in tl.static_range(H):
        # cache rows (gathered at cache_position)
        cache_row = (pid_b * H + h) * S * D + pos[:, None] * D
        gk1 = tl.load(GKC + cache_row + d1[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        gk2 = tl.load(GKC + cache_row + d2[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)

        # new-state rows
        new_row = (pid_b * H + h) * N * D + n[:, None] * D
        k1 = tl.load(KEY + new_row + d1[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(KEY + new_row + d2[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)

        # ---- grad_key_states (match reference intermediate bf16 rounding) ----
        t_cos1 = (gk1 * c1).to(tl.bfloat16).to(tl.float32)
        t_cos2 = (gk2 * c2).to(tl.bfloat16).to(tl.float32)
        t_sin1 = (gk1 * s1).to(tl.bfloat16).to(tl.float32)
        t_sin2 = (gk2 * s2).to(tl.bfloat16).to(tl.float32)

        o1 = (t_cos1 + t_sin2).to(tl.bfloat16)
        o2 = (t_cos2 - t_sin1).to(tl.bfloat16)
        tl.store(GKS + new_row + d1[None, :], o1, mask=nmask[:, None])
        tl.store(GKS + new_row + d2[None, :], o2, mask=nmask[:, None])

        # ---- grad_cos / grad_sin partials (bf16 products, fp32 accumulate) ---
        acc_cos1 += (gk1 * k1).to(tl.bfloat16).to(tl.float32)
        acc_cos2 += (gk2 * k2).to(tl.bfloat16).to(tl.float32)
        # k_rotated_half = cat(-k2, k1)
        acc_sin1 += (gk1 * (-k2)).to(tl.bfloat16).to(tl.float32)
        acc_sin2 += (gk2 * k1).to(tl.bfloat16).to(tl.float32)

        # ---- grad_value_states = grad_value_cache[:, :, cache_position] ------
        gv1 = tl.load(GVC + cache_row + d1[None, :], mask=nmask[:, None])
        gv2 = tl.load(GVC + cache_row + d2[None, :], mask=nmask[:, None])
        tl.store(GVS + new_row + d1[None, :], gv1, mask=nmask[:, None])
        tl.store(GVS + new_row + d2[None, :], gv2, mask=nmask[:, None])

        # ---- zero the touched rows of the cache-gradient outputs -------------
        tl.store(KO + cache_row + d1[None, :], zero, mask=nmask[:, None])
        tl.store(KO + cache_row + d2[None, :], zero, mask=nmask[:, None])
        tl.store(VO + cache_row + d1[None, :], zero, mask=nmask[:, None])
        tl.store(VO + cache_row + d2[None, :], zero, mask=nmask[:, None])

    tl.store(GCOS + cbase + d1[None, :], acc_cos1.to(tl.bfloat16), mask=nmask[:, None])
    tl.store(GCOS + cbase + d2[None, :], acc_cos2.to(tl.bfloat16), mask=nmask[:, None])
    tl.store(GSIN + cbase + d1[None, :], acc_sin1.to(tl.bfloat16), mask=nmask[:, None])
    tl.store(GSIN + cbase + d2[None, :], acc_sin2.to(tl.bfloat16), mask=nmask[:, None])


# ---------------------------------------------------------------------------
# Kernel 2: copy every cache row that is NOT in cache_position.
# ---------------------------------------------------------------------------
@triton.jit
def _copy_rest(
    GKC, GVC, KO, VO, MASK,
    TOTAL_ROWS, S,
    D: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    rmask = r < TOTAL_ROWS
    srow = r % S
    iscp = tl.load(MASK + srow, mask=rmask, other=1)
    keep = rmask & (iscp == 0)

    offs = r[:, None] * D + tl.arange(0, D)[None, :]
    m = keep[:, None]
    a = tl.load(GKC + offs, mask=m)
    b = tl.load(GVC + offs, mask=m)
    tl.store(KO + offs, a, mask=m)
    tl.store(VO + offs, b, mask=m)


def run(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    B, H, S, D = grad_key_cache.shape
    N = key_states.shape[2]
    HD = D // 2

    grad_key_cache = grad_key_cache.contiguous()
    grad_value_cache = grad_value_cache.contiguous()
    key_states = key_states.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    cache_position = cache_position.contiguous()

    dev = grad_key_cache.device
    gks = torch.empty_like(key_states)
    gvs = torch.empty_like(key_states)
    gcos = torch.empty_like(cos)
    gsin = torch.empty_like(cos)
    ko = torch.empty_like(grad_key_cache)
    vo = torch.empty_like(grad_value_cache)

    # boolean "row is in cache_position" table
    mask = torch.zeros(S, dtype=torch.int8, device=dev)
    mask[cache_position] = 1

    BN = 16 if N >= 16 else (8 if N >= 8 else 1)
    grid1 = (B, triton.cdiv(N, BN))
    _rope_bwd_new[grid1](
        grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position,
        gks, gvs, gcos, gsin, ko, vo,
        B, N, S,
        H=H, HD=HD, BN=BN,
        num_warps=4, num_stages=1,
    )

    total_rows = B * H * S
    BR = 16
    grid2 = (triton.cdiv(total_rows, BR),)
    _copy_rest[grid2](
        grad_key_cache, grad_value_cache, ko, vo, mask,
        total_rows, S,
        D=D, BR=BR,
        num_warps=4, num_stages=1,
    )

    return gks, gvs, gcos, gsin, ko, vo
