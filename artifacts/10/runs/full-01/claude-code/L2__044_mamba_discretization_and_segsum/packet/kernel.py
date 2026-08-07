import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: A_cumsum + decay_states  (tiny outputs)
#   A_discrete[b,h,c,i] = -exp(A_log[h]) * softplus(dt[b,c,i,h])
#   A_cumsum            = cumsum over i
#   decay_states        = exp(A_cumsum[..., -1] - A_cumsum)
# ---------------------------------------------------------------------------
@triton.jit
def _k_ac(dt_ptr, alog_ptr, ac_ptr, ds_ptr,
          NC,
          sdt_bc, sdt_i, sdt_h,
          sac_b, sac_h, sac_c,
          CS: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(0)          # b * NC + c
    b = pid // NC
    c = pid % NC
    i = tl.arange(0, CS)
    h = tl.arange(0, H)
    x = tl.load(dt_ptr + pid * sdt_bc + i[:, None] * sdt_i
                + h[None, :] * sdt_h).to(tl.float32)
    # torch softplus (beta=1, threshold=20)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    a = -tl.exp(tl.load(alog_ptr + h).to(tl.float32))          # [H]
    ad = a[None, :] * sp                                        # [CS, H]
    ac = tl.cumsum(ad, axis=0)                                  # [CS, H]
    last = tl.sum(tl.where(i[:, None] == (CS - 1), ac, 0.0), axis=0)  # [H]
    ds = tl.exp(last[None, :] - ac)
    off = b * sac_b + c * sac_c + h[None, :] * sac_h + i[:, None]
    tl.store(ac_ptr + off, ac)
    tl.store(ds_ptr + off, ds)


# ---------------------------------------------------------------------------
# Kernel B: L[b,h,c,i,j] = exp(A_cumsum[i] - A_cumsum[j]) for j <= i else 0
# ---------------------------------------------------------------------------
@triton.jit
def _k_L(ac_ptr, L_ptr,
         NC,
         sac_b, sac_h, sac_c,
         sL_b, sL_h, sL_c, sL_i,
         CS: tl.constexpr, H: tl.constexpr, BI: tl.constexpr):
    pid = tl.program_id(0)
    pi = tl.program_id(1)
    c = pid % NC
    t = pid // NC
    h = t % H
    b = t // H
    acb = ac_ptr + b * sac_b + h * sac_h + c * sac_c
    i = pi * BI + tl.arange(0, BI)
    j = tl.arange(0, CS)
    ai = tl.load(acb + i)
    aj = tl.load(acb + j)
    v = tl.exp(ai[:, None] - aj[None, :])
    v = tl.where(j[None, :] <= i[:, None], v, 0.0)
    o = L_ptr + b * sL_b + h * sL_h + c * sL_c + i[:, None] * sL_i + j[None, :]
    tl.store(o, v.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Kernel C: G[b,c,i,j,h] = sum_n C[b,c,i,n] * B[b,c,j,n]   (same for all h)
#           M[b,c,i,j,h] = G * L
# ---------------------------------------------------------------------------
@triton.jit
def _k_GM(Bp, Cp, ac_ptr, Gp, Mp,
          NC,
          sB_bc, sB_j, sC_bc, sC_i,
          sac_b, sac_h, sac_c,
          sG_bc, sG_i,
          CS: tl.constexpr, DS: tl.constexpr, H: tl.constexpr,
          BI: tl.constexpr, BJ: tl.constexpr):
    bc = tl.program_id(0)
    pi = tl.program_id(1)
    pj = tl.program_id(2)
    b = bc // NC
    c = bc % NC

    i = pi * BI + tl.arange(0, BI)
    j = pj * BJ + tl.arange(0, BJ)
    n = tl.arange(0, DS)

    ct = tl.load(Cp + bc * sC_bc + i[:, None] * sC_i + n[None, :])
    bt = tl.load(Bp + bc * sB_bc + j[:, None] * sB_j + n[None, :])
    acc = tl.dot(ct, tl.trans(bt), out_dtype=tl.float32)        # [BI, BJ]

    k = tl.arange(0, BJ * H)
    obase = bc * sG_bc + i[:, None] * sG_i + (pj * BJ * H) + k[None, :]

    g3 = tl.broadcast_to(acc.to(tl.bfloat16)[:, :, None], (BI, BJ, H))
    tl.store(Gp + obase, tl.reshape(g3, (BI, BJ * H)))

    hh = tl.arange(0, H)
    acb = ac_ptr + b * sac_b + c * sac_c
    aci = tl.load(acb + i[:, None] + hh[None, :] * sac_h)       # [BI, H]
    acj = tl.load(acb + j[:, None] + hh[None, :] * sac_h)       # [BJ, H]
    l3 = tl.where(j[None, :, None] <= i[:, None, None],
                  tl.exp(aci[:, None, :] - acj[None, :, :]), 0.0)
    m3 = acc[:, :, None] * l3
    tl.store(Mp + obase, tl.reshape(m3.to(tl.bfloat16), (BI, BJ * H)))


def _segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    device = input_tensor.device
    input_expanded = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=device,
                                 dtype=torch.bool), diagonal=-1)
    input_masked = input_expanded.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_masked, dim=-2)
    mask_final = torch.tril(torch.ones(chunk_size, chunk_size, device=device,
                                       dtype=torch.bool), diagonal=0)
    return tensor_segsum.masked_fill(~mask_final, float('-inf'))


def _torch_fallback(hidden_states, dt, A_log, B, C):
    num_heads = hidden_states.shape[3]
    n_groups = B.shape[3]
    A = -torch.exp(A_log.float())
    dt_soft = F.softplus(dt.float())
    A_discrete = A[None, :, None, None] * dt_soft.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_discrete, dim=-1)
    L = torch.exp(_segment_sum(A_discrete))
    r = num_heads // n_groups
    Be = B.repeat_interleave(r, dim=3)
    Ce = C.repeat_interleave(r, dim=3)
    G = torch.einsum('bcihd,bcjhd->bcijh', Ce.float(), Be.float())
    M = G * L.permute(0, 2, 3, 4, 1)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    return (L.to(torch.bfloat16), G.to(torch.bfloat16), M.to(torch.bfloat16),
            A_cumsum, decay_states)


def _pick(nbc):
    # (BI_L, BI_GM, BJ_GM)
    if nbc >= 16:
        return 64, 32, 32
    if nbc >= 4:
        return 32, 32, 32
    return 16, 16, 32


@torch.no_grad()
def run(hidden_states, dt, A_log, B, C):
    Bsz, NC, CS, H, _ = hidden_states.shape
    NG = B.shape[3]
    DS = B.shape[4]

    ok = (NG == 1 and hidden_states.is_cuda
          and (CS & (CS - 1)) == 0 and CS >= 64
          and (DS & (DS - 1)) == 0 and DS >= 16
          and (H & (H - 1)) == 0
          and B.stride(4) == 1 and C.stride(4) == 1
          and dt.stride(3) == 1 and A_log.stride(0) == 1
          and dt.stride(2) == H and dt.stride(1) == CS * H)
    if not ok:
        return _torch_fallback(hidden_states, dt, A_log, B, C)

    dev = hidden_states.device
    A_cumsum = torch.empty((Bsz, H, NC, CS), device=dev, dtype=torch.float32)
    decay_states = torch.empty((Bsz, H, NC, CS), device=dev, dtype=torch.float32)
    Lo = torch.empty((Bsz, H, NC, CS, CS), device=dev, dtype=torch.bfloat16)
    Go = torch.empty((Bsz, NC, CS, CS, H), device=dev, dtype=torch.bfloat16)
    Mo = torch.empty((Bsz, NC, CS, CS, H), device=dev, dtype=torch.bfloat16)

    nbc = Bsz * NC
    BI_L, BI, BJ = _pick(nbc)

    _k_ac[(nbc,)](
        dt, A_log, A_cumsum, decay_states,
        NC,
        dt.stride(1), dt.stride(2), dt.stride(3),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2),
        CS=CS, H=H, num_warps=4,
    )

    _k_L[(Bsz * H * NC, CS // BI_L)](
        A_cumsum, Lo,
        NC,
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2),
        Lo.stride(0), Lo.stride(1), Lo.stride(2), Lo.stride(3),
        CS=CS, H=H, BI=BI_L, num_warps=4,
    )

    _k_GM[(nbc, CS // BI, CS // BJ)](
        B, C, A_cumsum, Go, Mo,
        NC,
        B.stride(1), B.stride(2), C.stride(1), C.stride(2),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2),
        Go.stride(1), Go.stride(2),
        CS=CS, DS=DS, H=H, BI=BI, BJ=BJ, num_warps=4,
    )

    return Lo, Go, Mo, A_cumsum, decay_states
