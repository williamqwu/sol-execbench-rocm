import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
# The reference does:
#   ge      = -grad_cos*sin(emb) + grad_sin*cos(emb)        [P, HD]
#   gr      = ge[:, :HD/2] + ge[:, HD/2:]                   [P, HD/2]
#   gr      -> reshape [P, 2, HQ]
#   gf      = zeros[M, HQ]; gf.index_add_(0, pos[:,k], gr[:,k,:]) for k=0,1
#   out     = arange(M) @ gf                                [HQ]
#
# index_add_ puts row p's group-k vector into row pos[p,k]; the final matmul
# weights row i by i and sums.  Composing the two, the scatter buffer cancels
# out entirely:
#
#   out[j] = sum_p ( pos[p,0]*(ge[p, j] + ge[p, j+36])
#                  + pos[p,1]*(ge[p,18+j] + ge[p,54+j]) )
#
# Writing c = g*HQ + j for g in 0..3, the weight for column c is simply
# pos[p, g % 2].  So the whole chain is one weighted reduction over a single
# streaming pass of grad_cos / grad_sin / emb -- no M x HQ scratch buffer, no
# atomics, and no pos_ids.max().item() host sync.
# ---------------------------------------------------------------------------


@triton.jit
def _rope_bwd_partial(
    gc_ptr, gs_ptr, emb_ptr, pos_ptr, part_ptr,
    n_rows, stride_row, stride_pos,
    BLOCK_R: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, G)
    offs_j = tl.arange(0, BLOCK_J)
    jmask = offs_j < HQ

    # column offsets within a row: c = g*HQ + j   ->  [G, BLOCK_J]
    col = offs_g[:, None] * HQ + offs_j[None, :]
    # weight selector: group g draws its scale from pos_ids[:, g % 2]
    use_w1 = (offs_g % 2) == 1

    acc = tl.zeros((G, BLOCK_J), dtype=tl.float32)

    nblocks = tl.cdiv(n_rows, BLOCK_R)
    for b in range(pid, nblocks, nprog):
        r = b * BLOCK_R + offs_r
        rmask = r < n_rows

        off = r[:, None, None] * stride_row + col[None, :, :]
        m = rmask[:, None, None] & jmask[None, None, :]

        e = tl.load(emb_ptr + off, mask=m, other=0.0)
        gc = tl.load(gc_ptr + off, mask=m, other=0.0)
        gs = tl.load(gs_ptr + off, mask=m, other=0.0)

        # d/demb of (cos, sin) chain, exactly as the reference orders it
        ge = -gc * tl.sin(e) + gs * tl.cos(e)

        p0 = tl.load(pos_ptr + r * stride_pos, mask=rmask, other=0).to(tl.float32)
        p1 = tl.load(pos_ptr + r * stride_pos + 1, mask=rmask, other=0).to(tl.float32)
        w = tl.where(use_w1[None, :, None], p1[:, None, None], p0[:, None, None])

        acc += tl.sum(ge * w, axis=0)

    res = tl.sum(acc, axis=0)  # [BLOCK_J]
    tl.store(part_ptr + pid * HQ + offs_j, res, mask=jmask)


@triton.jit
def _rope_bwd_finalize(
    part_ptr, out_ptr,
    nprog,
    HQ: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    offs_p = tl.arange(0, BLOCK_P)
    offs_j = tl.arange(0, BLOCK_J)
    jmask = offs_j < HQ
    m = (offs_p[:, None] < nprog) & jmask[None, :]
    v = tl.load(part_ptr + offs_p[:, None] * HQ + offs_j[None, :], mask=m, other=0.0)
    tl.store(out_ptr + offs_j, tl.sum(v, axis=0), mask=jmask)


_BLOCK_R = 16
_MAX_PROG = 256


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor,
) -> torch.Tensor:
    n_rows, head_dim = grad_cos.shape
    hq = inv_freq.shape[0]
    g = head_dim // hq

    dev = inv_freq.device
    out = torch.empty(hq, device=dev, dtype=torch.float32)
    if n_rows == 0:
        return out.zero_()

    nblocks = triton.cdiv(n_rows, _BLOCK_R)
    nprog = min(nblocks, _MAX_PROG)

    block_j = _next_pow2(hq)

    # nprog == 1 lets stage 1 write straight into the output: one launch.
    part = out if nprog == 1 else torch.empty(nprog * hq, device=dev, dtype=torch.float32)

    _rope_bwd_partial[(nprog,)](
        grad_cos, grad_sin, emb, pos_ids, part,
        n_rows, grad_cos.stride(0), pos_ids.stride(0),
        BLOCK_R=_BLOCK_R, HQ=hq, G=g, BLOCK_J=block_j,
        num_warps=4, num_stages=1,
    )

    if nprog > 1:
        _rope_bwd_finalize[(1,)](
            part, out, nprog,
            HQ=hq, BLOCK_P=_next_pow2(nprog), BLOCK_J=block_j,
            num_warps=4, num_stages=1,
        )

    return out
