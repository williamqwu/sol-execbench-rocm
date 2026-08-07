import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["S"])
def _rope_bwd_kernel(
    GQE, GKE, Q, K, E,
    GQ, GK, GE,
    S,
    HALF: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Fused backward for vision-3D RoPE -- one pass over q/k/grads.

    Layouts (contiguous): q/k/grad_*_embed are (S, NH, 2*HALF); embeddings and
    grad_embeddings are (S, 2*HALF).

    Numerics match torch eager bit-for-bit:
      * the head reduction uses PyTorch's 4-accumulator unrolled order,
        acc[j] = ((x[j] + x[j+4]) + x[j+8]) + x[j+12], combined left-to-right;
      * the kernel is compiled with enable_fp_fusion=False, since contracting
        the mul/add pairs into FMAs would make us *more* accurate than the
        reference and push the exact-match comparison out of tolerance.
    """
    pid = tl.program_id(0)
    s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    m = s < S
    mm = m[:, None]

    d = tl.arange(0, HALF)
    D: tl.constexpr = 2 * HALF

    eoff = s[:, None] * D + d[None, :]
    e1 = tl.load(E + eoff, mask=mm, other=0.0)
    e2 = tl.load(E + eoff + HALF, mask=mm, other=0.0)

    cos1 = tl.cos(e1)
    sin1 = tl.sin(e1)
    cos2 = tl.cos(e2)
    sin2 = tl.sin(e2)

    base = s[:, None] * (NH * D) + d[None, :]

    z = tl.zeros((BLOCK_S, HALF), dtype=tl.float32)
    c1_0 = z; c1_1 = z; c1_2 = z; c1_3 = z
    c2_0 = z; c2_1 = z; c2_2 = z; c2_3 = z
    s1_0 = z; s1_1 = z; s1_2 = z; s1_3 = z
    s2_0 = z; s2_1 = z; s2_2 = z; s2_3 = z

    for hb in tl.static_range(0, NH, 4):
        for j in tl.static_range(0, 4):
            off = base + (hb + j) * D

            gq1 = tl.load(GQE + off, mask=mm, other=0.0)
            gq2 = tl.load(GQE + off + HALF, mask=mm, other=0.0)
            q1 = tl.load(Q + off, mask=mm, other=0.0)
            q2 = tl.load(Q + off + HALF, mask=mm, other=0.0)
            gk1 = tl.load(GKE + off, mask=mm, other=0.0)
            gk2 = tl.load(GKE + off + HALF, mask=mm, other=0.0)
            k1 = tl.load(K + off, mask=mm, other=0.0)
            k2 = tl.load(K + off + HALF, mask=mm, other=0.0)

            # grad wrt q/k:  g*cos + rotate_half_backward(g*sin),
            # where rotate_half_backward([a, b]) = [b, -a].
            tl.store(GQ + off, gq1 * cos1 + gq2 * sin2, mask=mm)
            tl.store(GQ + off + HALF, gq2 * cos2 - gq1 * sin1, mask=mm)
            tl.store(GK + off, gk1 * cos1 + gk2 * sin2, mask=mm)
            tl.store(GK + off + HALF, gk2 * cos2 - gk1 * sin1, mask=mm)

            # grad wrt cos/sin, with rotate_half(x) = [-x2, x1].
            vc1 = gq1 * q1 + gk1 * k1
            vc2 = gq2 * q2 + gk2 * k2
            vs1 = gq1 * -q2 + gk1 * -k2
            vs2 = gq2 * q1 + gk2 * k1

            if j == 0:
                c1_0 += vc1; c2_0 += vc2; s1_0 += vs1; s2_0 += vs2
            elif j == 1:
                c1_1 += vc1; c2_1 += vc2; s1_1 += vs1; s2_1 += vs2
            elif j == 2:
                c1_2 += vc1; c2_2 += vc2; s1_2 += vs1; s2_2 += vs2
            else:
                c1_3 += vc1; c2_3 += vc2; s1_3 += vs1; s2_3 += vs2

    gc1 = ((c1_0 + c1_1) + c1_2) + c1_3
    gc2 = ((c2_0 + c2_1) + c2_2) + c2_3
    gs1 = ((s1_0 + s1_1) + s1_2) + s1_3
    gs2 = ((s2_0 + s2_1) + s2_2) + s2_3

    tl.store(GE + eoff, gc1 * -sin1 + gs1 * cos1, mask=mm)
    tl.store(GE + eoff + HALF, gc2 * -sin2 + gs2 * cos2, mask=mm)


# Compiled variants, keyed by (BLOCK_S, num_warps).  Because S is marked
# do_not_specialize, one compile per tile shape serves every sequence length.
_VARIANTS = ((1, 1), (2, 2), (4, 4))

# key -> (launch_fn, function_handle, packed_metadata, BLOCK_S)
_cache = {}

# Cheap current-stream lookup (~0.04 us vs ~2.5 us for
# torch.cuda.current_stream().cuda_stream).  Queried every call rather than
# cached, so the kernel still lands on whatever stream the caller is using.
try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover
    def _raw_stream(_idx):
        return torch.cuda.current_stream().cuda_stream


def _build(key, HALF, NH):
    bs, nw = key
    dev = torch.cuda.current_device()
    dq = torch.empty(1, NH, 2 * HALF, device=dev, dtype=torch.float32)
    de = torch.empty(1, 2 * HALF, device=dev, dtype=torch.float32)
    ck = _rope_bwd_kernel.warmup(
        dq, dq, dq, dq, de, dq, dq, de, 1,
        grid=(1,), HALF=HALF, NH=NH, BLOCK_S=bs,
        num_warps=nw, num_stages=1, enable_fp_fusion=False,
    )
    ck._init_handles()
    entry = (ck.run.launch, ck.function, ck.packed_metadata, bs)
    _cache[key] = entry
    return entry


def _config(S):
    # Below ~1k tokens the kernel is launch-latency-bound and the tile shape
    # barely matters; past that, wider tiles amortise per-workgroup setup.
    if S <= 1024:
        return (1, 1)
    if S <= 2048:
        return (2, 2)
    return (4, 4)


@torch.no_grad()
def run(
    grad_q_embed: torch.Tensor,
    grad_k_embed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    embeddings: torch.Tensor,
):
    S, NH, D = q.shape
    HALF = D // 2

    if not (grad_q_embed.is_contiguous() and grad_k_embed.is_contiguous()
            and q.is_contiguous() and k.is_contiguous()
            and embeddings.is_contiguous()):
        grad_q_embed = grad_q_embed.contiguous()
        grad_k_embed = grad_k_embed.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        embeddings = embeddings.contiguous()

    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_embeddings = torch.empty_like(embeddings)

    key = _config(S)
    entry = _cache.get(key)
    if entry is None:
        entry = _build(key, HALF, NH)
    launch, fn, pmeta, bs = entry

    # Call the compiled launcher directly.  Triton's Python dispatch path costs
    # ~16 us per call, several times the actual kernel at the smaller sequence
    # lengths in this workload set.
    launch(
        False, (S + bs - 1) // bs, 1, 1,
        _raw_stream(q.device.index), fn, None, pmeta, None, None, None,
        grad_q_embed.data_ptr(), grad_k_embed.data_ptr(),
        q.data_ptr(), k.data_ptr(), embeddings.data_ptr(),
        grad_q.data_ptr(), grad_k.data_ptr(), grad_embeddings.data_ptr(),
        S, HALF, NH, bs,
    )
    return grad_q, grad_k, grad_embeddings
