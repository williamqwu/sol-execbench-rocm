import torch
import triton
import triton.language as tl


@triton.jit
def _fused_projection(a, w, out, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    group = 8
    group_width = group * grid_m
    group_id = pid // group_width
    first_n = group_id * group
    group_n = tl.minimum(group, tl.cdiv(N, BN) - first_n)
    pid_m = (pid % group_width) // group_n
    pid_n = first_n + (pid % group_width) % group_n

    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    b = rows // S
    s = rows - b * S
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kvals = k0 + kk
        h = kvals // D
        d = kvals - h * D
        aoffs = ((b[:, None] * (K // D) + h[None, :]) * S +
                 s[:, None]) * D + d[None, :]
        woffs = cols[None, :] * K + kvals[:, None]
        av = tl.load(a + aoffs, mask=rows[:, None] < M, other=0.0)
        wv = tl.load(w + woffs, mask=cols[None, :] < N, other=0.0)
        acc += tl.dot(av, wv)
    ooffs = rows[:, None] * N + cols[None, :]
    tl.store(out + ooffs, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    m = bsz * seq_len
    n, k = o_proj_weight.shape
    out = torch.empty((m, n), device=attn_output.device, dtype=attn_output.dtype)
    bm, bn, bk = 32, 64, 64
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _fused_projection[grid](attn_output, o_proj_weight, out, M=m, N=n, K=k,
                            S=seq_len, D=v_head_dim, BM=bm, BN=bn, BK=bk,
                            num_warps=8, num_stages=2)
    return out.view(bsz, seq_len, n)
