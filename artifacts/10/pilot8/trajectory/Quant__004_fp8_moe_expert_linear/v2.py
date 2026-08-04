import torch, triton, triton.language as tl
FP8_MAX = tl.constexpr(448.0); FP8 = tl.constexpr(tl.float8e4nv)

@triton.jit
def gemm1(A, SA, W, SW, GQ, GS, M, KB: tl.constexpr, NH: tl.constexpr,
          sam, ssam, swn, sswn, sgm, sgsm,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
          EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BLOCK_M); npn = NH // BLOCK_N
    width = GROUP_M * npn
    gid = pid // width
    gsz = min(npm - gid*GROUP_M, GROUP_M)
    pid_m = gid*GROUP_M + ((pid % width) % gsz)
    pid_n = (pid % width) // gsz

    rm = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    snb = pid_n*(BLOCK_N//128) + tl.arange(0, BLOCK_N)//128   # scale-block index per col

    am = rm if EVEN_M else tl.where(rm < M, rm, 0)
    a_ptrs = A + am[:, None]*sam + rk[None, :]
    wg_ptrs = W + rn[:, None]*swn + rk[None, :]
    wu_ptrs = W + (NH + rn)[:, None]*swn + rk[None, :]
    sg_ptrs = SW + snb*sswn
    su_ptrs = SW + (NH//128 + snb)*sswn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, KB):
        a = tl.load(a_ptrs)
        wg = tl.load(wg_ptrs); wu = tl.load(wu_ptrs)
        if EVEN_M:
            sa = tl.load(SA + rm*ssam + kb)
        else:
            sa = tl.load(SA + rm*ssam + kb, mask=rm < M, other=0.0)
        sg = tl.load(sg_ptrs + kb); su = tl.load(su_ptrs + kb)
        acc_g += tl.dot(a, tl.trans(wg)) * (sa[:, None] * sg[None, :])
        acc_u += tl.dot(a, tl.trans(wu)) * (sa[:, None] * su[None, :])
        a_ptrs += 128; wg_ptrs += 128; wu_ptrs += 128

    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g*tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    r = (s*u).to(tl.bfloat16).to(tl.float32)

    # per-128-col-block amax within this tile
    rr = tl.reshape(r, (BLOCK_M, BLOCK_N//128, 128))
    amax = tl.max(tl.abs(rr), axis=2)                 # [BM, BN/128]
    scale = tl.maximum(amax*(1.0/FP8_MAX), 1e-12)
    q = rr / scale[:, :, None]
    q = tl.reshape(tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX), (BLOCK_M, BLOCK_N))

    gp = GQ + rm[:, None]*sgm + rn[None, :]
    sp = GS + rm[:, None]*sgsm + (pid_n*(BLOCK_N//128) + tl.arange(0, BLOCK_N//128))[None, :]
    if EVEN_M:
        tl.store(gp, q.to(FP8)); tl.store(sp, scale)
    else:
        tl.store(gp, q.to(FP8), mask=(rm < M)[:, None])
        tl.store(sp, scale, mask=(rm < M)[:, None])

@triton.jit
def gemm2(A, SA, W, SW, RW, C, M, KB: tl.constexpr, H: tl.constexpr,
          sam, ssam, swn, sswn, scm,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
          EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BLOCK_M); npn = H // BLOCK_N
    width = GROUP_M*npn
    gid = pid // width
    gsz = min(npm - gid*GROUP_M, GROUP_M)
    pid_m = gid*GROUP_M + ((pid % width) % gsz)
    pid_n = (pid % width) // gsz

    rm = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    snb = pid_n*(BLOCK_N//128) + tl.arange(0, BLOCK_N)//128

    am = rm if EVEN_M else tl.where(rm < M, rm, 0)
    a_ptrs = A + am[:, None]*sam + rk[None, :]
    w_ptrs = W + rn[:, None]*swn + rk[None, :]
    sw_ptrs = SW + snb*sswn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, KB):
        a = tl.load(a_ptrs); w = tl.load(w_ptrs)
        if EVEN_M:
            sa = tl.load(SA + rm*ssam + kb)
        else:
            sa = tl.load(SA + rm*ssam + kb, mask=rm < M, other=0.0)
        sw = tl.load(sw_ptrs + kb)
        acc += tl.dot(a, tl.trans(w)) * (sa[:, None]*sw[None, :])
        a_ptrs += 128; w_ptrs += 128
    o = acc.to(tl.bfloat16).to(tl.float32)
    if EVEN_M:
        rw = tl.load(RW + rm).to(tl.float32)
    else:
        rw = tl.load(RW + rm, mask=rm < M, other=0.0).to(tl.float32)
    o = o*rw[:, None]
    cp = C + rm[:, None]*scm + rn[None, :]
    if EVEN_M: tl.store(cp, o.to(tl.bfloat16))
    else: tl.store(cp, o.to(tl.bfloat16), mask=(rm < M)[:, None])
