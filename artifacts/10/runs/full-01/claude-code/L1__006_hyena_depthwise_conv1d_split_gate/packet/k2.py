"""Candidate v2: S as constexpr (fewer runtime args), optional interior-block branch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_v2(
    u_ptr, w_ptr, b_ptr, o_ptr, TDS,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NSB: tl.constexpr,
    EVEN: tl.constexpr,
    BRANCH: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    DS: tl.constexpr = 256 * S
    BS3: tl.constexpr = 768 * S

    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    u_row = u_ptr + pid_b * BS3 + pid_c * S
    o_off = pid_b * DS + pid_c * S

    wo0 = pid_c * 3
    w00 = tl.load(w_ptr + wo0 + 0).to(tl.float64)
    w01 = tl.load(w_ptr + wo0 + 1).to(tl.float64)
    w02 = tl.load(w_ptr + wo0 + 2).to(tl.float64)
    bb0 = tl.load(b_ptr + pid_c).to(tl.float64)
    w10 = tl.load(w_ptr + wo0 + 768 + 0).to(tl.float64)
    w11 = tl.load(w_ptr + wo0 + 768 + 1).to(tl.float64)
    w12 = tl.load(w_ptr + wo0 + 768 + 2).to(tl.float64)
    bb1 = tl.load(b_ptr + pid_c + 256).to(tl.float64)
    w20 = tl.load(w_ptr + wo0 + 1536 + 0).to(tl.float64)
    w21 = tl.load(w_ptr + wo0 + 1536 + 1).to(tl.float64)
    w22 = tl.load(w_ptr + wo0 + 1536 + 2).to(tl.float64)
    bb2 = tl.load(b_ptr + pid_c + 512).to(tl.float64)

    p1 = u_row + DS
    p2 = u_row + 2 * DS

    if BRANCH and EVEN and (pid_s > 0):
        a00 = tl.load(u_row + offs).to(tl.float64)
        a01 = tl.load(u_row + offs - 1).to(tl.float64)
        a02 = tl.load(u_row + offs - 2).to(tl.float64)
        a10 = tl.load(p1 + offs).to(tl.float64)
        a11 = tl.load(p1 + offs - 1).to(tl.float64)
        a12 = tl.load(p1 + offs - 2).to(tl.float64)
        a20 = tl.load(p2 + offs).to(tl.float64)
        a21 = tl.load(p2 + offs - 1).to(tl.float64)
        a22 = tl.load(p2 + offs - 2).to(tl.float64)
        acc0 = w00 * a02 + w01 * a01 + w02 * a00 + bb0
        acc1 = w10 * a12 + w11 * a11 + w12 * a10 + bb1
        acc2 = w20 * a22 + w21 * a21 + w22 * a20 + bb2
        tl.store(o_ptr + o_off + offs, (acc2 * acc0).to(tl.float32))
        tl.store(o_ptr + TDS + o_off + offs, acc0.to(tl.float32))
        tl.store(o_ptr + 2 * TDS + o_off + offs, acc1.to(tl.float32))
    else:
        if EVEN:
            m0 = offs < 1000000000
        else:
            m0 = offs < S
        m1 = (offs >= 1) & m0
        m2 = (offs >= 2) & m0
        a00 = tl.load(u_row + offs, mask=m0, other=0.0).to(tl.float64)
        a01 = tl.load(u_row + offs - 1, mask=m1, other=0.0).to(tl.float64)
        a02 = tl.load(u_row + offs - 2, mask=m2, other=0.0).to(tl.float64)
        a10 = tl.load(p1 + offs, mask=m0, other=0.0).to(tl.float64)
        a11 = tl.load(p1 + offs - 1, mask=m1, other=0.0).to(tl.float64)
        a12 = tl.load(p1 + offs - 2, mask=m2, other=0.0).to(tl.float64)
        a20 = tl.load(p2 + offs, mask=m0, other=0.0).to(tl.float64)
        a21 = tl.load(p2 + offs - 1, mask=m1, other=0.0).to(tl.float64)
        a22 = tl.load(p2 + offs - 2, mask=m2, other=0.0).to(tl.float64)
        acc0 = w00 * a02 + w01 * a01 + w02 * a00 + bb0
        acc1 = w10 * a12 + w11 * a11 + w12 * a10 + bb1
        acc2 = w20 * a22 + w21 * a21 + w22 * a20 + bb2
        tl.store(o_ptr + o_off + offs, (acc2 * acc0).to(tl.float32), mask=m0)
        tl.store(o_ptr + TDS + o_off + offs, acc0.to(tl.float32), mask=m0)
        tl.store(o_ptr + 2 * TDS + o_off + offs, acc1.to(tl.float32), mask=m0)


# fp32 variant -- SPEED REFERENCE ONLY (known to fail tolerance on v_gated)
@triton.jit
def _fused_f32(u_ptr, w_ptr, b_ptr, o_ptr, TDS, S: tl.constexpr,
               BLOCK_S: tl.constexpr, NSB: tl.constexpr, EVEN: tl.constexpr,
               BRANCH: tl.constexpr):
    pid_s = tl.program_id(0); pid_c = tl.program_id(1); pid_b = tl.program_id(2)
    DS: tl.constexpr = 256 * S
    BS3: tl.constexpr = 768 * S
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    u_row = u_ptr + pid_b * BS3 + pid_c * S
    o_off = pid_b * DS + pid_c * S
    m0 = offs < S
    m1 = (offs >= 1) & m0
    m2 = (offs >= 2) & m0
    wo0 = pid_c * 3
    p1 = u_row + DS; p2 = u_row + 2 * DS
    a00 = tl.load(u_row + offs, mask=m0, other=0.0)
    a01 = tl.load(u_row + offs - 1, mask=m1, other=0.0)
    a02 = tl.load(u_row + offs - 2, mask=m2, other=0.0)
    a10 = tl.load(p1 + offs, mask=m0, other=0.0)
    a11 = tl.load(p1 + offs - 1, mask=m1, other=0.0)
    a12 = tl.load(p1 + offs - 2, mask=m2, other=0.0)
    a20 = tl.load(p2 + offs, mask=m0, other=0.0)
    a21 = tl.load(p2 + offs - 1, mask=m1, other=0.0)
    a22 = tl.load(p2 + offs - 2, mask=m2, other=0.0)
    acc0 = (tl.load(w_ptr + wo0 + 0)*a02 + tl.load(w_ptr + wo0 + 1)*a01
            + tl.load(w_ptr + wo0 + 2)*a00 + tl.load(b_ptr + pid_c))
    acc1 = (tl.load(w_ptr + wo0 + 768)*a12 + tl.load(w_ptr + wo0 + 769)*a11
            + tl.load(w_ptr + wo0 + 770)*a10 + tl.load(b_ptr + pid_c + 256))
    acc2 = (tl.load(w_ptr + wo0 + 1536)*a22 + tl.load(w_ptr + wo0 + 1537)*a21
            + tl.load(w_ptr + wo0 + 1538)*a20 + tl.load(b_ptr + pid_c + 512))
    tl.store(o_ptr + o_off + offs, acc2 * acc0, mask=m0)
    tl.store(o_ptr + TDS + o_off + offs, acc0, mask=m0)
    tl.store(o_ptr + 2 * TDS + o_off + offs, acc1, mask=m0)
