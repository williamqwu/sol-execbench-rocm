import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_bwd(
    go_ptr, hs_ptr, mask_ptr, gate_ptr,
    gres_ptr, ghs_ptr, part_ptr,
    H, NBLK,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # tanh(gate) in fp32, then rounded to bf16 exactly as torch does when a
    # 0-dim fp32 tensor multiplies a bf16 tensor.
    gate = tl.load(gate_ptr).to(tl.float32)
    gv = libdevice.tanh(gate)
    gvb = gv.to(tl.bfloat16).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs_n < H
    base = pid_m * H + offs_n

    go = tl.load(go_ptr + base, mask=m, other=0.0)
    hs = tl.load(hs_ptr + base, mask=m, other=0.0)
    mk = tl.load(mask_ptr + pid_m).to(tl.float32)

    # grad_residual = grad_output.clone()
    tl.store(gres_ptr + base, go, mask=m)

    gof = go.to(tl.float32)

    # grad_hidden_states = ((go * bf16(tanh(gate))) -> bf16) * mask -> bf16
    t = (gof * gvb).to(tl.bfloat16).to(tl.float32)
    ghs = (t * mk).to(tl.bfloat16)
    tl.store(ghs_ptr + base, ghs, mask=m)

    # partial of sum(fp32(go) * fp32(bf16(hs * mask)))
    mh = (hs.to(tl.float32) * mk).to(tl.bfloat16).to(tl.float32)
    acc = tl.sum(gof * mh, axis=0)
    tl.store(part_ptr + pid_m * NBLK + pid_n, acc)


@triton.jit
def _reduce(part_ptr, gate_ptr, out_ptr, N, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, N, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        acc += tl.load(part_ptr + offs, mask=offs < N, other=0.0)
    total = tl.sum(acc, axis=0)

    gate = tl.load(gate_ptr).to(tl.float32)
    gv = libdevice.tanh(gate)
    sech2 = 1.0 - gv * gv
    tl.store(out_ptr, (total * sech2).to(tl.bfloat16))


def _pick_block(M, H):
    if H >= 4096:
        return 1024
    if H >= 1024:
        return 512
    return max(64, triton.next_power_of_2(H))


@torch.no_grad()
def run(grad_output, gate, hidden_states, mask):
    go = grad_output.contiguous()
    hs = hidden_states.contiguous()
    mk = mask.contiguous()

    H = go.shape[-1]
    M = go.numel() // H if H > 0 else 0

    grad_residual = torch.empty_like(go)
    grad_hidden_states = torch.empty_like(go)
    grad_gate = torch.empty((), dtype=torch.bfloat16, device=go.device)

    if M == 0 or H == 0:
        grad_gate.zero_()
        return grad_residual, grad_hidden_states, grad_gate

    BLOCK_N = _pick_block(M, H)
    NBLK = triton.cdiv(H, BLOCK_N)
    partials = torch.empty(M * NBLK, dtype=torch.float32, device=go.device)

    _fused_bwd[(M, NBLK)](
        go, hs, mk, gate,
        grad_residual, grad_hidden_states, partials,
        H, NBLK,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    _reduce[(1,)](partials, gate, grad_gate, M * NBLK, BLOCK=1024, num_warps=4)

    return grad_residual, grad_hidden_states, grad_gate
