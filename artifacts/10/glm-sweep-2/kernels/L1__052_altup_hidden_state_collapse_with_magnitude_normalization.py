import torch
import triton
import triton.language as tl


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    unembed_proj_1: torch.Tensor,
    unembed_proj_2: torch.Tensor,
    epsilon: float,
):
    B, S, H = hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3]
    dev = hidden_states.device

    first = hidden_states[0].reshape(B * S, H)                       # bf16 [B*S,H]
    p1 = torch.matmul(hidden_states[1].reshape(B * S, H), unembed_proj_1.t())  # bf16 [B*S,H]
    p2 = torch.matmul(hidden_states[2].reshape(B * S, H), unembed_proj_2.t())  # bf16 [B*S,H]

    out = torch.empty((B * S, H), dtype=torch.bfloat16, device=dev)
    grid = (B * S,)
    _fuse_kernel[grid](first, p1, p2, out, epsilon, H, BLOCK=4096)
    return out.view(B, S, H)


@triton.jit
def _fuse_kernel(
    first_ptr, p1_ptr, p2_ptr, out_ptr,
    epsilon,
    H,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    base = row * H

    first = tl.load(first_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    p1 = tl.load(p1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    p2 = tl.load(p2_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    target_mag = tl.sqrt(tl.sum(first * first, axis=0) / H)
    mag1 = tl.sqrt(tl.maximum(tl.sum(p1 * p1, axis=0) / H, epsilon))
    mag2 = tl.sqrt(tl.maximum(tl.sum(p2 * p2, axis=0) / H, epsilon))

    n1 = p1 * (target_mag / mag1)
    n2 = p2 * (target_mag / mag2)
    out = (first + n1 + n2) * (1.0 / 3.0)

    tl.store(out_ptr + base + offs, out.to(tl.bfloat16), mask=mask)
