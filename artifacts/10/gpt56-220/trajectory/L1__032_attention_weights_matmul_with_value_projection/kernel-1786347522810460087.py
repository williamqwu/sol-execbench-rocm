import torch
import triton
import triton.language as tl


@triton.jit
def _attention_value_kernel(a_ptr, v_ptr, out_ptr, M: tl.constexpr,
                            BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    ks = tl.arange(0, BLOCK_K)

    a_base = a_ptr + pid_bh * M * M
    v_base = v_ptr + pid_bh * M * 128
    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for k0 in range(0, M, BLOCK_K):
        a = tl.load(a_base + rows[:, None] * M + k0 + ks[None, :],
                    mask=(rows[:, None] < M) & (k0 + ks[None, :] < M), other=0.0)
        v = tl.load(v_base + (k0 + ks[:, None]) * 128 + cols[None, :],
                    mask=k0 + ks[:, None] < M, other=0.0)
        acc += tl.dot(a, v)

    batch = pid_bh // 40
    head = pid_bh % 40
    out_offsets = batch * M * 5120 + rows[:, None] * 5120 + head * 128 + cols[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=rows[:, None] < M)


@torch.no_grad()
def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    batch, _, seq_len, _ = attn_weights.shape
    output = torch.empty((batch, seq_len, 5120), device=attn_weights.device,
                         dtype=attn_weights.dtype)
    _attention_value_kernel[(triton.cdiv(seq_len, 32), batch * 40)](
        attn_weights, value_states, output, M=seq_len, BLOCK_M=32, BLOCK_K=32,
        num_warps=8,
    )
    return output
