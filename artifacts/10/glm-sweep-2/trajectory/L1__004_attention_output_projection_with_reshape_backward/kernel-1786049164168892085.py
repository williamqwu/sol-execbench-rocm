import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(in_ptr, out_ptr, M, N,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_ptrs = in_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(in_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(out_ptrs, tl.trans(x),
             mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))


def _triton_transpose(x):
    M, N = x.shape
    out = torch.empty(N, M, device=x.device, dtype=x.dtype)
    BM, BN = 64, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _transpose_kernel[grid](x, out, M, N, BM, BN)
    return out


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    head_dim = 64

    grad_output_2d = grad_output.reshape(-1, hidden_size)
    reshaped_2d = reshaped.reshape(-1, hidden_size)

    bs = grad_output_2d.shape[0]
    if bs >= 4096:
        # Fast Triton transpose + NN GEMM beats the TN GEMM for large BS.
        go_t = _triton_transpose(grad_output_2d)
        grad_weight = go_t.mm(reshaped_2d)
    else:
        grad_weight = grad_output_2d.t().mm(reshaped_2d)

    grad_reshaped_2d = grad_output_2d.mm(weight)

    grad_transposed = grad_reshaped_2d.reshape(batch_size, seq_len, num_heads, head_dim)
    grad_attn_output = grad_transposed.transpose(1, 2)

    return grad_attn_output, grad_weight
