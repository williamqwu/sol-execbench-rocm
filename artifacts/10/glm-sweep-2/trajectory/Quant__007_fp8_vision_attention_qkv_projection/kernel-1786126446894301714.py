import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _act_quant_dequant_kernel(
    x_ptr, out_ptr,
    M, K,
    x_stride_m, x_stride_k,
    o_stride_m, o_stride_k,
    BLOCK: tl.constexpr,
):
    """Fuse 1x128 blockwise quant+dequant: x(fp32) -> scale -> fp8 -> dequant -> bf16.
    Each program handles one (row, block) of 128 elements."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs = tl.arange(0, BLOCK)  # 128
    x_ptrs = x_ptr + pid_m * x_stride_m + (pid_k * BLOCK + offs) * x_stride_k
    x = tl.load(x_ptrs).to(tl.float32)
    # block max over 128
    block_max = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(block_max / 448.0, 1e-12)
    # quantize
    x_q = x / scale
    x_q = tl.clamp(x_q, -448.0, 448.0)
    # round to fp8 e4m3: cast
    x_fp8 = x_q.to(tl.float8e4nv)
    # dequant
    x_dq = x_fp8.to(tl.float32) * scale
    out_ptrs = out_ptr + pid_m * o_stride_m + (pid_k * BLOCK + offs) * o_stride_k
    tl.store(out_ptrs, x_dq.to(tl.bfloat16))


@triton.jit
def _weight_quant_dequant_kernel(
    w_ptr, out_ptr,
    K, N,
    w_stride_k, w_stride_n,
    o_stride_n, o_stride_k,
    BLOCK: tl.constexpr,
):
    """Fuse 128x128 blockwise quant+dequant on weight (K,N) -> output (N,K) bf16.
    Each program handles one 128x128 block. Output is transposed (N,K)."""
    pid_k = tl.program_id(0)  # K//128
    pid_n = tl.program_id(1)  # N//128
    offs_k = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    # load 128x128 block of w (K,N)
    w_ptrs = (w_ptr
              + (pid_k * BLOCK + offs_k[:, None]) * w_stride_k
              + (pid_n * BLOCK + offs_n[None, :]) * w_stride_n)
    w = tl.load(w_ptrs).to(tl.float32)
    block_max = tl.max(tl.abs(w), axis=0)  # over K (axis 0) -> (128,)
    block_max = tl.max(block_max, axis=0)  # scalar
    scale = tl.maximum(block_max / 448.0, 1e-12)
    w_q = w / scale
    w_q = tl.clamp(w_q, -448.0, 448.0)
    w_fp8 = w_q.to(tl.float8e4nv)
    w_dq = w_fp8.to(tl.float32) * scale
    # store transposed: out (N,K). out[n,k] = w_dq[k,n]
    out_ptrs = (out_ptr
                + (pid_n * BLOCK + offs_n[:, None]) * o_stride_n
                + (pid_k * BLOCK + offs_k[None, :]) * o_stride_k)
    tl.store(out_ptrs, w_dq.T.to(tl.bfloat16))


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    num_heads = 16
    head_dim = 96
    seq_length = hidden_states.shape[0]

    x_f32 = hidden_states.to(torch.float32).contiguous()
    M, K = x_f32.shape

    a_bf16 = torch.empty((M, K), device=x_f32.device, dtype=torch.bfloat16)
    grid = (M, K // 128)
    _act_quant_dequant_kernel[grid](
        x_f32, a_bf16, M, K,
        x_f32.stride(0), x_f32.stride(1),
        a_bf16.stride(0), a_bf16.stride(1),
        BLOCK=128,
    )

    # weight: (N, K_in) = (4608, 1536). Transpose to (K, N) for blocking.
    w_fp32 = qkv_weight.T.to(torch.float32).contiguous()  # (K, N)
    K2, N = w_fp32.shape
    b_bf16 = torch.empty((N, K2), device=w_fp32.device, dtype=torch.bfloat16)
    grid_w = (K2 // 128, N // 128)
    _weight_quant_dequant_kernel[grid_w](
        w_fp32, b_bf16, K2, N,
        w_fp32.stride(0), w_fp32.stride(1),
        b_bf16.stride(0), b_bf16.stride(1),
        BLOCK=128,
    )

    qkv = a_bf16 @ b_bf16.t()
    qkv = qkv + qkv_bias

    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=1)
    return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()
