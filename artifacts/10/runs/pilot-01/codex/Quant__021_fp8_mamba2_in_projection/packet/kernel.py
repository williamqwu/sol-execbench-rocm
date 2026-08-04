import torch
import aiter
import triton
import triton.language as tl


E4M3_MAX = 448.0
K_BLOCKS = 21
N_BLOCKS = 109
BLOCK = 128
_quant_1x128 = aiter.get_hip_quant(aiter.QuantType.per_1x128)


@triton.jit
def _weight_quant_kernel(
    weight,
    qweight,
    scale_w,
    KB: tl.constexpr,
    B: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    kb = pid % KB
    nb = pid // KB

    offs = tl.arange(0, BLOCK_ELEMS)
    rows = offs // B
    cols = offs - rows * B
    w_offsets = (nb * B + rows) * (KB * B) + kb * B + cols

    vals = tl.load(weight + w_offsets).to(tl.float32)
    abs_vals = tl.abs(vals)
    block_max = tl.max(abs_vals, axis=0)
    block_max = tl.maximum(block_max, 1.0e-12)
    scale = block_max / E4M3
    tl.store(scale_w + nb * KB + kb, scale)

    q = vals / scale
    q = tl.maximum(tl.minimum(q, E4M3), -E4M3)
    tl.store(qweight + w_offsets, q)


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    M = hidden_states.shape[0]

    qx, scale_x = _quant_1x128(hidden_states, quant_dtype=torch.float8_e4m3fn)
    scale_w = torch.empty((N_BLOCKS, K_BLOCKS), device=weight.device, dtype=torch.float32)
    qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    _weight_quant_kernel[(N_BLOCKS * K_BLOCKS,)](
        weight,
        qw,
        scale_w,
        KB=K_BLOCKS,
        B=BLOCK,
        E4M3=E4M3_MAX,
        BLOCK_ELEMS=BLOCK * BLOCK,
        num_warps=8,
    )

    out = torch.empty((M, N_BLOCKS * BLOCK), device=hidden_states.device, dtype=torch.bfloat16)
    return aiter.gemm_a8w8_blockscale_cktile(qx, qw, scale_x, scale_w, out)
