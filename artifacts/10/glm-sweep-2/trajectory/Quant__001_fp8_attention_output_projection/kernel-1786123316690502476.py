import torch
import triton
import triton.language as tl

# --- inlined fp8_reference (kept for API parity) ---
import torch
from enum import StrEnum


class ScalingType(StrEnum):
    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"

    @property
    def shape(self) -> tuple[int, int]:
        return {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }[self]


class BlockwiseScaler:
    E4M3_MAX = 448.0

    def __init__(self, scaling_type: ScalingType):
        self.scaling_type = scaling_type
        self.shape = self.scaling_type.shape
        scaling_map = {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }
        self.block_size_m, self.block_size_k = scaling_map[scaling_type]
        self.block_size = self.block_size_m if self.block_size_m else None

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scaling_type == ScalingType.TensorWise:
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX
        M, K = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)
        assert M % self.block_size_m == 0
        assert K % self.block_size_k == 0
        new_shape = (M // self.block_size_m, self.block_size_m, K // self.block_size_k, self.block_size_k)
        tensor_blocked = tensor.reshape(new_shape)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(self, tensor, scales, inverse=False, clamp_to_fp8_range=False):
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            M, K = tensor.shape
            new_shape = (M // self.block_size_m, self.block_size_m, K // self.block_size_k, self.block_size_k)
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)
        if inverse:
            tensor_scaled = tensor * scales
        else:
            tensor_scaled = tensor / scales
            if clamp_to_fp8_range:
                tensor_scaled = torch.clamp(tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX)
        return tensor_scaled.reshape(*old_shape)


class CuBLASRefBlockwiseGemm:
    def scaled_mm(self, mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b,
                 bias=None, output_dtype=torch.bfloat16, use_fast_accum=True):
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)
        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)
        y = a_f32 @ b_f32.T
        if bias is not None and bias.numel():
            y = y + bias
        return y.to(output_dtype)

# --- end inlined fp8_reference ---


@triton.jit
def _qd_act_1x128(x_ptr, out_ptr, M, K, BLOCK_K: tl.constexpr):
    # Fused blockwise (1x128) FP8 quantize->dequantize for activations.
    # One program per (row, 128-element K block).
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = tl.arange(0, BLOCK_K)
    k_idx = kb * BLOCK_K + offs
    ptrs = x_ptr + row * K + k_idx
    mask = k_idx < K
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax, 1e-12) / 448.0
    q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
    dq = (q.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(out_ptr + row * K + k_idx, dq, mask=mask)


@triton.jit
def _qd_weight_128x128(w_ptr, out_ptr, N, K, BLOCK: tl.constexpr):
    # Fused blockwise (128x128) FP8 quantize->dequantize for weights.
    # w is (N, K) row-major; output is (K, N) row-major (transposed & dequantized).
    kb = tl.program_id(0)
    nb = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, BLOCK)
    k_idx = kb * BLOCK + offs_k
    n_idx = nb * BLOCK + offs_n
    ptrs = w_ptr + n_idx[:, None] * K + k_idx[None, :]
    mask = (n_idx[:, None] < N) & (k_idx[None, :] < K)
    w = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(w), axis=None)
    scale = tl.maximum(amax, 1e-12) / 448.0
    q = tl.clamp(w / scale, -448.0, 448.0).to(tl.float8e4nv)
    dq = (q.to(tl.float32) * scale).to(tl.bfloat16)
    out_ptrs = out_ptr + k_idx[:, None] * N + n_idx[None, :]
    tl.store(out_ptrs, tl.trans(dq), mask=(k_idx[:, None] < K) & (n_idx[None, :] < N))


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor,
):
    """
    FP8-quantized attention output projection.

    Computes the same blockwise-FP8 quantized GEMM as the reference:
    activations use BlockWise1x128 scaling, weights use BlockWise128x128.
    The FP8 quantize->dequantize is fused into Triton kernels producing bf16
    tensors, which are then multiplied with a bf16 GEMM (fp32 accumulate).
    """
    M, K = attn_output.shape
    N = o_proj_weight.shape[0]

    a_bf16 = torch.empty_like(attn_output)
    _qd_act_1x128[(M, K // 128)](attn_output, a_bf16, M, K, BLOCK_K=128)

    w_bf16 = torch.empty(K, N, dtype=torch.bfloat16, device=attn_output.device)
    _qd_weight_128x128[(K // 128, N // 128)](o_proj_weight, w_bf16, N, K, BLOCK=128)

    y = a_bf16 @ w_bf16
    if o_proj_bias is not None and o_proj_bias.numel():
        y = y + o_proj_bias
    return y
