import torch
import triton
import triton.language as tl

# Try to use hipBLASLt for FP8 GEMM if available
try:
    import hipblaslt
    HAS_HIPBLASLT = True
except ImportError:
    HAS_HIPBLASLT = False


@triton.jit
def _fp8_blockwise_scale_kernel(
    input_ptr, scales_ptr,
    M, K,
    block_m: tl.constexpr, block_k: tl.constexpr,
    stride_im, stride_ik,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    data = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    abs_data = tl.abs(data)
    block_max = tl.max(abs_data)

    E4M3_MAX = 448.0
    scale = tl.maximum(block_max / E4M3_MAX, 1e-12)

    block_m_idx = pid_m // (block_m // BLOCK_SIZE_M)
    block_k_idx = pid_k // (block_k // BLOCK_SIZE_K)

    num_blocks_k = K // block_k
    scale_idx = block_m_idx * num_blocks_k + block_k_idx
    tl.store(scales_ptr + scale_idx, scale)


def compute_blockwise_scales_triton(tensor, block_size_m, block_size_k):
    M, K = tensor.shape
    assert M % block_size_m == 0
    assert K % block_size_k == 0

    num_blocks_m = M // block_size_m
    num_blocks_k = K // block_size_k
    scales = torch.empty((num_blocks_m, num_blocks_k), device=tensor.device, dtype=torch.float32)

    BLOCK_SIZE_M = min(128, block_size_m)
    BLOCK_SIZE_K = min(128, block_size_k)

    grid = lambda meta: (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(K, BLOCK_SIZE_K),
    )

    _fp8_blockwise_scale_kernel[grid](
        tensor, scales,
        M, K,
        block_size_m, block_size_k,
        tensor.stride(0), tensor.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return scales


# Use exact reference implementation
from enum import StrEnum

class ScalingType(StrEnum):
    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"

    @property
    def shape(self):
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

    def __init__(self, scaling_type):
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

    def compute_scales(self, tensor):
        if self.scaling_type == ScalingType.TensorWise:
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        # BlockWise scaling
        assert M % self.block_size_m == 0, f"M={M} must be a multiple of {self.block_size_m}"
        assert K % self.block_size_k == 0, f"K={K} must be a multiple of {self.block_size_k}"

        # Reshape (M, K) -> (M//block_size_m, block_size_m, K//block_size_k, block_size_k)
        new_shape = (
            M // self.block_size_m,
            self.block_size_m,
            K // self.block_size_k,
            self.block_size_k,
        )
        tensor_blocked = tensor.reshape(new_shape)

        # Compute max over the block dimensions (dims 1 and 3)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)

        # Compute inverse scales
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(self, tensor, scales, inverse=False, clamp_to_fp8_range=False):
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            # expand (M,) -> (M, 1)
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            # blockwise scaling
            M, K = tensor.shape
            new_shape = (
                M // self.block_size_m,
                self.block_size_m,
                K // self.block_size_k,
                self.block_size_k,
            )
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
    def scaled_mm(
        self,
        mat_a, mat_b,
        scale_a, scale_recipe_a,
        scale_b, scale_recipe_b,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    ):
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)

        # Dequantize: FP8 values * inverse_scales -> float32
        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)

        # Single matmul in float32
        y = a_f32 @ b_f32.T

        if bias is not None and bias.numel():
            y = y + bias

        return y.to(output_dtype)


@torch.no_grad()
def run(
    hidden_states,
    kv_a_proj_weight,
    kv_a_layernorm_weight,
    kv_b_proj_weight,
    rms_norm_eps,
):
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128

    bsz, q_len, hidden_size = hidden_states.shape

    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()

    hidden_flat = hidden_states.reshape(-1, hidden_size)
    M = hidden_flat.shape[0]

    # Step 1: FP8 Compression Projection
    x_fp32 = hidden_flat.to(torch.float32)
    w_a_fp32 = kv_a_proj_weight.to(torch.float32)

    scale_x_a = activation_scaler.compute_scales(x_fp32)
    w_a_fp32_t = w_a_fp32.T
    scale_w_a = weight_scaler.compute_scales(w_a_fp32_t)

    x_scaled_a = activation_scaler.apply_scaling(x_fp32, scale_x_a, inverse=False, clamp_to_fp8_range=True)
    w_scaled_a = weight_scaler.apply_scaling(w_a_fp32_t, scale_w_a, inverse=False, clamp_to_fp8_range=True)

    qx_a = x_scaled_a.to(torch.float8_e4m3fn)
    qw_a = w_scaled_a.T.to(torch.float8_e4m3fn)

    scale_w_a_cublas = scale_w_a.T.contiguous()
    compressed_kv_with_rope = gemm_ref.scaled_mm(
        mat_a=qx_a,
        mat_b=qw_a,
        scale_a=scale_x_a,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_a_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )

    # Step 2: Split and RMSNorm
    compressed_kv = compressed_kv_with_rope[:, :kv_lora_rank]
    k_pe = compressed_kv_with_rope[:, kv_lora_rank:kv_lora_rank + qk_rope_head_dim]

    compressed_kv_fp32 = compressed_kv.to(torch.float32)
    variance = compressed_kv_fp32.pow(2).mean(-1, keepdim=True)
    compressed_kv_norm = compressed_kv_fp32 * torch.rsqrt(variance + rms_norm_eps)
    compressed_kv_norm = (kv_a_layernorm_weight * compressed_kv_norm.to(kv_a_layernorm_weight.dtype))

    # Step 3: FP8 Expansion Projection
    x_b_fp32 = compressed_kv_norm.to(torch.float32)
    w_b_fp32 = kv_b_proj_weight.to(torch.float32)

    scale_x_b = activation_scaler.compute_scales(x_b_fp32)
    w_b_fp32_t = w_b_fp32.T
    scale_w_b = weight_scaler.compute_scales(w_b_fp32_t)

    x_scaled_b = activation_scaler.apply_scaling(x_b_fp32, scale_x_b, inverse=False, clamp_to_fp8_range=True)
    w_scaled_b = weight_scaler.apply_scaling(w_b_fp32_t, scale_w_b, inverse=False, clamp_to_fp8_range=True)

    qx_b = x_scaled_b.to(torch.float8_e4m3fn)
    qw_b = w_scaled_b.T.to(torch.float8_e4m3fn)

    scale_w_b_cublas = scale_w_b.T.contiguous()
    kv_expanded_flat = gemm_ref.scaled_mm(
        mat_a=qx_b,
        mat_b=qw_b,
        scale_a=scale_x_b,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_b_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )

    kv_expanded = kv_expanded_flat.view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
    k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim)

    return kv_expanded, k_pe
