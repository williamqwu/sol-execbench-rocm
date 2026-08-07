import torch
# --- inlined fp8_reference ---
import torch

from enum import StrEnum


class ScalingType(StrEnum):
    """
    Enum for different FP8 scaling strategies.

    Scaling types:
    - TensorWise: Global per-tensor scaling (no blocks)
    - RowWise: Per-row scaling (1 scale per row)
    - BlockWise1x16: 1x16 blocks (per-tensor in M, 16-sized blocks in K)
    - BlockWise1x32: 1x32 blocks (per-tensor in M, 32-sized blocks in K)
    - BlockWise1x128: 1x128 blocks (per-tensor in M, 128-sized blocks in K)
    - BlockWise128x128: 128x128 blocks (blockwise in both dimensions)
    """

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
    """
    Compute and apply scales for FP8 tensors.

    Supports various scaling strategies via ScalingType enum:
    - TensorWise: Global per-tensor scaling
    - RowWise: Per-row scaling
    - BlockWise1x16/32/128: Rectangular blocks
    - BlockWise128x128: Square blocks
    """

    E4M3_MAX = 448.0  # Maximum representable value in E4M3

    def __init__(self, scaling_type: ScalingType):
        """
        Initialize BlockwiseScaler with a specific scaling strategy.

        Args:
            scaling_type: ScalingType enum value
                Examples:
                - ScalingType.TensorWise -> global per-tensor scaling
                - ScalingType.RowWise -> per-row scaling (1 scale per row)
                - ScalingType.BlockWise1x128 -> 1x128 blocks
                - ScalingType.BlockWise128x128 -> 128x128 blocks
        """
        self.scaling_type = scaling_type
        self.shape = self.scaling_type.shape

        # Map enum to block dimensions (M, K)
        scaling_map = {
            ScalingType.TensorWise: (None, None),  # No blocking
            ScalingType.RowWise: (1, None),  # Per-row, full K dimension
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }

        self.block_size_m, self.block_size_k = scaling_map[scaling_type]

        # Keep for backward compatibility (use first dimension if available)
        self.block_size = self.block_size_m if self.block_size_m else None

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute scale factors based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Returns scalar tensor
        - RowWise: Returns (M,) tensor
        - BlockWise*: Returns (M//block_size_m, K//block_size_k) tensor

        Args:
            tensor: Input tensor (typically M, K for 2D)

        Returns:
            Scale tensor with shape depending on scaling type.
            These are inverse scales (amax / dtype_max) used for dequantization.
        """
        if self.scaling_type == ScalingType.TensorWise:
            # Global per-tensor scaling
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            # Per-row scaling: (M, K) -> (M,)
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        # BlockWise scaling
        assert M % self.block_size_m == 0, (
            f"M={M} must be a multiple of {self.block_size_m}"
        )
        assert K % self.block_size_k == 0, (
            f"K={K} must be a multiple of {self.block_size_k}"
        )

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

    def apply_scaling(
        self,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        inverse: bool = False,
        clamp_to_fp8_range: bool = False,
    ) -> torch.Tensor:
        """
        Apply scaling to tensor based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Uses scalar scale
        - RowWise: Uses per-row scales (M,)
        - BlockWise*: Uses blockwise scales (M//block_size_m, K//block_size_k)

        Args:
            tensor: Input tensor (typically M, K for 2D)
            scales: Scale tensor with shape depending on scaling type
                   These are inverse scales (amax / dtype_max)
            inverse: If True, multiply by scales (dequantization)
                    If False, divide by scales (quantization)
            clamp_to_fp8_range: If True, clamp to FP8 range before returning

        Returns:
            Scaled tensor (same shape as input)
        """
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
                tensor_scaled = torch.clamp(
                    tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX
                )

        return tensor_scaled.reshape(*old_shape)


class CuBLASRefBlockwiseGemm:
    """
    Reference implementation of blockwise-scaled GEMM via dequantize-then-matmul.
    """

    def scaled_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_recipe_a: ScalingType,
        scale_b: torch.Tensor,
        scale_recipe_b: ScalingType,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        """
        Scaled matrix multiplication: dequantize A and B, then matmul in float32.

        Args:
            mat_a: Input matrix A (M, K) in float8_e4m3fn
            mat_b: Input matrix B (N, K) in float8_e4m3fn
            scale_a: Scaling factors for A
            scale_recipe_a: Scaling type for A
            scale_b: Scaling factors for B
            scale_recipe_b: Scaling type for B
            bias: Optional bias vector (N,)
            output_dtype: Output data type
            use_fast_accum: Unused (kept for API compatibility)

        Returns:
            Result matrix (M, N) with dtype=output_dtype
        """
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

# --- end inlined fp8_reference ---



def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """
    Generate FP8 inputs with proper quantization and scales.
    
    Args:
        axes_and_scalars: Dictionary containing axis values
        device: Target device
    
    Returns:
        Dictionary of input tensors
    """
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    
    # Constants
    input_dim = 128 * 128  # 16384
    hidden_size = 7168
    
    # Generate random tensors in bfloat16
    hidden_states_bf16 = torch.randn(batch_size, seq_len, input_dim, dtype=torch.bfloat16, device=device)
    weight_bf16 = torch.randn(hidden_size, input_dim, dtype=torch.bfloat16, device=device)
    
    # Convert to FP32 for scale computation
    x_fp32 = hidden_states_bf16.view(-1, input_dim).to(torch.float32)  # (B*L, 16384)
    w_fp32 = weight_bf16.to(torch.float32)  # (7168, 16384)
    
    # Transpose weight for blockwise scale computation
    w_fp32_t = w_fp32.T  # (16384, 7168)
    
    # Compute scales using BlockwiseScaler
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    
    scale_x_2d = activation_scaler.compute_scales(x_fp32)  # (B*L, 128)
    scale_w = weight_scaler.compute_scales(w_fp32_t)  # (128, 56)
    
    # Reshape scale_x back to 3D
    scale_x = scale_x_2d.view(batch_size, seq_len, -1)  # (B, L, 128)
    
    # Quantize to FP8
    x_scaled = activation_scaler.apply_scaling(x_fp32, scale_x_2d, inverse=False, clamp_to_fp8_range=True)
    w_scaled = weight_scaler.apply_scaling(w_fp32_t, scale_w, inverse=False, clamp_to_fp8_range=True)
    
    hidden_states_fp8 = x_scaled.view(batch_size, seq_len, input_dim).to(torch.float8_e4m3fn)
    weight_fp8 = w_scaled.T.to(torch.float8_e4m3fn)  # (7168, 16384)
    
    return {
        "hidden_states": hidden_states_fp8,
        "weight": weight_fp8,
        "scale_x": scale_x,
        "scale_w": scale_w,
    }


import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BM": 256, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BM": 64, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 64, "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 256, "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 256, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 256, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 256, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def blockwise_fp8_gemm_kernel(
    a_ptr, b_ptr, sx_ptr, sw_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sxm, stride_sxk,
    stride_swk, stride_swn,
    stride_cm, stride_cn,
    BLOCK_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    # swizzle for L2
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % group_size_m))
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    num_k_blocks = K // BLOCK_K
    nb_idx = offs_n // 128  # loop-invariant

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    n_iters = num_k_blocks // (BK // BLOCK_K)
    for i in range(0, n_iters):
        for j in tl.static_range(0, BK // BLOCK_K):
            kb = i * (BK // BLOCK_K) + j
            k_start = kb * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_n[None, :] < N), other=0.0)
            d = tl.dot(a, b, out_dtype=tl.float32)  # (BM, BN)
            sx = tl.load(sx_ptr + offs_m * stride_sxm + kb * stride_sxk, mask=offs_m < M, other=0.0)
            sw = tl.load(sw_ptr + kb * stride_swk + nb_idx * stride_swn, mask=offs_n < N, other=0.0)
            scale = sx[:, None] * sw[None, :]
            acc += d * scale

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)


def blockwise_fp8_gemm(a, b, sx, sw, out=None):
    M, K = a.shape
    N, K2 = b.shape
    Kb = K // 128
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    blockwise_fp8_gemm_kernel[grid](
        a, b, sx, sw, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        sx.stride(0), sx.stride(1),
        sw.stride(0), sw.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=128,
    )
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
):
    """
    FP8 MLA attention output projection.
    """
    batch_size, seq_len, input_dim = hidden_states.shape
    hidden_size = weight.shape[0]

    x = hidden_states.view(-1, input_dim)  # (M, K)
    M = x.shape[0]

    # Activation scales: BlockWise1x128 -> (M, Kb)
    sx_2d = scale_x.view(M, -1).to(torch.float32)
    # Weight scales: scale_w is (Kb, Nb) = (128, 56); B is (N, K) BlockWise128x128
    sw_f32 = scale_w.to(torch.float32)

    output = blockwise_fp8_gemm(x, weight, sx_2d, sw_f32)  # (M, N)
    output = output.view(batch_size, seq_len, hidden_size)
    return output
