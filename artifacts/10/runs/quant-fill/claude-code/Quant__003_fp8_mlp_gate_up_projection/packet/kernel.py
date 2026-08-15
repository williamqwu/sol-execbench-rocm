import torch
import triton
import triton.language as tl
from enum import StrEnum


class ScalingType(StrEnum):
    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"


class BlockwiseScaler:
    E4M3_MAX = 448.0

    def __init__(self, scaling_type: ScalingType):
        self.scaling_type = scaling_type
        scaling_map = {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }
        self.block_size_m, self.block_size_k = scaling_map[scaling_type]

    def apply_scaling(
        self, tensor: torch.Tensor, scales: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            M, K = tensor.shape
            new_shape = (
                M // self.block_size_m,
                self.block_size_m,
                K // self.block_size_k,
                self.block_size_k,
            )
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)

        tensor_scaled = tensor * scales if inverse else tensor / scales
        return tensor_scaled.reshape(*old_shape)


@triton.jit
def silu_mul_kernel(
    gate_ptr, up_ptr, output_ptr,
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused SiLU and element-wise multiply: output = silu(gate) * up"""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Compute pointers
    gate_ptrs = gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    up_ptrs = up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load inputs
    gate_block = tl.load(gate_ptrs, mask=mask, other=0.0)
    up_block = tl.load(up_ptrs, mask=mask, other=0.0)

    # Convert to float32 for computation
    gate_f32 = gate_block.to(tl.float32)
    up_f32 = up_block.to(tl.float32)

    # SiLU: x * sigmoid(x)
    # Use fast math approximation for better performance
    gate_sigmoid = tl.sigmoid(gate_f32)
    gate_activated = gate_f32 * gate_sigmoid

    # Element-wise multiply
    output = gate_activated * up_f32

    # Store as bfloat16
    tl.store(output_ptrs, output.to(tl.bfloat16), mask=mask)


def run(
    x: torch.Tensor,
    scale_x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    scale_gate: torch.Tensor,
    up_proj_weight: torch.Tensor,
    scale_up: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 fused gate-up projection with SiLU activation.

    Computes: output = silu(gate_proj(x)) * up_proj(x)

    Follows reference semantics: dequantize -> matmul in fp32 -> convert to bf16 -> fuse silu+mul
    """
    M, K = x.shape
    N, _ = gate_proj_weight.shape

    scaler_x = BlockwiseScaler(ScalingType.BlockWise1x128)
    scaler_weight = BlockwiseScaler(ScalingType.BlockWise128x128)

    # Dequantize inputs - do this once for both matmuls
    x_f32 = scaler_x.apply_scaling(x.to(torch.float32), scale_x, inverse=True)

    # Dequantize gate weight
    gate_f32 = scaler_weight.apply_scaling(
        gate_proj_weight.to(torch.float32), scale_gate, inverse=True
    )

    # Dequantize up weight
    up_f32 = scaler_weight.apply_scaling(
        up_proj_weight.to(torch.float32), scale_up, inverse=True
    )

    # Perform matmuls in float32 and convert to bfloat16
    # Use @ operator which may call optimized BLAS
    gate_output = (x_f32 @ gate_f32.T).to(torch.bfloat16)
    up_output = (x_f32 @ up_f32.T).to(torch.bfloat16)

    # Fuse SiLU and multiply in Triton
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    # Tune block sizes for better occupancy
    BLOCK_M = 64
    BLOCK_N = 256
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    silu_mul_kernel[grid](
        gate_output, up_output, output,
        M, N,
        gate_output.stride(0), gate_output.stride(1),
        up_output.stride(0), up_output.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output
