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


E4M3_MAX = 448.0


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    attn_output = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device)
    attn_output_fp32 = attn_output.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    weight_fp32_t = weight_fp32.T
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    scale_attn = activation_scaler.compute_scales(attn_output_fp32)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    scale_weight = weight_scaler.compute_scales(weight_fp32_t)
    return {
        "attn_output": attn_output,
        "weight": weight,
        "scale_attn": scale_attn,
        "scale_weight": scale_weight,
    }


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


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    scale_attn: torch.Tensor,
    scale_weight: torch.Tensor,
):
    M, K = attn_output.shape
    N, K_w = weight.shape

    x_fp32 = attn_output.to(torch.float32)
    w_fp32 = weight.to(torch.float32)
    w_fp32_t = w_fp32.T  # (K, N)

    # Quantize activation (1x128): divide by scale, clamp, cast to fp8
    x_scaled = (x_fp32.reshape(M, K // 128, 128) / scale_attn.unsqueeze(2)).reshape(M, K).clamp(-E4M3_MAX, E4M3_MAX)
    qx = x_scaled.to(torch.float8_e4m3fn)  # (M, K)

    # Quantize weight (128x128): scale_weight is (K//128, N//128) computed on w_fp32_t (K, N)
    w_scaled = (w_fp32_t.reshape(K // 128, 128, N // 128, 128) / scale_weight.unsqueeze(1).unsqueeze(3)).reshape(K, N).clamp(-E4M3_MAX, E4M3_MAX)
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # (N, K)

    # Dequantize: a_f32[m,k] = qx[m,k] * scale_attn[m, k//128]
    a_f32 = (qx.to(torch.float32).reshape(M, K // 128, 128) * scale_attn.unsqueeze(2)).reshape(M, K)
    # Dequantize: b_f32[n,k] = qw[n,k] * scale_weight[k//128, n//128]
    sb = scale_weight.T.contiguous()  # (N//128, K//128)
    b_f32 = (qw.to(torch.float32).reshape(N, N // 128, K // 128, 128).permute(0, 2, 1, 3).reshape(N, N // 128, K // 128, 128) * sb.unsqueeze(1).unsqueeze(3)).reshape(N, N // 128, K // 128, 128)
    # simpler: expand scales
    b_f32 = qw.to(torch.float32) * sb.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)

    torch.backends.cuda.matmul.allow_tf32 = True
    y = (a_f32 @ b_f32.T).to(attn_output.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    return y
