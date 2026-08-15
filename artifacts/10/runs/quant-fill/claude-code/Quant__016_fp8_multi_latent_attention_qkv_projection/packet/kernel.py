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

        assert M % self.block_size_m == 0, (
            f"M={M} must be a multiple of {self.block_size_m}"
        )
        assert K % self.block_size_k == 0, (
            f"K={K} must be a multiple of {self.block_size_k}"
        )

        new_shape = (
            M // self.block_size_m,
            self.block_size_m,
            K // self.block_size_k,
            self.block_size_k,
        )
        tensor_blocked = tensor.reshape(new_shape)

        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)

        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(
        self,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        inverse: bool = False,
        clamp_to_fp8_range: bool = False,
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
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)

        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)

        y = a_f32 @ b_f32.T

        if bias is not None and bias.numel():
            y = y + bias

        return y.to(output_dtype)


@triton.jit
def rms_norm_kernel(
    x_ptr, weight_ptr, out_ptr,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Compute variance
    acc = 0.0
    for start_n in range(0, N, BLOCK_SIZE):
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        x_vals = tl.load(x_row_ptr + n_offsets, mask=mask, other=0.0)
        x_vals_f32 = x_vals.to(tl.float32)
        acc += tl.sum(x_vals_f32 * x_vals_f32)

    variance = acc / N
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Normalize and scale
    for start_n in range(0, N, BLOCK_SIZE):
        n_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        x_vals = tl.load(x_row_ptr + n_offsets, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + n_offsets, mask=mask, other=0.0)

        x_vals_f32 = x_vals.to(tl.float32)
        weight_vals_f32 = weight_vals.to(tl.float32)

        normed = x_vals_f32 * rstd
        out_vals = (normed * weight_vals_f32).to(x_vals.dtype)

        tl.store(out_row_ptr + n_offsets, out_vals, mask=mask)


def _rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    M, N = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(min(N, 2048))
    grid = (M,)

    rms_norm_kernel[grid](
        x, weight, output,
        M, N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Use Triton for 2D tensors
    if x.ndim == 2:
        return _rms_norm_triton(x, weight, eps)

    # Fallback for 3D
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def _fp8_linear(x: torch.Tensor, weight: torch.Tensor,
                activation_scaler: BlockwiseScaler,
                weight_scaler: BlockwiseScaler,
                gemm_ref: CuBLASRefBlockwiseGemm) -> torch.Tensor:
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w

    x_fp32 = x.to(torch.float32)
    w_fp32 = weight.T.to(torch.float32)

    scale_x = activation_scaler.compute_scales(x_fp32)
    scale_w = weight_scaler.compute_scales(w_fp32)

    x_scaled = activation_scaler.apply_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    w_scaled = weight_scaler.apply_scaling(
        w_fp32, scale_w, inverse=False, clamp_to_fp8_range=True
    )

    qx = x_scaled.to(torch.float8_e4m3fn)
    qw = w_scaled.T.to(torch.float8_e4m3fn)

    scale_w_cublas = scale_w.T.contiguous()

    output = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True
    )

    return output


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    num_heads = 128
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    q_head_dim = 192

    bsz, q_len, hidden_size = hidden_states.shape

    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()

    hidden_flat = hidden_states.reshape(-1, hidden_size)

    # Q Path
    q_a = _fp8_linear(hidden_flat, q_a_proj_weight, activation_scaler, weight_scaler, gemm_ref)

    q_a = q_a.reshape(bsz, q_len, q_lora_rank)
    q_a_norm = _rms_norm(q_a, q_a_layernorm_weight, rms_norm_eps)
    q_a_norm_flat = q_a_norm.reshape(-1, q_lora_rank)

    q = _fp8_linear(q_a_norm_flat, q_b_proj_weight, activation_scaler, weight_scaler, gemm_ref)

    q = q.view(bsz, q_len, num_heads, q_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    # KV Path
    compressed_kv_with_pe_padded = _fp8_linear(hidden_flat, kv_a_proj_weight, activation_scaler, weight_scaler, gemm_ref)

    compressed_kv_with_pe = compressed_kv_with_pe_padded[:, :kv_lora_rank + qk_rope_head_dim]

    compressed_kv_with_pe = compressed_kv_with_pe.reshape(bsz, q_len, kv_lora_rank + qk_rope_head_dim)
    compressed_kv, k_pe = torch.split(compressed_kv_with_pe, [kv_lora_rank, qk_rope_head_dim], dim=-1)

    k_pe = k_pe.unsqueeze(2)

    compressed_kv_norm = _rms_norm(compressed_kv, kv_a_layernorm_weight, rms_norm_eps)
    compressed_kv_norm_flat = compressed_kv_norm.reshape(-1, kv_lora_rank)

    kv = _fp8_linear(compressed_kv_norm_flat, kv_b_proj_weight, activation_scaler, weight_scaler, gemm_ref)

    kv = kv.view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    return q_nope, q_pe, k_nope, k_pe, value_states
