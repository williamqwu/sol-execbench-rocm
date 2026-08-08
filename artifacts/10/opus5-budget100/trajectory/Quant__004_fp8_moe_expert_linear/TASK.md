# Optimize a GPU kernel for AMD Instinct MI350X

You are given a working PyTorch reference implementation. **Make it faster
while keeping it numerically correct.** Whatever is in `kernel.py` when you
stop is your submission.

## The hardware

AMD Instinct MI350X, CDNA4, `gfx950`. 256 CUs, 288 GB HBM3E, 8 TB/s.
ROCm 7.2, PyTorch 2.9.1. Clocks are locked at 1300 MHz, so timings are
repeatable — a change in measured latency is a real change.

This is **not** an NVIDIA GPU. A wavefront is 64 lanes, not 32. There are no
tensor cores in the CUDA sense; the matrix engine is MFMA. `torch.compile`,
Triton and HIP C++ all work. Do not assume a CUDA idiom transfers.

## Your loop

```bash
./evaluate          # evaluates kernel.py: correctness + latency vs reference
```

It prints one line per workload: PASS/FAIL, your latency, the reference
latency, and your speedup. **Every workload must PASS.** A faster kernel that
fails one workload scores zero — correctness is a gate, not a trade-off.

Run `./evaluate` as often as you like; it takes seconds to a couple of minutes.
Iterate: measure, change one thing, measure again.

## Rules

* `kernel.py` must define `run(...)` with the **same signature and return type**
  as the reference.
* Compute the real thing. Caching results across calls, returning inputs,
  writing into the output without computing it, or special-casing the specific
  shapes the harness happens to use are all detected and rejected by the
  harness's anti-reward-hack checks. A rejected submission scores zero.
* You may use `torch`, `torch.compile`, Triton, or hand-written HIP via
  `torch.utils.cpp_extension`. `aiter` and `hipblaslt` are installed.
* Numerical tolerance is fixed and generous but real: the harness compares
  against the reference with a per-workload atol/rtol derived on this hardware.
  Reordering a reduction is fine. Dropping precision to bf16 where the
  reference uses fp32 usually is not.

## The problem: `004_fp8_moe_expert_linear`

FP8-quantized MoE expert computation with fused gate-up projection, SiLU activation, and down projection. Uses blockwise FP8 quantization with BlockWise1x128 for activations and BlockWise128x128 for weights.

**Axes** (workload dimensions):

- `num_tokens` (varies per workload) — 
- `hidden_size` = 3584 (constant) — 
- `intermediate_size` = 2048 (constant) — 
- `double_intermediate_size` = `None` (derived) — 
- `hidden_blocks` = `None` (derived) — 
- `intermediate_blocks` = `None` (derived) — 
- `double_intermediate_blocks` = `None` (derived) — 

**Inputs**

- `hidden_states`: [num_tokens, hidden_size], `bfloat16` — Input hidden states
- `routing_weight`: [num_tokens, 1], `bfloat16` — Routing weight for this expert
- `gate_up_weight`: [double_intermediate_size, hidden_size], `bfloat16` — Gate-up projection weight matrix
- `down_weight`: [hidden_size, intermediate_size], `bfloat16` — Down projection weight matrix

**Outputs**

- `output`: [num_tokens, hidden_size], `bfloat16` — Expert output weighted by routing weight

**Workload shapes you will be evaluated on** (16 of them):

- num_tokens=3584
- num_tokens=2432
- num_tokens=1152
- num_tokens=1024
- num_tokens=2816
- num_tokens=3712
- num_tokens=640
- num_tokens=1792
- num_tokens=2048
- num_tokens=1920
- num_tokens=1536
- num_tokens=384
- num_tokens=2176
- num_tokens=896
- num_tokens=4096
- num_tokens=3072

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import torch
import torch.nn.functional as F
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
    """Generate inputs with scaled weights to keep outputs in reasonable range."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = 3584
    intermediate_size = 2048

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    routing_weight = torch.randn(num_tokens, 1, dtype=torch.bfloat16, device=device)
    gate_up_weight = torch.randn(intermediate_size * 2, hidden_size, dtype=torch.bfloat16, device=device) * (hidden_size ** -0.5)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) * (intermediate_size ** -0.5)

    return {
        "hidden_states": hidden_states,
        "routing_weight": routing_weight,
        "gate_up_weight": gate_up_weight,
        "down_weight": down_weight,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    """
    FP8-quantized MoE expert computation.
    
    1. Gate-up projection: hidden_states @ gate_up_weight.T -> [num_tokens, 2*intermediate_size]
    2. Split and apply SiLU activation: silu(gate) * up
    3. Down projection: gated_output @ down_weight.T -> [num_tokens, hidden_size]
    4. Apply routing weight
    """
    # Initialize scalers
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Step 1: Gate-up projection with FP8
    # Compute scales for input activation
    hidden_fp32 = hidden_states.to(torch.float32)
    scale_hidden = activation_scaler.compute_scales(hidden_fp32)
    
    # Compute scales for gate_up weight
    gate_up_weight_fp32 = gate_up_weight.to(torch.float32)
    gate_up_weight_t = gate_up_weight_fp32.T  # (hidden_size, 2*intermediate_size)
    scale_gate_up = weight_scaler.compute_scales(gate_up_weight_t)
    
    # Apply scaling and quantize
    hidden_scaled = activation_scaler.apply_scaling(
        hidden_fp32, scale_hidden, inverse=False, clamp_to_fp8_range=True
    )
    gate_up_scaled = weight_scaler.apply_scaling(
        gate_up_weight_t, scale_gate_up, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    hidden_fp8 = hidden_scaled.to(torch.float8_e4m3fn)
    gate_up_fp8 = gate_up_scaled.T.to(torch.float8_e4m3fn)  # Back to (2*intermediate, hidden)
    
    # Transpose weight scales for CuBLAS format
    scale_gate_up_cublas = scale_gate_up.T.contiguous()
    
    # FP8 GEMM: hidden @ gate_up.T -> [num_tokens, 2*intermediate_size]
    gate_up_output = gemm_ref.scaled_mm(
        mat_a=hidden_fp8,
        mat_b=gate_up_fp8,
        scale_a=scale_hidden,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_gate_up_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    # Step 2: Split and apply activation (NOT quantized)
    gate, up = gate_up_output.chunk(2, dim=-1)
    gated_output = F.silu(gate) * up  # SiLU activation on gate, element-wise multiply
    
    # Step 3: Down projection with FP8
    # Compute scales for gated output
    gated_fp32 = gated_output.to(torch.float32)
    scale_gated = activation_scaler.compute_scales(gated_fp32)
    
    # Compute scales for down weight
    down_weight_fp32 = down_weight.to(torch.float32)
    down_weight_t = down_weight_fp32.T  # (intermediate_size, hidden_size)
    scale_down = weight_scaler.compute_scales(down_weight_t)
    
    # Apply scaling and quantize
    gated_scaled = activation_scaler.apply_scaling(
        gated_fp32, scale_gated, inverse=False, clamp_to_fp8_range=True
    )
    down_scaled = weight_scaler.apply_scaling(
        down_weight_t, scale_down, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    gated_fp8 = gated_scaled.to(torch.float8_e4m3fn)
    down_fp8 = down_scaled.T.to(torch.float8_e4m3fn)  # Back to (hidden, intermediate)
    
    # Transpose weight scales for CuBLAS format
    scale_down_cublas = scale_down.T.contiguous()
    
    # FP8 GEMM: gated @ down.T -> [num_tokens, hidden_size]
    output = gemm_ref.scaled_mm(
        mat_a=gated_fp8,
        mat_b=down_fp8,
        scale_a=scale_gated,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_down_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    # Step 4: Apply routing weight (NOT quantized)
    weighted_output = output * routing_weight
    
    return weighted_output

```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
