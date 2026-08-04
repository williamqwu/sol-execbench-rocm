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

## The problem: `053_gaussian_topk_sparse_activation`

Gaussian-based top-k sparse activation mechanism. Computes adaptive sparsity by calculating a threshold based on input's mean and standard deviation, scaled by the inverse CDF of a normal distribution at the target sparsity level. Applies ReLU with this dynamic cutoff to create learned sparse activations.

**Axes** (workload dimensions):

- `batch_size` (varies per workload) — Batch size
- `seq_len` (varies per workload) — Sequence length
- `intermediate_size` (varies per workload) — Intermediate/hidden dimension for the activation

**Inputs**

- `inputs`: [batch_size, seq_len, intermediate_size], `bfloat16` — Input tensor, typically output of gate_proj in MLP
- `target_sparsity`: scalar, `float32` — Target sparsity level in [0, 1]. 0.0 means no sparsity, higher values mean more aggressive sparsity.

**Outputs**

- `output`: [batch_size, seq_len, intermediate_size], `bfloat16` — Sparsified tensor with values below adaptive threshold set to zero

**Workload shapes you will be evaluated on** (12 of them):

- batch_size=1, seq_len=512, intermediate_size=12288
- batch_size=4, seq_len=2048, intermediate_size=12288
- batch_size=32, seq_len=128, intermediate_size=12288
- batch_size=2, seq_len=211, intermediate_size=8192
- batch_size=1, seq_len=8192, intermediate_size=4096
- batch_size=1, seq_len=1024, intermediate_size=16384
- batch_size=16, seq_len=1163, intermediate_size=8192
- batch_size=4, seq_len=541, intermediate_size=8192
- batch_size=4, seq_len=449, intermediate_size=4096
- batch_size=64, seq_len=1024, intermediate_size=8192
- batch_size=2, seq_len=131, intermediate_size=4096
- batch_size=2, seq_len=293, intermediate_size=12288

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import torch
import torch.nn.functional as F
import math


def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
    This is a rational approximation that works well for p in (0, 1).
    """
    # Constants for the approximation
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00
    
    p_low = 0.02425
    p_high = 1.0 - p_low
    
    result = torch.zeros_like(p)
    
    # Lower region
    mask_low = p < p_low
    if mask_low.any():
        q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
        result[mask_low] = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                           ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    
    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid.any():
        q = p[mask_mid] - 0.5
        r = q * q
        result[mask_mid] = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / \
                           (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    
    # Upper region
    mask_high = p > p_high
    if mask_high.any():
        q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
        result[mask_high] = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    
    return result


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Gaussian-based top-k sparse activation.
    
    Computes adaptive sparsity threshold based on input statistics:
    1. Compute mean and std of input across feature dimension
    2. Calculate threshold = mean + std * norm.icdf(target_sparsity)
    3. Apply ReLU(input - threshold) to create sparse activations
    
    Args:
        inputs: Input tensor of shape [batch_size, seq_len, intermediate_size]
        target_sparsity: Float in [0, 1] indicating target sparsity level.
                        0.0 means no sparsity (all activations pass through).
    
    Returns:
        Sparsified tensor of same shape as input.
    """
    # Early return if no sparsity requested
    if target_sparsity == 0.0:
        return inputs
    
    # Convert to float32 for numerical stability in statistics computation
    inputs_f32 = inputs.to(torch.float32)
    
    # Compute statistics along the feature dimension (last dim)
    # Shape: [batch_size, seq_len, 1]
    inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
    
    # Compute the standard deviation multiplier using inverse CDF
    # This maps the target sparsity to a z-score in the normal distribution
    target_sparsity_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
    std_multiplier = _ndtri(target_sparsity_tensor)
    
    # Compute the adaptive cutoff threshold
    # Values below this threshold will be zeroed out
    # Shape: [batch_size, seq_len, 1]
    cutoff_threshold = inputs_mean + inputs_std * std_multiplier
    
    # Apply ReLU with the adaptive threshold
    # This is equivalent to: max(0, inputs - cutoff_threshold)
    sparse_output = F.relu(inputs_f32 - cutoff_threshold)
    
    return sparse_output.to(torch.bfloat16)

```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
