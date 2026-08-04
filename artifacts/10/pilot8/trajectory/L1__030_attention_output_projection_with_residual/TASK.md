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

## The problem: `030_attention_output_projection_with_residual`

Fused attention output projection (o_proj) with residual addition. After computing attention outputs, this operation performs the final linear projection and adds the residual connection. The projection is a matmul (hidden_size x hidden_size) fused with elementwise residual add to eliminate intermediate memory traffic.

**Axes** (workload dimensions):

- `batch_size` (varies per workload) — Batch size
- `seq_len` (varies per workload) — Sequence length
- `hidden_size` = 2560 (constant) — Hidden dimension size

**Inputs**

- `attn_output`: [batch_size, seq_len, hidden_size], `bfloat16` — Attention output tensor after reshape from multi-head format
- `residual`: [batch_size, seq_len, hidden_size], `bfloat16` — Residual tensor from before attention layer
- `o_proj_weight`: [hidden_size, hidden_size], `bfloat16` — Output projection weight matrix (hidden_size x hidden_size)

**Outputs**

- `output`: [batch_size, seq_len, hidden_size], `bfloat16` — Output with projection and residual added

**Workload shapes you will be evaluated on** (16 of them):

- batch_size=16, seq_len=512
- batch_size=4, seq_len=128
- batch_size=8, seq_len=1024
- batch_size=1, seq_len=1571
- batch_size=4, seq_len=1024
- batch_size=2, seq_len=2053
- batch_size=8, seq_len=997
- batch_size=16, seq_len=256
- batch_size=64, seq_len=128
- batch_size=32, seq_len=256
- batch_size=8, seq_len=512
- batch_size=1, seq_len=1024
- batch_size=16, seq_len=128
- batch_size=2, seq_len=293
- batch_size=1, seq_len=2048
- batch_size=1, seq_len=256

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import torch


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fused attention output projection with residual addition.
    
    This performs:
    1. Linear projection: attn_output @ o_proj_weight.T
    2. Residual addition: projected + residual
    
    In a custom CUDA kernel, these operations would be fused to:
    - Compute matmul tiles in registers
    - Add residual directly to register-held results
    - Write final result to global memory (eliminating intermediate write)
    
    Args:
        attn_output: Attention output of shape (batch, seq_len, hidden_size)
        residual: Original input before attention of shape (batch, seq_len, hidden_size)
        o_proj_weight: Output projection weight of shape (hidden_size, hidden_size)
    
    Returns:
        Output with residual added, shape (batch, seq_len, hidden_size)
    """
    # Linear projection: (batch, seq_len, hidden_size) @ (hidden_size, hidden_size).T
    # -> (batch, seq_len, hidden_size)
    projected = torch.matmul(attn_output, o_proj_weight.t())
    
    # Residual addition
    output = projected + residual
    
    return output

```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
