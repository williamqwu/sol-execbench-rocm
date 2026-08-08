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

## The problem: `046_attention_softmax_with_softcapping_and_dropout`

Attention softmax with logit softcapping (tanh-based clamping to ±30.0) applied before softmax, as used in Gemma-2 models. This operation applies: tanh(logits / 30.0) * 30.0 to stabilize attention scores, then softmax normalization.

**Axes** (workload dimensions):

- `batch_size` (varies per workload) — Batch size
- `num_heads` = 8 (constant) — Number of attention heads
- `seq_len_q` (varies per workload) — Query sequence length
- `seq_len_k` (varies per workload) — Key sequence length
- `attn_logit_softcapping` = 30 (constant) — Softcapping value for attention logits

**Inputs**

- `attn_weights`: [batch_size, num_heads, seq_len_q, seq_len_k], `bfloat16` — Attention logits before softcapping and softmax

**Outputs**

- `output`: [batch_size, num_heads, seq_len_q, seq_len_k], `bfloat16` — Normalized attention weights after softcapping and softmax

**Workload shapes you will be evaluated on** (16 of them):

- batch_size=1, seq_len_q=691, seq_len_k=691
- batch_size=4, seq_len_q=512, seq_len_k=512
- batch_size=2, seq_len_q=2048, seq_len_k=2048
- batch_size=4, seq_len_q=1, seq_len_k=853
- batch_size=64, seq_len_q=128, seq_len_k=128
- batch_size=4, seq_len_q=256, seq_len_k=256
- batch_size=16, seq_len_q=256, seq_len_k=256
- batch_size=8, seq_len_q=256, seq_len_k=256
- batch_size=1, seq_len_q=293, seq_len_k=293
- batch_size=8, seq_len_q=128, seq_len_k=128
- batch_size=1, seq_len_q=2048, seq_len_k=2048
- batch_size=32, seq_len_q=128, seq_len_k=128
- batch_size=1, seq_len_q=512, seq_len_k=512
- batch_size=2, seq_len_q=1024, seq_len_k=1024
- batch_size=4, seq_len_q=1024, seq_len_k=1024
- batch_size=8, seq_len_q=512, seq_len_k=512

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Apply Gemma3's softcapping transformation followed by softmax.
    
    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30]
    
    Args:
        attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        
    Returns:
        Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    SOFTCAP = 30.0
    
    # Apply softcapping transformation
    # Step 1: Divide by softcap
    scaled = attn_weights / SOFTCAP
    
    # Step 2: Apply tanh to clamp to [-1, 1]
    clamped = torch.tanh(scaled)
    
    # Step 3: Multiply by softcap to restore scale (now in [-30, 30])
    softcapped = clamped * SOFTCAP
    
    # Apply softmax normalization along the key dimension
    # Upcast to float32 for numerical stability, then cast back
    output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
    
    return output

```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
