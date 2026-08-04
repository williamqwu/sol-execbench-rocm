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

## The problem: `015_audio_sinusoidal_position_embedding_with_conv_projection`

Audio encoder's sinusoidal position embedding generation followed by 3-stage Conv2d downsampling (8x total reduction), GELU activations, and linear projection to model dimension. Processes mel-spectrogram through conv layers, adds cached sinusoidal position embeddings, and returns padded output.

**Axes** (workload dimensions):

- `batch_size` (varies per workload) — Number of audio samples in batch
- `time_dim` (varies per workload) — Time dimension of mel-spectrogram input
- `time_after_conv` (varies per workload) — Time dimension after 3 stride-2 convolutions (approximately time_dim // 8)
- `d_model` = 1024 (constant) — Model hidden dimension
- `num_mel_bins` = 80 (constant) — Number of mel frequency bins
- `max_source_positions` = 1500 (constant) — Maximum sequence length for position embeddings
- `downsample_hidden_size` = 384 (constant) — Hidden size in conv layers
- `freq_dim_after_conv` = 10 (constant) — Frequency dimension after 3 stride-2 convs: 80->40->20->10
- `conv_out_dim` = 3840 (constant) — Flattened conv output: 384 * 10
- `one` = 1 (constant) — Constant 1 for input channel dimension
- `kernel_size` = 3 (constant) — Conv kernel size

**Inputs**

- `input_features`: [batch_size, one, num_mel_bins, time_dim], `bfloat16` — Mel-spectrogram input
- `conv2d1_weight`: [downsample_hidden_size, one, kernel_size, kernel_size], `bfloat16` — First conv layer weights
- `conv2d1_bias`: [downsample_hidden_size], `bfloat16` — First conv layer bias
- `conv2d2_weight`: [downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size], `bfloat16` — Second conv layer weights
- `conv2d2_bias`: [downsample_hidden_size], `bfloat16` — Second conv layer bias
- `conv2d3_weight`: [downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size], `bfloat16` — Third conv layer weights
- `conv2d3_bias`: [downsample_hidden_size], `bfloat16` — Third conv layer bias
- `conv_out_weight`: [d_model, conv_out_dim], `bfloat16` — Linear projection weights
- `positional_embedding`: [max_source_positions, d_model], `bfloat16` — Cached sinusoidal position embeddings
- `embed_scale`: scalar, `float32` — Embedding scale factor sqrt(d_model)

**Outputs**

- `hidden_states`: [batch_size, time_after_conv, d_model], `bfloat16` — Output embeddings with position encoding added

**Workload shapes you will be evaluated on** (16 of them):

- batch_size=2, time_dim=1688, time_after_conv=211
- batch_size=32, time_dim=4328, time_after_conv=541
- batch_size=1, time_dim=1048, time_after_conv=131
- batch_size=1, time_dim=1808, time_after_conv=226
- batch_size=32, time_dim=920, time_after_conv=115
- batch_size=64, time_dim=128, time_after_conv=16
- batch_size=16, time_dim=2048, time_after_conv=256
- batch_size=16, time_dim=384, time_after_conv=48
- batch_size=4, time_dim=2216, time_after_conv=277
- batch_size=2, time_dim=256, time_after_conv=32
- batch_size=2, time_dim=2528, time_after_conv=316
- batch_size=8, time_dim=1976, time_after_conv=247
- batch_size=8, time_dim=3256, time_after_conv=407
- batch_size=32, time_dim=768, time_after_conv=96
- batch_size=64, time_dim=512, time_after_conv=64
- batch_size=1, time_dim=3000, time_after_conv=375

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import math

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    time_dim = axes_and_scalars["time_dim"]
    d_model = 1024
    max_source_positions = 1500
    downsample_hidden_size = 384
    conv_out_dim = 3840  # 384 * 10
    kernel_size = 3
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    # Sinusoidal positional embedding
    pe = torch.zeros(max_source_positions, d_model, device=device)
    position = torch.arange(0, max_source_positions, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return {
        "input_features": torch.randn(batch_size, 1, 80, time_dim, device=device, generator=g).to(dtype),
        # Conv weights — Kaiming init
        "conv2d1_weight": kaiming_conv(downsample_hidden_size, 1, kernel_size, kernel_size),
        "conv2d1_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d2_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d2_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        "conv2d3_weight": kaiming_conv(downsample_hidden_size, downsample_hidden_size, kernel_size, kernel_size),
        "conv2d3_bias": torch.randn(downsample_hidden_size, device=device, generator=g).to(dtype),
        # Linear projection weight
        "conv_out_weight": xavier(d_model, conv_out_dim),
        # Sinusoidal positional embedding
        "positional_embedding": pe.to(dtype),
        # embed_scale = sqrt(d_model)
        "embed_scale": math.sqrt(d_model),
    }


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # Stage 1: Conv2d (1 -> 384 channels) + GELU
    x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Stage 2: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Stage 3: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
    x = F.gelu(x)
    
    # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
    b, c, f, t = x.size()
    x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    
    # Linear projection to d_model (no bias)
    x = F.linear(x, conv_out_weight)
    
    # Scale embeddings
    x = x * embed_scale
    
    # Add positional embeddings
    seq_len = x.shape[1]
    pos_embed = positional_embedding[:seq_len, :].unsqueeze(0)
    x = x + pos_embed
    
    return x

```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
