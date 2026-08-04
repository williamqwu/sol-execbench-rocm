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

## The problem: `050_vae_decoder_mid_block_attention_resnet`

VAE decoder mid-block combining ResNet blocks with self-attention at the bottleneck. Processes compressed latent representation through: ResNet -> Attention -> ResNet. The attention operates on spatial features with GroupNorm, QKV projections, multi-head self-attention, and residual connections. Uses 512 channels at the bottleneck with 4 groups for GroupNorm.

**Axes** (workload dimensions):

- `batch_size` (varies per workload) — Batch size
- `height` (varies per workload) — Spatial height of feature map
- `width` (varies per workload) — Spatial width of feature map
- `in_channels` = 512 (constant) — Number of input/output channels at bottleneck
- `temb_channels` = 512 (constant) — Time embedding channels
- `num_groups` = 32 (constant) — Number of groups for GroupNorm
- `attention_head_dim` = 1 (constant) — Dimension per attention head
- `num_heads` = `None` (derived) — Number of attention heads

**Inputs**

- `hidden_states`: [batch_size, in_channels, height, width], `float32` — Input feature map [B, C, H, W]
- `temb`: [batch_size, temb_channels], `float32` — Time embedding [B, temb_channels]
- `resnet1_norm1_weight`: [in_channels], `float32` — ResNet1 first GroupNorm weight
- `resnet1_norm1_bias`: [in_channels], `float32` — ResNet1 first GroupNorm bias
- `resnet1_conv1_weight`: [in_channels, in_channels, 3, 3], `float32` — ResNet1 first conv weight
- `resnet1_conv1_bias`: [in_channels], `float32` — ResNet1 first conv bias
- `resnet1_time_emb_proj_weight`: [in_channels, temb_channels], `float32` — ResNet1 time embedding projection weight
- `resnet1_time_emb_proj_bias`: [in_channels], `float32` — ResNet1 time embedding projection bias
- `resnet1_norm2_weight`: [in_channels], `float32` — ResNet1 second GroupNorm weight
- `resnet1_norm2_bias`: [in_channels], `float32` — ResNet1 second GroupNorm bias
- `resnet1_conv2_weight`: [in_channels, in_channels, 3, 3], `float32` — ResNet1 second conv weight
- `resnet1_conv2_bias`: [in_channels], `float32` — ResNet1 second conv bias
- `attn_group_norm_weight`: [in_channels], `float32` — Attention GroupNorm weight
- `attn_group_norm_bias`: [in_channels], `float32` — Attention GroupNorm bias
- `attn_to_q_weight`: [in_channels, in_channels], `float32` — Attention query projection weight
- `attn_to_q_bias`: [in_channels], `float32` — Attention query projection bias
- `attn_to_k_weight`: [in_channels, in_channels], `float32` — Attention key projection weight
- `attn_to_k_bias`: [in_channels], `float32` — Attention key projection bias
- `attn_to_v_weight`: [in_channels, in_channels], `float32` — Attention value projection weight
- `attn_to_v_bias`: [in_channels], `float32` — Attention value projection bias
- `attn_to_out_weight`: [in_channels, in_channels], `float32` — Attention output projection weight
- `attn_to_out_bias`: [in_channels], `float32` — Attention output projection bias
- `resnet2_norm1_weight`: [in_channels], `float32` — ResNet2 first GroupNorm weight
- `resnet2_norm1_bias`: [in_channels], `float32` — ResNet2 first GroupNorm bias
- `resnet2_conv1_weight`: [in_channels, in_channels, 3, 3], `float32` — ResNet2 first conv weight
- `resnet2_conv1_bias`: [in_channels], `float32` — ResNet2 first conv bias
- `resnet2_time_emb_proj_weight`: [in_channels, temb_channels], `float32` — ResNet2 time embedding projection weight
- `resnet2_time_emb_proj_bias`: [in_channels], `float32` — ResNet2 time embedding projection bias
- `resnet2_norm2_weight`: [in_channels], `float32` — ResNet2 second GroupNorm weight
- `resnet2_norm2_bias`: [in_channels], `float32` — ResNet2 second GroupNorm bias
- `resnet2_conv2_weight`: [in_channels, in_channels, 3, 3], `float32` — ResNet2 second conv weight
- `resnet2_conv2_bias`: [in_channels], `float32` — ResNet2 second conv bias
- `eps`: scalar, `float32` — Epsilon for GroupNorm

**Outputs**

- `output`: [batch_size, in_channels, height, width], `float32` — Output feature map [B, C, H, W]

**Workload shapes you will be evaluated on** (11 of them):

- batch_size=1, height=32, width=32
- batch_size=1, height=61, width=61
- batch_size=2, height=64, width=64
- batch_size=1, height=48, width=48
- batch_size=16, height=32, width=32
- batch_size=4, height=16, width=16
- batch_size=8, height=32, width=32
- batch_size=32, height=32, width=32
- batch_size=4, height=48, width=48
- batch_size=2, height=41, width=41
- batch_size=1, height=16, width=16

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    resnet1_norm1_weight: torch.Tensor,
    resnet1_norm1_bias: torch.Tensor,
    resnet1_conv1_weight: torch.Tensor,
    resnet1_conv1_bias: torch.Tensor,
    resnet1_time_emb_proj_weight: torch.Tensor,
    resnet1_time_emb_proj_bias: torch.Tensor,
    resnet1_norm2_weight: torch.Tensor,
    resnet1_norm2_bias: torch.Tensor,
    resnet1_conv2_weight: torch.Tensor,
    resnet1_conv2_bias: torch.Tensor,
    attn_group_norm_weight: torch.Tensor,
    attn_group_norm_bias: torch.Tensor,
    attn_to_q_weight: torch.Tensor,
    attn_to_q_bias: torch.Tensor,
    attn_to_k_weight: torch.Tensor,
    attn_to_k_bias: torch.Tensor,
    attn_to_v_weight: torch.Tensor,
    attn_to_v_bias: torch.Tensor,
    attn_to_out_weight: torch.Tensor,
    attn_to_out_bias: torch.Tensor,
    resnet2_norm1_weight: torch.Tensor,
    resnet2_norm1_bias: torch.Tensor,
    resnet2_conv1_weight: torch.Tensor,
    resnet2_conv1_bias: torch.Tensor,
    resnet2_time_emb_proj_weight: torch.Tensor,
    resnet2_time_emb_proj_bias: torch.Tensor,
    resnet2_norm2_weight: torch.Tensor,
    resnet2_norm2_bias: torch.Tensor,
    resnet2_conv2_weight: torch.Tensor,
    resnet2_conv2_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1  # Single-head attention at VAE bottleneck
    head_dim = channels  # head_dim equals channels when num_heads=1
    scale = head_dim ** -0.5
    
    # ============ ResNet Block 1 ============
    residual1 = hidden_states
    
    # GroupNorm1 + SiLU + Conv1
    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv1_weight, resnet1_conv1_bias, padding=1)
    
    # Add time embedding
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    
    # GroupNorm2 + SiLU + Conv2
    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet1_conv2_weight, resnet1_conv2_bias, padding=1)
    
    # Residual connection
    hidden_states = h + residual1
    
    # ============ Attention Block ============
    attn_residual = hidden_states
    
    # GroupNorm
    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)
    
    # Reshape to [B, H*W, C]
    h = h.view(batch, channels, height * width).transpose(1, 2)
    
    # QKV projections
    query = F.linear(h, attn_to_q_weight, attn_to_q_bias)
    key = F.linear(h, attn_to_k_weight, attn_to_k_bias)
    value = F.linear(h, attn_to_v_weight, attn_to_v_bias)
    
    # Reshape for multi-head attention [B, num_heads, H*W, head_dim]
    seq_len = height * width
    query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention scores [B, num_heads, H*W, H*W]
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)
    
    # Apply attention to values [B, num_heads, H*W, head_dim]
    h = torch.matmul(attention_probs, value)
    
    # Reshape back [B, H*W, C]
    h = h.transpose(1, 2).reshape(batch, seq_len, channels)
    
    # Output projection
    h = F.linear(h, attn_to_out_weight, attn_to_out_bias)
    
    # Reshape to [B, C, H, W]
    h = h.transpose(1, 2).view(batch, channels, height, width)
    
    # Residual connection
    hidden_states = h + attn_residual
    
    # ============ ResNet Block 2 ============
    residual2 = hidden_states
    
    # GroupNorm1 + SiLU + Conv1
    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv1_weight, resnet2_conv1_bias, padding=1)
    
    # Add time embedding
    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]
    
    # GroupNorm2 + SiLU + Conv2
    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resnet2_conv2_weight, resnet2_conv2_bias, padding=1)
    
    # Residual connection
    output = h + residual2
    
    return output

```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
