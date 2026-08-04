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

## The problem: `069_joint_transformer_block_residual_path`

Complete residual path through a JointTransformerBlock including dual-stream attention, adaptive normalization, and feedforward with residual connections for SD3.5.

**Axes** (workload dimensions):

- `batch_size` (varies per workload) — 
- `seq_len` (varies per workload) — 
- `context_len` (varies per workload) — 
- `num_attention_heads` = 24 (constant) — 
- `attention_head_dim` = 64 (constant) — 
- `inner_dim` = 1536 (constant) — 
- `context_dim` = 1152 (constant) — 
- `ff_inner_dim` = 6144 (constant) — 
- `mod_dim_6x` = `None` (derived) — 
- `context_mod_dim_6x` = `None` (derived) — 

**Inputs**

- `hidden_states`: [batch_size, seq_len, inner_dim], `float32` — Image latent tokens
- `encoder_hidden_states`: [batch_size, context_len, context_dim], `float32` — Text/context tokens
- `temb`: [batch_size, inner_dim], `float32` — Timestep embeddings
- `norm1_weight`: [mod_dim_6x, inner_dim], `float32` — 
- `norm1_bias`: [mod_dim_6x], `float32` — 
- `norm1_context_weight`: [context_mod_dim_6x, inner_dim], `float32` — 
- `norm1_context_bias`: [context_mod_dim_6x], `float32` — 
- `to_q_weight`: [inner_dim, inner_dim], `float32` — 
- `to_q_bias`: [inner_dim], `float32` — 
- `to_k_weight`: [inner_dim, inner_dim], `float32` — 
- `to_k_bias`: [inner_dim], `float32` — 
- `to_v_weight`: [inner_dim, inner_dim], `float32` — 
- `to_v_bias`: [inner_dim], `float32` — 
- `add_q_proj_weight`: [inner_dim, context_dim], `float32` — 
- `add_q_proj_bias`: [inner_dim], `float32` — 
- `add_k_proj_weight`: [inner_dim, context_dim], `float32` — 
- `add_k_proj_bias`: [inner_dim], `float32` — 
- `add_v_proj_weight`: [inner_dim, context_dim], `float32` — 
- `add_v_proj_bias`: [inner_dim], `float32` — 
- `to_out_weight`: [inner_dim, inner_dim], `float32` — 
- `to_out_bias`: [inner_dim], `float32` — 
- `to_add_out_weight`: [context_dim, inner_dim], `float32` — 
- `to_add_out_bias`: [context_dim], `float32` — 
- `ff_linear1_weight`: [ff_inner_dim, inner_dim], `float32` — 
- `ff_linear1_bias`: [ff_inner_dim], `float32` — 
- `ff_linear2_weight`: [inner_dim, ff_inner_dim], `float32` — 
- `ff_linear2_bias`: [inner_dim], `float32` — 
- `ff_context_linear1_weight`: [ff_inner_dim, context_dim], `float32` — 
- `ff_context_linear1_bias`: [ff_inner_dim], `float32` — 
- `ff_context_linear2_weight`: [context_dim, ff_inner_dim], `float32` — 
- `ff_context_linear2_bias`: [context_dim], `float32` — 

**Outputs**

- `output_encoder_hidden_states`: [batch_size, context_len, context_dim], `float32` — 
- `output_hidden_states`: [batch_size, seq_len, inner_dim], `float32` — 

**Workload shapes you will be evaluated on** (16 of them):

- batch_size=1, seq_len=1024, context_len=77
- batch_size=1, seq_len=2048, context_len=77
- batch_size=32, seq_len=256, context_len=77
- batch_size=64, seq_len=128, context_len=77
- batch_size=1, seq_len=2053, context_len=77
- batch_size=1, seq_len=8192, context_len=77
- batch_size=2, seq_len=256, context_len=154
- batch_size=2, seq_len=2048, context_len=77
- batch_size=1, seq_len=293, context_len=77
- batch_size=8, seq_len=256, context_len=77
- batch_size=4, seq_len=512, context_len=77
- batch_size=16, seq_len=512, context_len=77
- batch_size=1, seq_len=128, context_len=154
- batch_size=4, seq_len=1024, context_len=77
- batch_size=2, seq_len=4096, context_len=77
- batch_size=4, seq_len=2048, context_len=77

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import math

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    context_len = axes_and_scalars["context_len"]

    dim = 1536
    context_dim = 1152
    ff_inner_dim = 6144

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def xavier(out_f, in_f):
        return torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)

    return {
        # Activation tensors
        "hidden_states": torch.randn(batch_size, seq_len, dim, device=device, generator=g),
        "encoder_hidden_states": torch.randn(batch_size, context_len, context_dim, device=device, generator=g),
        # Time embedding — small magnitude
        "temb": torch.randn(batch_size, dim, device=device, generator=g) * 0.1,
        # Modulation projection weights (NOT norm weights — they project temb to 6*dim)
        "norm1_weight": xavier(6 * dim, dim),
        "norm1_bias": torch.randn(6 * dim, device=device, generator=g),
        "norm1_context_weight": xavier(6 * context_dim, dim),
        "norm1_context_bias": torch.randn(6 * context_dim, device=device, generator=g),
        # Image stream QKV
        "to_q_weight": xavier(dim, dim),
        "to_q_bias": torch.randn(dim, device=device, generator=g),
        "to_k_weight": xavier(dim, dim),
        "to_k_bias": torch.randn(dim, device=device, generator=g),
        "to_v_weight": xavier(dim, dim),
        "to_v_bias": torch.randn(dim, device=device, generator=g),
        # Context stream QKV
        "add_q_proj_weight": xavier(dim, context_dim),
        "add_q_proj_bias": torch.randn(dim, device=device, generator=g),
        "add_k_proj_weight": xavier(dim, context_dim),
        "add_k_proj_bias": torch.randn(dim, device=device, generator=g),
        "add_v_proj_weight": xavier(dim, context_dim),
        "add_v_proj_bias": torch.randn(dim, device=device, generator=g),
        # Output projections
        "to_out_weight": xavier(dim, dim),
        "to_out_bias": torch.randn(dim, device=device, generator=g),
        "to_add_out_weight": xavier(context_dim, dim),
        "to_add_out_bias": torch.randn(context_dim, device=device, generator=g),
        # FF — image stream
        "ff_linear1_weight": xavier(ff_inner_dim, dim),
        "ff_linear1_bias": torch.randn(ff_inner_dim, device=device, generator=g),
        "ff_linear2_weight": xavier(dim, ff_inner_dim),
        "ff_linear2_bias": torch.randn(dim, device=device, generator=g),
        # FF — context stream
        "ff_context_linear1_weight": xavier(ff_inner_dim, context_dim),
        "ff_context_linear1_bias": torch.randn(ff_inner_dim, device=device, generator=g),
        "ff_context_linear2_weight": xavier(context_dim, ff_inner_dim),
        "ff_context_linear2_bias": torch.randn(context_dim, device=device, generator=g),
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    norm1_context_weight: torch.Tensor,
    norm1_context_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_q_bias: torch.Tensor,
    to_k_weight: torch.Tensor,
    to_k_bias: torch.Tensor,
    to_v_weight: torch.Tensor,
    to_v_bias: torch.Tensor,
    add_q_proj_weight: torch.Tensor,
    add_q_proj_bias: torch.Tensor,
    add_k_proj_weight: torch.Tensor,
    add_k_proj_bias: torch.Tensor,
    add_v_proj_weight: torch.Tensor,
    add_v_proj_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    to_add_out_weight: torch.Tensor,
    to_add_out_bias: torch.Tensor,
    ff_linear1_weight: torch.Tensor,
    ff_linear1_bias: torch.Tensor,
    ff_linear2_weight: torch.Tensor,
    ff_linear2_bias: torch.Tensor,
    ff_context_linear1_weight: torch.Tensor,
    ff_context_linear1_bias: torch.Tensor,
    ff_context_linear2_weight: torch.Tensor,
    ff_context_linear2_bias: torch.Tensor,
):
    batch_size = hidden_states.shape[0]
    image_seq_len = hidden_states.shape[1]
    context_seq_len = encoder_hidden_states.shape[1]
    
    dim = 1536
    context_dim = 1152
    num_heads = 24
    head_dim = 64
    scale = head_dim ** -0.5
    
    # AdaLayerNormZero modulation
    norm_hidden_states = F.layer_norm(hidden_states, (dim,), eps=1e-6)
    norm_encoder_hidden_states = F.layer_norm(encoder_hidden_states, (context_dim,), eps=1e-6)
    
    # Get modulation parameters from timestep embedding
    temb_silu = F.silu(temb)
    mod_params = F.linear(temb_silu, norm1_weight, norm1_bias)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.chunk(6, dim=-1)
    
    mod_params_context = F.linear(temb_silu, norm1_context_weight, norm1_context_bias)
    c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = mod_params_context.chunk(6, dim=-1)
    
    # Apply modulation to normalized states
    norm_hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)
    
    # Dual-stream attention
    # Image stream QKV
    q = F.linear(norm_hidden_states, to_q_weight, to_q_bias)
    k = F.linear(norm_hidden_states, to_k_weight, to_k_bias)
    v = F.linear(norm_hidden_states, to_v_weight, to_v_bias)
    
    # Text stream QKV
    c_q = F.linear(norm_encoder_hidden_states, add_q_proj_weight, add_q_proj_bias)
    c_k = F.linear(norm_encoder_hidden_states, add_k_proj_weight, add_k_proj_bias)
    c_v = F.linear(norm_encoder_hidden_states, add_v_proj_weight, add_v_proj_bias)
    
    # Concatenate for joint attention
    q = torch.cat([q, c_q], dim=1)
    k = torch.cat([k, c_k], dim=1)
    v = torch.cat([v, c_v], dim=1)
    
    # Reshape for multi-head attention
    total_seq_len = q.shape[1]
    q = q.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, total_seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, total_seq_len, -1)
    
    # Split back into image and text streams
    attn_output_img, c_attn_output = attn_output.split([image_seq_len, context_seq_len], dim=1)
    
    # Output projections
    attn_output_img = F.linear(attn_output_img, to_out_weight, to_out_bias)
    c_attn_output = F.linear(c_attn_output, to_add_out_weight, to_add_out_bias)
    
    # Gated residual connection for attention
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output_img
    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * c_attn_output
    
    # Feedforward path
    # Normalize
    norm_hidden_states = F.layer_norm(hidden_states, (dim,), eps=1e-6)
    norm_encoder_hidden_states = F.layer_norm(encoder_hidden_states, (context_dim,), eps=1e-6)
    
    # Apply modulation
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)
    
    # Feedforward with GELU-approximate
    ff_hidden = F.linear(norm_hidden_states, ff_linear1_weight, ff_linear1_bias)
    ff_hidden = F.gelu(ff_hidden, approximate='tanh')
    ff_output = F.linear(ff_hidden, ff_linear2_weight, ff_linear2_bias)
    
    c_ff_hidden = F.linear(norm_encoder_hidden_states, ff_context_linear1_weight, ff_context_linear1_bias)
    c_ff_hidden = F.gelu(c_ff_hidden, approximate='tanh')
    c_ff_output = F.linear(c_ff_hidden, ff_context_linear2_weight, ff_context_linear2_bias)
    
    # Gated residual connection for feedforward
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * c_ff_output
    
    return encoder_hidden_states, hidden_states

```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
