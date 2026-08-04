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

## The problem: `019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`

Batched Multi-head Latent Attention prefill with a paged KV cache. Causal mask is applied. Captured from DeepSeek-V3 during incremental prefill with tensor parallel size 8.

**Axes** (workload dimensions):

- `num_qo_heads` = 16 (constant) — Number of query heads after tensor parallel split (128/8=16).
- `head_dim_ckv` = 512 (constant) — 
- `head_dim_kpe` = 64 (constant) — 
- `page_size` = 1 (constant) — 
- `total_q` (varies per workload) — Total number of query tokens.
- `num_pages` (varies per workload) — Total allocated pages in KV cache.
- `len_indptr` (varies per workload) — Length of indptr arrays (batch_size + 1).
- `num_kv_indices` (varies per workload) — Total number of KV indices.

**Inputs**

- `q_nope`: [total_q, num_qo_heads, head_dim_ckv], `bfloat16` — Query tensor without positional encoding component.
- `q_pe`: [total_q, num_qo_heads, head_dim_kpe], `bfloat16` — Query positional encoding component.
- `ckv_cache`: [num_pages, page_size, head_dim_ckv], `bfloat16` — Compressed key-value cache.
- `kpe_cache`: [num_pages, page_size, head_dim_kpe], `bfloat16` — Key positional encoding cache.
- `qo_indptr`: [len_indptr], `int32` — Query offsets for each sequence.
- `kv_indptr`: [len_indptr], `int32` — KV page offsets for each sequence.
- `kv_indices`: [num_kv_indices], `int32` — Page indices for KV cache lookups.
- `sm_scale`: scalar, `float32` — Softmax scale. Default is (1/sqrt(128 + 64) = 1/sqrt(192)), based on head dimensions before matrix absorption.

**Outputs**

- `output`: [total_q, num_qo_heads, head_dim_ckv], `bfloat16` — 
- `lse`: [total_q, num_qo_heads], `float32` — The 2-based log-sum-exp of attention logits.

**Workload shapes you will be evaluated on** (38 of them):

- total_q=33, num_pages=989669, len_indptr=2, num_kv_indices=34
- total_q=1, num_pages=989669, len_indptr=2, num_kv_indices=34
- total_q=17, num_pages=989669, len_indptr=2, num_kv_indices=19
- total_q=52, num_pages=989669, len_indptr=5, num_kv_indices=67
- total_q=376, num_pages=989669, len_indptr=2, num_kv_indices=381
- total_q=287, num_pages=989669, len_indptr=2, num_kv_indices=288
- total_q=5, num_pages=989669, len_indptr=2, num_kv_indices=7
- total_q=1187, num_pages=989669, len_indptr=4, num_kv_indices=1205
- total_q=10, num_pages=989669, len_indptr=2, num_kv_indices=12
- total_q=3, num_pages=989669, len_indptr=2, num_kv_indices=5
- total_q=13, num_pages=989669, len_indptr=2, num_kv_indices=14
- total_q=26, num_pages=989669, len_indptr=3, num_kv_indices=32
- total_q=8987, num_pages=989669, len_indptr=57, num_kv_indices=14390
- total_q=29, num_pages=989669, len_indptr=2, num_kv_indices=34
- total_q=2, num_pages=989669, len_indptr=3, num_kv_indices=53
- total_q=1028, num_pages=989669, len_indptr=2, num_kv_indices=1038
- total_q=22, num_pages=989669, len_indptr=23, num_kv_indices=17759
- total_q=15, num_pages=989669, len_indptr=2, num_kv_indices=18
- total_q=69, num_pages=989669, len_indptr=4, num_kv_indices=90
- total_q=3024, num_pages=989669, len_indptr=4, num_kv_indices=3029
- ... and 18 more

## The reference implementation

It is in `reference.py`, and `kernel.py` currently holds an identical copy.
Read it first.

```python
import torch
import math


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or page_beg >= page_end:
            # No queries or KV for this batch element
            continue

        kv_len = page_end - page_beg
        pages = kv_indices[page_beg:page_end]

        # Since page_size=1, pages are token indices
        tok_idx = pages[:kv_len].to(torch.long)
        Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe]

        q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
        q_pe_batch = q_pe[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_kpe]

        q_len = q_end - q_start

        for i in range(q_len):
            qn = q_nope_batch[i]  # [num_heads, head_dim_ckv]
            qp = q_pe_batch[i]  # [num_heads, head_dim_kpe]

            logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_heads, kv_len]
            logits_scaled = logits * sm_scale

            # Apply causal mask
            prefix_len = kv_len - q_len  # Number of previously cached tokens
            query_abs_pos = prefix_len + i  # Absolute position of current query
            
            causal_mask = torch.arange(kv_len, device=logits_scaled.device) > query_abs_pos
            logits_scaled.masked_fill_(causal_mask.unsqueeze(0), -float("inf"))

            # Compute 2-base LSE
            lse[q_start + i] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [num_heads, L_tokens]
            out = attn @ Kc  # [num_heads, head_dim_ckv]
            output[q_start + i] = out.to(torch.bfloat16)

    return output, lse
```

## Finishing

Your session has a spend cap and may be cut off without warning. Treat
`kernel.py` as always-shippable: never leave it in a state that has not just
passed `./evaluate`. If an experiment does not work out, revert `kernel.py` to
the last version that passed before moving on.

Begin. Measure before you optimize, and measure after every change.
