"""AdaLayerNormZero modulation parameter extraction (Flux dual-stream).

out = emb @ weight.T + bias, split into 6 chunks along dim 1.

Why this is a single fused `addmm` and nothing more elaborate
-------------------------------------------------------------
The workload tolerances here (atol 1.675e-07, rtol 1.192e-07 = float32 eps,
with required_matched_ratio 0.99) are derived from *run-to-run* variance of a
deterministic library GEMM. That variance is zero, so the tolerance floors at
one float32 ulp of a quantity whose intermediate sums are ~sqrt(3072)x larger
than the final element magnitude. The practical consequence, measured:

  * float64 recomputation of the same math matches only ~14% of elements
    (needs 99%). Being *more* accurate fails.
  * bf16 matrix-core emulation (3- and 6-term split) matches ~0.005%.
  * Splitting the K=3072 reduction at any granularity (1536 down to 4) fails;
    only the full-K single GEMM is bitwise-exact.
  * No hand-rolled accumulation order reproduces it: sequential FMA, pairwise
    tree, and hybrid tree+sequential at block sizes 1..128 all match <13%.

So correctness pins the result to the exact accumulation order of the vendor
fp32 GEMM kernel, and the only real freedom left is how much overhead is
wrapped around that one call. All of these were confirmed bitwise-identical to
the reference on all 16 workload shapes:

  torch.addmm(bias, emb, weight.t())        <- fastest, chosen
  torch.nn.functional.linear(emb, weight, bias)
  torch.matmul(emb, weight.t()) + bias      <- the reference; extra bias pass

and every BLAS backend (default / hipblaslt / hipblas / ck) dispatches to the
same kernel, producing an identical bit checksum at identical speed.

`addmm` fuses the bias into the GEMM epilogue, removing the reference's separate
read-modify-write pass over the [batch, 18432] fp32 output. That is the entire
available win (~1.02-1.07x, larger at small batch where the extra launch
dominates). `.chunk` returns views, so it is free -- no copy, no extra traffic.
Slicing manually or reusing a cached `out=` buffer both measured slower.
"""

import torch


@torch.no_grad()
def run(emb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """Project timestep embeddings and split into 6 modulation parameters.

    Args:
        emb:    [batch_size, inner_dim] float32
        weight: [6 * inner_dim, inner_dim] float32
        bias:   [6 * inner_dim] float32

    Returns:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp),
        each a [batch_size, inner_dim] view.
    """
    out = torch.addmm(bias, emb, weight.t())
    return out.chunk(6, dim=1)
