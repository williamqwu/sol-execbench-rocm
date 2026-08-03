# T_b candidate variants (task 06)

Pre-authored optimized-PyTorch formulations, one directory per problem. These
are **platform-independent PyTorch**, so they were written without hardware —
the node's job is measurement and selection, not authoring. That converts the
most human-in-the-loop task in the project into a batch job.

Layout: `<problem>/v<N>_<description>.py`, each exposing `run(*inputs)`.

## The policy, which must match upstream's

- eager vs `torch.compile`, whichever measures faster
- contiguity and layout hygiene
- **no handwritten kernels** — T_b is the PyTorch anchor, not a tuned kernel.
  Breaking this makes S=0.5 mean something different on AMD than on NVIDIA and
  quietly breaks cross-platform interpretation.

## Status

**Not yet populated.** Generating these is the highest-value remaining CPU-only
task: every variant written here is measurement time saved on the node, and
none of it needs a GPU. Suggested minimum per problem: eager baseline,
`torch.compile(mode="max-autotune")`, and one manual-fusion variant where the
problem has an obvious fusion.

Record any variant you add on the node, and why the pre-authored set was
missing it.
