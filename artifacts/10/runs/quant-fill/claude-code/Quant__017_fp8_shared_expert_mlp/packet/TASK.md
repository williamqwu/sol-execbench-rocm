# Write the fastest correct kernel for `Quant/017_fp8_shared_expert_mlp`

You are optimizing one problem from SOL-ExecBench on an **AMD Instinct MI355X**
(CDNA4, `gfx950`, 256 CUs, 288 GiB HBM3E). You have one GPU to yourself; it is
already the only one visible to you, so address it as `cuda:0`.

## The problem

`reference.py` defines the semantics. Your kernel must produce the same outputs
for every workload in `workload.jsonl`, within the tolerance each workload
states. `definition.json` describes the input shapes and dtypes, where the axes
are symbolic (`batch_size`, `seq_len`, ...) and each workload binds them to
concrete values.

Read all three files before writing anything. There are **16
workloads** and they cover a range of shapes; a kernel that only handles the
first one is not a solution.

## What you produce

A single file `solution.json` in this directory. Nothing else is collected.

```json
{
  "name": "017_fp8_shared_expert_mlp_agent",
  "definition": "017_fp8_shared_expert_mlp",
  "author": "agent",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["MI355X", "LOCAL"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch", "triton"],
    "destination_passing_style": false
  },
  "sources": [
    {"path": "kernel.py", "content": ""}
  ]
}
```

Write your kernel to `kernel.py` as a real file and leave `content` as `""` --
the loader fills it in from `path`. You may also inline the content instead if
you prefer; both work. Multiple source files are fine.

**Write `solution.json` before you start iterating, not at the end.** Only that
file is collected, and a session can end without warning -- a wallclock cap, or
the API gateway dropping the connection. If that happens with no
`solution.json` on disk, a working kernel scores exactly the same as no kernel
at all. Write it as soon as you have a first attempt, then keep improving
`kernel.py` in place; the loader re-reads the file each time you verify, so you
do not need to touch `solution.json` again.

The entry point function must accept the inputs in the order
`definition.json` lists them and **return** the outputs (that is what
`destination_passing_style: false` means). Match `reference.py`'s signature.

## Languages

Any of these. You are not restricted to one, and you are not expected to use
the most exotic one that fits -- use whatever actually gets the speed.

| `spec.languages` value | What it is | Notes |
|---|---|---|
| `pytorch` | plain torch ops, `torch.compile` allowed | the baseline; correct but rarely fast |
| `triton` | Triton JIT kernels | the usual first choice on ROCm |
| `triton` + Gluon | `triton.experimental.gluon` | lower-level tile/layout control, same `triton` language value |
| `hip_cpp` | HIP C++, compiled `--offload-arch=gfx950` | inline GCN assembly via `asm volatile` lives here |
| `ck` / `ck_tile` | AMD Composable Kernel | headers under `/opt/rocm/include` |
| `hipblaslt` | AMD hipBLASLt | GEMM with epilogue fusion |
| `miopen` | AMD MIOpen | convolutions |
| `aiter` | AMD AI Tensor Engine for ROCm | prebuilt fused kernels |


C++ and Python languages cannot be mixed in one solution. For `hip_cpp`, source
files use `.hip` or `.cpp`; `--offload-arch=gfx950` is injected for you.

## Verifying

```bash
./verify
```

Compiles and evaluates your current `solution.json` against every workload on
real hardware and prints, per workload: pass/fail, the measured error against
the allowed tolerance, and your latency beside the reference's.

**You may run it 5 times.** Attempt counts are tracked for you and each run tells you how many remain. Use them:
a kernel that has never been executed is usually wrong. But do read the
reference carefully first rather than spending attempts on guesses -- the most
common failure here is not a slow kernel, it is a kernel that gets the
numerics subtly wrong.

`./verify` is your feedback, not your score. Your solution is re-evaluated
afterwards from a clean tree on a different GPU, and that run is what counts.

## How you are scored

1. **Correctness first.** A wrong kernel scores nothing regardless of speed.
   All 16 workloads must pass.
2. **Then speed**, as proximity to an analytically derived Speed-of-Light bound
   for this hardware -- not as speedup over PyTorch. Being 2x faster than the
   reference is worth little if the hardware bound is 10x away; getting close
   to the bound is worth a lot.

So: match the numerics exactly, then remove memory traffic and launch overhead.
Fusing the reference's separate operations into one pass over the data is
usually where the win is.

## Numerics, specifically

The reference's *order of operations* is part of the specification. Reproduce
its intermediate rounding, not just its algebra: if it rounds a `float32`
accumulator to `bfloat16` before a multiply, so must you, or you will be more
accurate than the reference and still fail the tolerance. Accumulate in
`float32` where the reference does.

## Not allowed

The harness detects all of these and a solution caught by any of them scores
zero:

- Monkey-patching torch, the harness, or the timing path.
- Caching outputs across calls, or returning a lazily-evaluated handle so the
  work lands outside the timed region.
- Touching GPU clocks, `rocm-smi`, or `amd-smi`.
- Editing anything outside this directory.
- Special-casing the specific input values or workload shapes to skip work that
  the semantics require.

Making the kernel genuinely faster is the whole exercise. Everything above is a
way of appearing faster instead, and is checked for.
