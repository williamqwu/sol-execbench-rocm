# SPDX-License-Identifier: Apache-2.0
"""Build the working directory an agent sees for one problem.

The packet is deliberately minimal. It contains the problem and nothing else:

    definition.json    the problem, verbatim from the dataset
    reference.py       the PyTorch semantics the kernel must reproduce
    workload.jsonl     the workloads, with AMD-DERIVED tolerances substituted
    TASK.md            what to do, what is allowed, how it is scored
    verify             run the harness against the current solution.json
    solution.json      the agent writes this

What it does **not** contain matters as much. It has no copy of
``artifacts/05`` (the tolerance derivations), no ``artifacts/02`` reference
timings, no ``artifacts/03`` SOL bounds, and no other problem. An agent that
could read the tolerance derivation could tune to the tolerance instead of to
the semantics, and an agent that could read another problem's solution would be
scored on a different task than the one it was given.

The tolerances *are* in ``workload.jsonl``, because the harness needs them there
to judge a workload and hiding them would only mean the agent guesses at what
"correct" means. Knowing the tolerance is legitimate; deriving a kernel from the
tolerance derivation's seed statistics is not.

Containment is not the security boundary here and is not claimed to be -- the
agent runs as a local process with a writable filesystem. The boundary is that
**scoring never trusts anything inside the packet**: ``scripts/score_solutions.py``
re-evaluates the harvested ``solution.json`` from a pristine tree on the
authoritative GPU, and checks that tree was not modified. ``verify`` is a
feedback channel, not an oracle.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The agent may use any of these. The list is not a suggestion to use the
# exotic ones -- it is there so that "I could not express this" is never a
# reason a problem is unsolved.
LANGUAGE_MENU = """\
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
| `flydsl` | FlyDSL kernel DSL (`flydsl==0.2.4` is installed) | Python-hosted: `@flyc.kernel` device body plus a `@flyc.jit` launcher |
| `assembly` | hand-written `gfx950` ISA | not a host language -- pair it with the language of the file the asm lives in |
"""

TASK_TEMPLATE = """\
# Write the fastest correct kernel for `{problem_key}`

You are optimizing one problem from SOL-ExecBench on an **AMD Instinct MI355X**
(CDNA4, `gfx950`, 256 CUs, 288 GiB HBM3E). You have one GPU to yourself; it is
already the only one visible to you, so address it as `cuda:0`.

## The problem

`reference.py` defines the semantics. Your kernel must produce the same outputs
for every workload in `workload.jsonl`, within the tolerance each workload
states. `definition.json` describes the input shapes and dtypes, where the axes
are symbolic (`batch_size`, `seq_len`, ...) and each workload binds them to
concrete values.

Read all three files before writing anything. There are **{n_workloads}
workloads** and they cover a range of shapes; a kernel that only handles the
first one is not a solution.

## What you produce

A single file `solution.json` in this directory. Nothing else is collected.

```json
{{
  "name": "{suggested_name}",
  "definition": "{definition_name}",
  "author": "{author}",
  "spec": {{
    "languages": ["triton"],
    "target_hardware": ["MI355X", "LOCAL"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch", "triton"],
    "destination_passing_style": false
  }},
  "sources": [
    {{"path": "kernel.py", "content": ""}}
  ]
}}
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

{language_menu}

C++ and Python languages cannot be mixed in one solution. For `hip_cpp`, source
files use `.hip` or `.cpp`; `--offload-arch=gfx950` is injected for you.

`assembly` is the one exception to that rule, because it says how the kernel
body was written rather than which language hosts it: declare it *alongside*
the language of the file the asm actually lives in --
`["assembly", "hip_cpp"]` for inline `asm volatile` in a compiled source, or
`["assembly", "pytorch"]` for an ISA blob you assemble and load at run time
from `kernel.py`. Declared on its own it is held to a `.py` entry point, since
there is no build path here that assembles a standalone `.s`/`.S` file.

## Verifying

```bash
./verify
```

Compiles and evaluates your current `solution.json` against every workload on
real hardware and prints, per workload: pass/fail, the measured error against
the allowed tolerance, and your latency beside the reference's.

**You may run it {max_attempts} times.** Attempt {attempt_hint}. Use them:
a kernel that has never been executed is usually wrong. But do read the
reference carefully first rather than spending attempts on guesses -- the most
common failure here is not a slow kernel, it is a kernel that gets the
numerics subtly wrong.

`./verify` is your feedback, not your score. Your solution is re-evaluated
afterwards from a clean tree on a different GPU, and that run is what counts.

## How you are scored

1. **Correctness first.** A wrong kernel scores nothing regardless of speed.
   All {n_workloads} workloads must pass.
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
"""

VERIFY_SCRIPT = """\
#!/usr/bin/env bash
# Evaluate the current solution.json on real hardware. Written by the harness;
# do not edit -- it is regenerated per attempt and the score does not come from
# here anyway.
set -uo pipefail
exec {python} {verify_py} --packet {packet_dir} "$@"
"""

# What `agent_verify.py` needs in order to compile and evaluate a solution.
# Everything else in the repo is deliberately absent from the view the agent's
# verify runs against -- see build_verify_root.
VERIFY_ROOT_CONTENTS = (
    "scripts/agent_verify.py",
    "scripts/provenance.py",
    "scripts/runners/_common.py",
    "src/sol_execbench",
    "src/solexbench_rocm",
)


def build_verify_root(repo_root: Path, dest: Path) -> Path:
    """A minimal harness tree for agents to verify against.

    An observed smoke run had the agent leave its packet and read
    ``src/sol_execbench/core/bench/timing.py`` to see how it would be measured.
    Reading the harness is arguably fair -- it is Apache-2.0 code the agent could
    have read anywhere -- but the same reach exposes three things that are not:

    - ``artifacts/05/`` the tolerance *derivations*, with per-seed error
      statistics. An agent that reads those can target the tolerance instead of
      the semantics.
    - ``artifacts/03/t_sol.json`` the analytic bounds, i.e. the answer key for
      how fast the kernel is supposed to get.
    - ``artifacts/02/`` reference timings, and every other problem's data.

    So the agent's ``verify`` runs against this tree instead of the repo. It holds
    the evaluation code and nothing else, which keeps the feedback loop fully
    functional while removing the parts that would let a kernel be tuned to its
    grader.

    This is a *reduction in exposure*, not a security boundary, and the
    distinction is worth keeping straight: the agent is a local root process and
    could still walk the filesystem. What makes the result defensible is that
    scoring re-evaluates from a fingerprinted tree, and that out-of-packet reads
    are detected and recorded per session.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for rel in VERIFY_ROOT_CONTENTS:
        src = repo_root / rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            continue
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc"))
        elif src.is_file():
            shutil.copy2(src, target)

    # The FlashInfer-Bench safetensors inputs are resolved relative to
    # FLASHINFER_TRACE_DIR. Symlinked rather than copied: 304 blobs, and they are
    # read-only inputs the agent is entitled to use.
    blobs = repo_root / "data" / "flashinfer-trace"
    if blobs.exists():
        link = dest / "data" / "flashinfer-trace"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(blobs)
    return dest


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def amd_workload_path(problem_dir: Path, workloads_root: Path | None) -> Path:
    """The workload file to hand the agent.

    Prefers the AMD-derived tolerances from task 05. Refuses to fall back
    silently: the shipped ``workload.jsonl`` carries B200 tolerances, and
    judging an AMD kernel correct-or-not by a tolerance measured on other
    silicon is prime directive 2 in its most consequential form. If a problem
    has no AMD entry (the 15 deferred NVFP4 problems), the caller decides what
    to do about it -- this raises rather than quietly downgrading.
    """
    if workloads_root is None:
        return problem_dir / "workload.jsonl"
    cand = workloads_root / problem_dir.parent.name / problem_dir.name / "workload.jsonl"
    if not cand.exists():
        raise FileNotFoundError(
            f"no AMD-derived workloads for {problem_dir.parent.name}/"
            f"{problem_dir.name} under {workloads_root}"
        )
    return cand


def build_packet(
    problem_dir: Path,
    packet_dir: Path,
    *,
    max_attempts: int,
    author: str,
    workloads_root: Path | None,
    verify_root: Path | None = None,
    clean: bool = True,
) -> dict:
    """Materialize a task packet. Returns a small manifest describing it."""
    problem_key = f"{problem_dir.parent.name}/{problem_dir.name}"
    if clean and packet_dir.exists():
        shutil.rmtree(packet_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)

    definition = json.loads((problem_dir / "definition.json").read_text())
    workload_src = amd_workload_path(problem_dir, workloads_root)
    workload_lines = [
        ln for ln in workload_src.read_text().splitlines() if ln.strip()
    ]

    shutil.copy2(problem_dir / "definition.json", packet_dir / "definition.json")
    (packet_dir / "workload.jsonl").write_text("\n".join(workload_lines) + "\n")

    # reference.py is written from definition.json's own `reference` field rather
    # than copied from the dataset directory. The materializer writes the source
    # to both places and round-trip-checks them, but the field is the one the
    # harness actually executes, so it is the one the agent should read.
    (packet_dir / "reference.py").write_text(definition["reference"])

    harness_root = verify_root or REPO_ROOT
    _write_executable(
        packet_dir / "verify",
        VERIFY_SCRIPT.format(
            python=os.environ.get("SOLEXBENCH_PYTHON", "python3"),
            verify_py=harness_root / "scripts" / "agent_verify.py",
            packet_dir=packet_dir,
        ),
    )

    suggested = f"{problem_dir.name}_agent".replace("-", "_")
    (packet_dir / "TASK.md").write_text(
        TASK_TEMPLATE.format(
            problem_key=problem_key,
            n_workloads=len(workload_lines),
            suggested_name=suggested,
            definition_name=definition["name"],
            author=author,
            language_menu=LANGUAGE_MENU,
            max_attempts=max_attempts,
            attempt_hint=(
                "counts are tracked for you and each run tells you how many remain"
            ),
        )
    )

    manifest = {
        "problem": problem_key,
        "problem_dir": str(problem_dir),
        "packet_dir": str(packet_dir),
        "definition_name": definition["name"],
        "n_workloads": len(workload_lines),
        "workload_source": str(workload_src),
        "tolerances": "amd-derived" if workloads_root else "dataset-shipped-b200",
        "max_attempts": max_attempts,
        "verify_root": str(harness_root),
    }
    (packet_dir / ".packet.json").write_text(json.dumps(manifest, indent=2))
    return manifest
