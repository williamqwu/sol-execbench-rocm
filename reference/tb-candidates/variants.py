#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""T_b candidate variants (task 06).

T_b is the *optimized-PyTorch anchor* — the S=0.5 point of the score curve. It
must be re-measured on AMD rather than ported, because Inductor makes different
decisions on ROCm and different fusions are profitable. What must NOT change is
the optimization *policy*, or S=0.5 would mean something different on the two
platforms:

    eager vs torch.compile, whichever measures faster
    contiguity and layout hygiene
    NO handwritten kernels

Each variant is a source-to-source transform of the problem's own reference, so
one set covers all 235 problems and none of it is per-problem authoring. That
is deliberate: the task file says the node's job here is measurement and
selection, not creativity. A problem whose best formulation is genuinely
missing from this set gets a hand-written override in
``reference/tb-candidates/<problem>/vN_<name>.py`` (see README), and the
addition is recorded.

Why no cudagraph modes
----------------------
``mode="reduce-overhead"`` and plain ``max-autotune`` enable CUDA/HIP graphs.
The harness times each iteration with the **shifting allocator**, which hands
every iteration tensors at a different ``data_ptr`` on purpose (it is what
defeats address-keyed output caching, an actual observed exploit family). A
captured graph replays against the addresses it captured, so pairing the two
either crashes or, worse, silently replays stale pointers and reports a
brilliant time for a kernel that computed nothing. ``-no-cudagraphs`` is
therefore not a performance compromise, it is a correctness requirement here.
"""

from __future__ import annotations

from typing import Callable

# The reference module defines `run`. Every wrapper below rebinds `run`, so it
# must first capture the original under a private name -- rebinding in place
# would make the wrapper call itself.
_CAPTURE = "\n_solb_ref_run = run\n"


def _eager(src: str) -> str:
    return src


def _compile(mode: str | None) -> Callable[[str], str]:
    mode_arg = "None" if mode is None else repr(mode)

    def transform(src: str) -> str:
        return (
            src
            + _CAPTURE
            + f'''
import torch as _solb_torch
import torch._dynamo.config as _solb_dynamo_cfg

# STATE.md D50. `dynamic=False` asks for one specialized kernel per shape, and
# the harness sweeps up to 47 shapes through ONE module-level compiled
# callable. Dynamo's default `recompile_limit` is 8: past the eighth distinct
# shape it logs a message and SILENTLY RUNS THE FRAME EAGERLY. 225 of 235
# problems have >=9 distinct shapes, so 2061 of 3957 workloads were never
# compiled at all -- they were timed eager and labelled compile, and the
# numerical failures the board reports for the compile variants are a floor
# rather than a count.
#
# Two changes, and the second matters more than the first: raise the limit far
# past any problem's shape count, and make exhausting it RAISE. A variant that
# cannot compile is a legitimate result the runner records; a variant that
# quietly stops compiling and keeps reporting times is not detectable
# downstream by anything.
_solb_dynamo_cfg.recompile_limit = 256
_solb_dynamo_cfg.accumulated_recompile_limit = 4096
_solb_dynamo_cfg.fail_on_recompile_limit_hit = True

# Compilation happens on the first call for each shape and is therefore
# outside the timed region (the harness warms up before it times).
_solb_compiled = _solb_torch.compile(_solb_ref_run, mode={mode_arg}, dynamic=False)


def run(*args, **kwargs):
    return _solb_compiled(*args, **kwargs)
'''
        )

    return transform


def _contiguous(src: str) -> str:
    return (
        src
        + _CAPTURE
        + '''
import torch as _solb_torch


# Layout hygiene: a non-contiguous input forces strided access in every
# consumer. `.contiguous()` returns self when the tensor already is one, so
# this costs nothing on the common path. Explicitly part of upstream's T_b
# policy, hence a candidate rather than a hidden default.
def run(*args, **kwargs):
    args = tuple(
        a.contiguous() if isinstance(a, _solb_torch.Tensor) else a for a in args
    )
    kwargs = {
        k: (v.contiguous() if isinstance(v, _solb_torch.Tensor) else v)
        for k, v in kwargs.items()
    }
    return _solb_ref_run(*args, **kwargs)
'''
    )


def _compile_contiguous(src: str) -> str:
    return _compile("max-autotune-no-cudagraphs")(_contiguous(src))


# Ordered: cheapest and most likely first, so a truncated sweep still has the
# variants that win most often.
VARIANTS: dict[str, Callable[[str], str]] = {
    # The reference as written. This is the "eager" arm of the policy, and it
    # is also the T_ref the score formula measures headroom against, so it is
    # never skipped.
    "v1_eager": _eager,
    # Default Inductor. Wins on fusible elementwise/normalization chains.
    "v2_compile": _compile(None),
    # Autotuned Triton templates for the GEMM-shaped work. On ROCm this
    # searches a different template space than on CUDA, which is precisely why
    # T_b cannot be ported.
    "v3_compile_max_autotune": _compile("max-autotune-no-cudagraphs"),
    # Layout hygiene alone, without the compiler.
    "v4_contiguous": _contiguous,
    # Both. Sometimes beats v3 because Inductor sees a clean layout.
    "v5_compile_contiguous": _compile_contiguous,
}

__all__ = ["VARIANTS"]
