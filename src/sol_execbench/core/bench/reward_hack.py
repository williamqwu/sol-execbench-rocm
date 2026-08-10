# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward hack defenses for SOL ExecBench evaluation.

Provides detection functions for four common reward-hacking patterns.
The identity of torch.cuda.Event.elapsed_time is captured at module load
time — before any user code is imported — so patching after the fact is
detected.
"""

from __future__ import annotations

from typing import Any, List

import torch

# ---------------------------------------------------------------------------
# Capture timing function identity at module load, before any user code runs.
# Used by check_monkey_patch() to detect post-import patching.
# ---------------------------------------------------------------------------
_ELAPSED_TIME_ADDR: int | None = None

try:
    import torch.cuda as _tc_init

    _ELAPSED_TIME_ADDR = id(_tc_init.Event.elapsed_time)
except Exception:
    pass


class RewardHackDetected(RuntimeError):
    """Raised when a reward-hacking pattern is detected in a submission."""


def check_monkey_patch() -> None:
    """Detect if torch.cuda.Event.elapsed_time has been patched.

    Compares the current function identity against the address captured at
    module load time.  Must be called before the timed section.

    Raises:
        RewardHackDetected: If the timing function has been replaced.
    """
    try:
        import torch.cuda as _tc

        if (
            _ELAPSED_TIME_ADDR is not None
            and id(_tc.Event.elapsed_time) != _ELAPSED_TIME_ADDR
        ):
            raise RewardHackDetected(
                "torch.cuda.Event.elapsed_time has been monkey-patched"
            )
    except RewardHackDetected:
        raise
    except Exception:
        pass


def check_thread_injection(threads_before: int, threads_after: int) -> None:
    """Detect if user code spawned background threads.

    Capture ``threading.active_count()`` before and after the user call and
    pass both values here.

    Raises:
        RewardHackDetected: If the thread count increased.
    """
    if threads_after > threads_before:
        raise RewardHackDetected(
            f"Thread injection detected: "
            f"{threads_after} threads after call vs {threads_before} before"
        )


def check_lazy_outputs(outputs: List[Any]) -> None:
    """Detect lazy/proxy tensors in the user output.

    Uses strict ``type()`` equality — not ``isinstance`` — so any subclass
    (including ``FakeTensor``) is rejected.

    Raises:
        RewardHackDetected: If any output is not exactly ``torch.Tensor``.
    """
    for t in outputs:
        if type(t) is not torch.Tensor:
            raise RewardHackDetected(
                f"Lazy evaluation detected: output is {type(t).__name__}, not torch.Tensor"
            )


def snapshot_critical_functions(namespace: dict, names: List[str]) -> dict[str, int]:
    """Capture ``id()`` of named functions from a namespace.

    Call this **before** user code is imported.  Pass the returned dict to
    :func:`check_eval_integrity` after user code runs.

    Args:
        namespace: The globals dict to snapshot (typically ``globals()``).
        names: Function names to capture.

    Returns:
        Mapping of name → ``id()`` for each name present in *namespace*.
    """
    return {name: id(namespace[name]) for name in names if name in namespace}


def check_default_stream() -> None:
    """Verify user code left the current stream as the default stream.

    AMD: an interim guard, and it is load-bearing only until task 04 lands.

    Upstream's strongest concurrency defense is the CUPTI activity-sequence
    count assertion: work issued on a stream the harness is not watching still
    shows up as unattributed activity. The AMD port ships on ``hip_events``
    first, which cannot see that -- an event pair recorded on the default
    stream simply does not observe a kernel running on another one, so the
    work is free.

    So while the port is on events, the stream itself is checked. This is
    weaker (it catches a submission that *leaves* a non-default stream
    current, not one that carefully restores it), which is exactly why the
    posture is recorded per trace rather than assumed: anything measured
    before the rocprofiler source lands was measured under this weaker guard.

    Raises:
        RewardHackDetected: If the current stream is not the default stream.
    """
    try:
        if not torch.cuda.is_available():
            return
        current = torch.cuda.current_stream()
        default = torch.cuda.default_stream()
    except Exception:
        return
    if current != default:
        raise RewardHackDetected(
            f"Non-default stream is current after user code: {current!r}. "
            f"Work issued on an unwatched stream is not observed by the "
            f"event-pair timing methodology."
        )


def install_stream_tracking() -> None:
    """Close the hole ``check_default_stream`` above only narrows.

    That guard catches a submission that *leaves* a non-default stream current.
    A submission that restores it walks straight past -- and the cost of doing
    so was never measured until 2026-08-10, when it turned out to be **1743x**
    in the adversarial form and **1.42x** in a submitted kernel that was not
    trying to exploit anything (`artifacts/11/side-stream-timing-hole.json`).

    The durable fix is the rocprofiler methodology, which attributes by
    activity record and closes this by construction -- but switching the
    methodology of record would move every measurement ever taken, so it is not
    something to do to get past an obstacle. This joins the submission's own
    streams into the current one before the end event is recorded instead,
    which leaves a single-stream submission on exactly the path it was on.

    Delegates so the defenses stay listed in one place; the mechanism is in
    ``bench/streams.py``.
    """
    from sol_execbench.core.bench import streams

    streams.install()


# Vendor management tools. A submission that raises the clock cap mid-run
# defeats the locked-clock calibration entirely -- and unlike most exploits it
# leaves no trace in the output, because the numbers stay self-consistent.
# nvidia-smi is listed too: the NVIDIA image has the same latent hole.
_SMI_BINARIES = frozenset(
    {"rocm-smi", "amd-smi", "amdsmi", "rocm_smi", "nvidia-smi", "rocm-smi.py"}
)


def _names_in(cmd: Any) -> list[str]:
    """Executable basenames referenced by a subprocess-style argument."""
    import os
    import shlex

    if isinstance(cmd, (list, tuple)):
        parts = [str(c) for c in cmd]
    elif isinstance(cmd, (str, bytes)):
        text = cmd.decode() if isinstance(cmd, bytes) else cmd
        try:
            parts = shlex.split(text)
        except ValueError:
            parts = text.split()
    else:
        return []
    return [os.path.basename(p) for p in parts]


def check_smi_invocation(cmd: Any) -> None:
    """Reject a subprocess invocation that reaches a GPU management tool.

    Raises:
        RewardHackDetected: If *cmd* invokes rocm-smi / amd-smi / nvidia-smi.
    """
    hit = _SMI_BINARIES.intersection(_names_in(cmd))
    if hit:
        raise RewardHackDetected(
            f"Submission invoked a GPU management tool: {sorted(hit)}. "
            f"Changing clocks, power caps or partitioning from inside a "
            f"submission invalidates the locked-clock calibration every "
            f"measurement in this benchmark depends on."
        )


#: Global backend switches that lower the precision of matmul below the dtype
#: the tensors declare. Names differ across torch versions, so each is probed
#: rather than assumed present -- a missing one is a torch that does not have
#: that knob, not a failure to guard.
_PRECISION_KNOBS = (
    ("torch.backends.cuda.matmul", "allow_tf32"),
    ("torch.backends.cuda.matmul", "fp32_precision"),
    ("torch.backends.cudnn", "allow_tf32"),
    ("torch.backends.cudnn.conv", "fp32_precision"),
    ("torch.backends", "fp32_precision"),
)


def _resolve(path: str):
    import importlib
    obj = importlib.import_module("torch")
    for part in path.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def snapshot_matmul_precision() -> dict[str, Any]:
    """Capture the backend precision switches. Call BEFORE importing user code."""
    out: dict[str, Any] = {}
    for mod_path, attr in _PRECISION_KNOBS:
        mod = _resolve(mod_path)
        if mod is not None and hasattr(mod, attr):
            try:
                out[f"{mod_path}.{attr}"] = getattr(mod, attr)
            except Exception:                      # a knob that raises on read
                pass
    return out


def check_matmul_precision(snapshot: dict[str, Any]) -> None:
    """Reject a submission that lowered matmul precision behind the dtypes.

    AMD, 2026-08-10 (STATE.md D35). `torch.backends.cuda.matmul.allow_tf32 =
    True` takes fp32 matmul on this part from 31,834 to 147,954 MAC/cycle --
    **4.52x** -- from one line, on unchanged fp32 tensors, with no cast
    anywhere in the kernel. Measured on MI350X, GPU 0, 8192-cube GEMM.

    It is banned for a reason that is about the score and not about taste.
    T_SOL is derived from the MAC/cycle rate of the dtype the problem
    DECLARES. A submission computing the same problem on a 4.5x datapath is
    not measured against its own roofline, it is measured against somebody
    else's -- so `S` stops meaning "fraction of the way to the hardware limit"
    and starts being reachable above 1. Either the arithmetic happens at the
    declared precision, or every bound in the manifest has to be re-derived at
    the cheapest precision that passes tolerance, which is not this benchmark.

    Tolerance is not a sufficient defence here and that is the whole point.
    The AMD tolerances are run-to-run error of the reference times 1.25
    (task 05); tf32 error is usually larger and usually fails, and "usually"
    is exactly the gap -- the case that passes is silent, is 4.5x fast, and
    looks like an excellent kernel.

    An explicit `.to(torch.bfloat16)` is the same exploit written visibly and
    is caught by `test_precision_downgrade`. This catches the invisible form.
    """
    for name, before in snapshot.items():
        mod = _resolve(name.rsplit(".", 1)[0])
        after = getattr(mod, name.rsplit(".", 1)[1], None) if mod else None
        if after != before:
            raise RewardHackDetected(
                f"Matmul precision was lowered by the submission: {name} "
                f"changed from {before!r} to {after!r}. The declared dtype is "
                f"the contract for the arithmetic, not only for the interface "
                f"-- T_SOL is derived from that dtype's MAC/cycle rate, and a "
                f"kernel computing on a faster datapath is not measured "
                f"against its own bound.")


def install_smi_guard() -> None:
    """Block GPU management tools from submission subprocesses.

    Call **before** importing user code. Wraps the process-spawning entry
    points rather than relying on PATH sanitation, because a submission can
    always call an absolute path.

    Idempotent: re-installing does not stack wrappers.
    """
    import os
    import subprocess

    def wrap(mod: Any, name: str, argindex: int = 0) -> None:
        original = getattr(mod, name, None)
        if original is None or getattr(original, "_solb_smi_guarded", False):
            return

        def guarded(*args, **kwargs):
            if len(args) > argindex:
                check_smi_invocation(args[argindex])
            return original(*args, **kwargs)

        guarded._solb_smi_guarded = True          # type: ignore[attr-defined]
        guarded._solb_original = original         # type: ignore[attr-defined]
        setattr(mod, name, guarded)

    # Popen is the chokepoint for run/call/check_output; the others are
    # wrapped anyway so a direct call is caught with a clearer message.
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        wrap(subprocess, name)
    for name in ("system", "popen"):
        wrap(os, name)
    for name in ("execv", "execvp", "execl", "execlp", "spawnv", "spawnvp"):
        wrap(os, name)


# Patterns screened for in submission SOURCE, before anything is compiled or
# imported. Static screening exists because the runtime guards are Python-level
# and a C++/HIP submission runs underneath them: `install_smi_guard` cannot see
# a `system("rocm-smi ...")` inside a compiled extension, and the stream check
# cannot see a stream created with `hipStreamCreate` and used only inside the
# kernel launch.
#
# Each entry is (regex, why). The message is shown to the submitter, so it says
# what to do instead rather than only what was refused.
_SOURCE_HAZARDS: tuple[tuple[str, str], ...] = (
    (
        r"\b(hip|cuda)StreamCreate(WithFlags|WithPriority)?\b",
        "creates a raw stream; submissions must issue work on the default "
        "stream so the harness can attribute it. Use the stream the harness "
        "provides.",
    ),
    (
        r"\b(rocm-smi|amd-smi|nvidia-smi|amdsmi_set|rsmi_dev_)\b",
        "invokes a GPU management interface; changing clocks, power caps or "
        "partitioning from a submission invalidates the locked-clock "
        "calibration the whole benchmark is expressed at.",
    ),
    (
        r"power_dpm_force_performance_level|/sys/class/drm/card",
        "writes GPU policy through sysfs, which is the same escape as calling "
        "an smi tool.",
    ),
    (r"\bsudo\b", "attempts privilege escalation."),
    (
        r"\bhipDeviceSetCacheConfig\b|\bhipSetDeviceFlags\b",
        "changes device-wide configuration that outlives the submission and "
        "would perturb subsequent measurements on the same GPU.",
    ),
)


# AMD: hazards in the source FILENAME rather than in its contents.
#
# Python imports some names automatically at interpreter startup, before any
# harness guard has installed itself and outside every timed region. A submission
# that ships one is executing code the harness never invoked, which is the same
# escape as monkey-patching — and it is invisible to a content scan, because the
# content can be, and was, entirely innocuous.
#
# Observed: a submission shipped `sitecustomize.py` defining `enum.StrEnum` to
# work around an interpreter older than the project requires. That particular
# patch was benign and arguably a repair; it was also the difference between a
# problem scoring 16/16 and not evaluating at all, which is far too much leverage
# to leave outside the record.
_PATH_HAZARDS: tuple[tuple[str, str], ...] = (
    (
        r"(^|/)(sitecustomize|usercustomize)\.py$",
        "is imported automatically at interpreter startup, so it runs before any "
        "harness guard and outside every timed region.",
    ),
    (
        r"\.pth$",
        "a .pth file in a site directory executes lines beginning with `import` "
        "at interpreter startup, before the harness runs.",
    ),
    (
        r"(^|/)conftest\.py$",
        "is imported automatically by pytest, outside the evaluation path.",
    ),
)


def static_source_screen(sources: Any) -> list[tuple[str, str, str]]:
    """Screen submission sources for hazards. Returns [(path, pattern, why)].

    *sources* is any iterable of objects with ``path`` and ``content``, or of
    ``(path, content)`` pairs.

    Screens both the content and the **filename**: some names the interpreter
    imports by itself, so their content never has to look suspicious.

    Reports rather than raises, so a caller can decide between refusing the
    submission and recording the finding. Empty list means clean.
    """
    import re

    findings: list[tuple[str, str, str]] = []
    for src in sources or ():
        if isinstance(src, (tuple, list)) and len(src) == 2:
            path, content = src
        else:
            path, content = getattr(src, "path", "?"), getattr(src, "content", "")
        for pattern, why in _SOURCE_HAZARDS:
            if re.search(pattern, content or ""):
                findings.append((str(path), pattern, why))
        for pattern, why in _PATH_HAZARDS:
            if re.search(pattern, str(path)):
                findings.append((str(path), pattern, why))
    return findings


def check_static_source_screen(sources: Any) -> None:
    """Raise if any submission source trips the static screen."""
    findings = static_source_screen(sources)
    if findings:
        detail = "; ".join(f"{p}: {why}" for p, _, why in findings)
        raise RewardHackDetected(f"Submission source rejected — {detail}")


def compute_partition_mode() -> str | None:
    """Current compute-partition mode (SPX / DPX / CPX ...), or None.

    Recorded on every trace. MI300-series and later expose compute
    partitioning with no NVIDIA analogue, and the partition mode changes how
    many CUs a kernel can reach -- so two traces taken under different modes
    are not comparable, and nothing else in a trace would reveal the
    difference.
    """
    import glob
    from pathlib import Path

    modes = set()
    for f in sorted(glob.glob("/sys/class/drm/card*/device/current_compute_partition")):
        try:
            modes.add(Path(f).read_text().strip())
        except OSError:
            continue
    if not modes:
        return None
    # A node with mixed partition modes is a node whose GPUs are not
    # interchangeable; say so rather than picking one.
    return sorted(modes)[0] if len(modes) == 1 else "MIXED:" + ",".join(sorted(modes))


def check_eval_integrity(snapshot: dict[str, int], namespace: dict) -> None:
    """Verify that critical eval-driver functions have not been replaced.

    Compares the current ``id()`` of each snapshotted name against the
    value captured before user code was imported.

    Args:
        snapshot: The dict returned by :func:`snapshot_critical_functions`.
        namespace: The current globals dict to check.

    Raises:
        RewardHackDetected: If any function identity has changed.
    """
    for name, expected_id in snapshot.items():
        current = namespace.get(name)
        if current is None or id(current) != expected_id:
            raise RewardHackDetected(
                f"Eval driver integrity violated: '{name}' has been monkey-patched"
            )
