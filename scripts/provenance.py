#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Provenance stamping. Every artifact carries one of these.

An artifact without provenance is not usable for scoring: six months from now
nobody can tell whether a number was measured at the right clock, on the right
stack, from the right commit.

CPU-safe: every GPU/ROCm probe degrades to None rather than raising, so this
module imports and runs anywhere.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_sha(repo_root: Path | None = None) -> str | None:
    cwd = str(repo_root) if repo_root else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=cwd, timeout=30,
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=cwd, timeout=30,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return None


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    info = {
        "available": True,
        "version": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
    }
    try:
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception:
        info["device_count"] = 0
        info["devices"] = []
    return info


def rocm_info() -> dict:
    version = None
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            version = Path(path).read_text().strip()
            break
        except Exception:
            continue
    return {
        "version": version,
        "driver": _run(["cat", "/sys/module/amdgpu/version"]),
        "amd_smi": _run(["amd-smi", "version"]),
    }


def kernel_stack() -> dict:
    """Which kernel-authoring toolchains were available, and from where.

    Upstream needed none of this: a CUDA C++ solution is pinned by the CUDA
    version already in ``rocm``/``torch``. The AMD side is different in a way
    that matters for the agent scoreboard, because an agent may write a solution
    in any of these and the *build* of the toolchain then decides what the
    kernel is:

    - ``triton`` may resolve to a release wheel or to a development checkout.
      A Gluon kernel that compiles against a checkout may not exist at all in a
      release, so "triton 3.6.0" alone does not identify the compiler. The
      import path is recorded for that reason.
    - ``aiter`` is a source checkout with its own git SHA; the library *is* the
      kernel, so its SHA is part of the result.
    - ``ck`` / ``ck_tile`` / ``hipblaslt`` / ``miopen`` ship with ROCm, so the
      ROCm version pins them, but their presence is recorded so a failure to
      use one can be distinguished from an inability to.

    Never raises: a missing toolchain is recorded as absent, which is itself a
    fact worth having on the artifact.
    """
    import importlib.metadata as md
    import importlib.util

    def _pkg(name: str, module: str | None = None) -> dict:
        """Locate a package WITHOUT importing it.

        Deliberately uses ``find_spec`` rather than ``import_module``: importing
        ``aiter`` loads a compiled extension, and a provenance stamp taken in the
        middle of a timing run must not create a HIP context or perturb the
        device it is describing. Versions come from installed metadata, which
        needs no import either.
        """
        entry: dict = {"available": False}
        try:
            spec = importlib.util.find_spec(module or name)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            return entry
        if spec is None:
            return entry
        entry["available"] = True
        entry["path"] = spec.origin
        try:
            entry["dist_version"] = md.version(name)
        except Exception:
            entry["dist_version"] = None
        return entry

    triton = _pkg("triton")
    if triton.get("available"):
        # A dist version carrying a git suffix, or an import path outside
        # site-packages, both mean "not a release wheel".
        path = triton.get("path") or ""
        dist = triton.get("dist_version") or ""
        triton["is_release_wheel"] = ("site-packages" in path) and ("git" not in dist)
        triton["gluon"] = _pkg("triton", "triton.experimental.gluon").get("available", False)

    stack: dict = {
        "triton": triton,
        "aiter": _pkg("aiter"),
        "hipcc": _run(["hipcc", "--version"]),
        "rocm_libraries": {
            name: Path(f"/opt/rocm/include/{name}").exists()
            for name in ("ck", "ck_tile", "hipblaslt", "miopen")
        },
    }

    aiter = stack["aiter"]
    if aiter.get("available") and aiter.get("path"):
        repo = Path(aiter["path"]).resolve().parent.parent
        aiter["git_sha"] = git_sha(repo)

    return stack


def part_name() -> str | None:
    """The Instinct part these measurements were taken on, e.g. ``MI355X``.

    Recorded explicitly because MI350X and MI355X are the same gfx950 die and
    are therefore indistinguishable from ``gcnArchName`` alone, while their
    measured quantities -- F_LOCK above all -- do not transfer between them.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from solexbench_rocm.parts import detect_part

        return detect_part().name
    except Exception:
        return None


def env_mode() -> dict:
    """Whether this ran in the pinned container or natively against it.

    ``env/solb`` runs inside ``solbench:rocm7.2-torch2.9.1``; ``env/solb-native``
    reproduces the same environment contract on a node with no docker, and
    asserts the stack matches rather than assuming it. The two are intended to
    be equivalent, so which one produced a number is exactly the kind of thing
    that should be on the record rather than inferred later.
    """
    return {
        "mode": os.environ.get("SOLEXBENCH_ENV_MODE", "unknown"),
        "in_docker": Path("/.dockerenv").exists(),
        "stack_drift_allowed": os.environ.get("SOLB_ALLOW_STACK_DRIFT") == "1",
    }


def f_lock_mhz() -> int | None:
    """F_LOCK for this artifact: the clock its measurements were taken at.

    Resolution order:
      1. ``SOLEXBENCH_F_LOCK_MHZ`` — an explicit override.
      2. The measured preset for the GPU this process can see.

    Step 2 exists because an env var is exactly the kind of thing a sweep
    forgets to export, and an artifact whose F_LOCK is null cannot be used for
    scoring.

    **This is the INTENDED clock, and it cannot tell you the hardware agreed.**
    An earlier version of this docstring claimed that reading the value from
    the same table the lock is applied from meant the recorded and applied
    clocks could not disagree. That is false, and it cost a day: the table is
    not the hardware. See ``clock_lock_state`` for the readback that can.

    Note this returns the ACHIEVED clock, not the requested one: on AMD they
    differ (see ``ClockPreset``), and the achieved value is the one every
    T_SOL and T_b is expressed at.
    """
    env = os.environ.get("SOLEXBENCH_F_LOCK_MHZ")
    if env:
        try:
            return int(env)
        except ValueError:
            return None
    try:
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from sol_execbench.core.bench.config import get_clock_preset

        preset = get_clock_preset(torch.cuda.get_device_name(0))
        return preset.f_lock_mhz if preset else None
    except Exception:
        return None


def clock_lock_state() -> dict | None:
    """The clock lock as the DEVICES report it, not as the preset table says.

    ``f_lock_mhz()`` cannot detect a wrong lock, because it reads the table the
    lock is supposed to have been applied from. On 2026-08-04 an unreset
    determinism sweep left this node at a 1900 MHz setpoint; 138 authoritative
    T_b were then measured at ~1860 MHz and every one was stamped
    ``f_lock_mhz: 1640``. Nothing downstream could see it — including
    ``build_manifest.collect_t_b``, which rejects artifacts measured at the
    wrong clock by comparing that same stamp.

    ``max_clk`` is the determinism setpoint read back off the device, so it is
    a genuinely independent second source. It reads correctly while the GPU is
    idle, which is what makes it usable here: an artifact is normally stamped
    before its load starts, so the live clock would just read the idle floor.

    Every GPU is checked, not just the one being timed. A partial lock is the
    dangerous case, and `rocm-smi --setperfdeterminism` is applied node-wide
    often enough that "some GPUs took it" is a real outcome.
    """
    expected = None
    try:
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from sol_execbench.core.bench.config import get_clock_preset

        preset = get_clock_preset(torch.cuda.get_device_name(0))
        # The REQUESTED setpoint, not the achieved F_LOCK: `max_clk` reads back
        # what was asked for.
        expected = preset.gpu_clk_mhz if preset else None
    except Exception:
        pass

    setpoints: list[int | None] = []
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        for h in amdsmi.amdsmi_get_processor_handles():
            try:
                info = amdsmi.amdsmi_get_clock_info(h, amdsmi.AmdSmiClkType.GFX)
                setpoints.append(info.get("max_clk"))
            except Exception:
                setpoints.append(None)
    except Exception:
        return None

    import glob

    levels = []
    for f in sorted(glob.glob(
            "/sys/class/drm/card*/device/power_dpm_force_performance_level")):
        try:
            levels.append(Path(f).read_text().strip())
        except Exception:
            pass

    distinct = sorted({s for s in setpoints if s})
    return {
        "expected_setpoint_mhz": expected,
        "setpoint_mhz_per_gpu": setpoints,
        "perf_levels": sorted(set(levels)),
        "agrees": bool(expected) and distinct == [expected]
                  and set(levels) == {"perf_determinism"},
    }


def assert_clock_lock() -> None:
    """Refuse to measure when the hardware lock is not what the preset claims.

    Timing runners call this at startup. A measurement taken at the wrong clock
    is not a slightly-wrong measurement -- T_b is a wall-clock time, so it is
    wrong by the clock ratio, silently, and looks entirely plausible.
    """
    st = clock_lock_state()
    if st is None:
        raise SystemExit(
            "REFUSING to measure: could not read the clock lock back from the "
            "devices (amdsmi unavailable). An unverified lock means every "
            "timing is taken at an unknown clock.")
    if not st["agrees"]:
        raise SystemExit(
            "REFUSING to measure: the hardware clock lock is not what the "
            f"preset table claims.\n"
            f"  expected setpoint : {st['expected_setpoint_mhz']} MHz on every GPU\n"
            f"  actual setpoints  : {st['setpoint_mhz_per_gpu']}\n"
            f"  perf levels       : {st['perf_levels']}\n"
            "Apply it with `clock_calibrate.py lock --freq-mhz <setpoint> "
            "--all-gpus`, then re-run.")


# GFX activity, not power, decides whether a sample counts. Power lags: a GPU
# running kernels between compilations reads 100% busy at 273 W, barely above
# the ~240 W idle draw, so a power threshold high enough to mean "working"
# discards most of the samples a timing run offers.
_BUSY_ACTIVITY_PCT = 50
_DEFAULT_TOLERANCE = 0.03


class ClockMonitor:
    """Sample the GFX clock while a measurement runs, and report what it was.

    ``assert_clock_lock()`` proves the setpoint was *applied*. It cannot prove
    what the silicon did with it, and on this part those are different
    questions: at one setting GPU 0 holds 1647 MHz alone and 1394 MHz with its
    siblings loaded, and GPU 2 never reaches the request at all (D27). So F_LOCK
    as a table constant is a claim about one GPU in one node condition, and the
    only way an artifact can be trusted to have been measured at it is to have
    measured it.

    Sampling is deliberately indirect: it reads every GPU and takes the busiest
    one, rather than resolving which physical GPU ``HIP_VISIBLE_DEVICES`` maps
    to. That needs no torch and no PCI mapping, so the monitor cannot create a
    HIP context on the device it is describing before the measurement does --
    and because the authoritative pass requires an idle node, the busiest GPU
    *is* the one being timed. ``busy_gpus`` records how many were working, which
    catches the case where that assumption did not hold.

    Samples taken while nothing is busy are dropped. A runner spends much of its
    wall clock in Triton compilation, which is CPU-bound; counting those would
    drag the median toward the idle floor and flag every problem. Measured on
    this node: 25 s of a `max_autotune` compile yielded zero busy samples, which
    is why ``n_busy_samples`` is reported and a count of zero is not a pass.

    Known limitation, stated rather than papered over: this covers GPU work
    across the whole runner, which includes autotuning, not only the timed
    iterations. Isolating those needs a hook inside the harness's timing loop.
    It is enough for what went wrong -- a node at the wrong setpoint, or a node
    that was not idle, is wrong for the entire run -- and ``min_mhz``/``max_mhz``
    are recorded so a run whose phases used different clocks shows it.
    """

    def __init__(self, hz: float = 5.0, tolerance: float | None = None):
        self.hz = hz
        if tolerance is None:
            try:
                tolerance = float(os.environ.get(
                    "SOLEXBENCH_CLOCK_TOLERANCE", _DEFAULT_TOLERANCE))
            except ValueError:
                tolerance = _DEFAULT_TOLERANCE
        self.tolerance = tolerance
        self._samples: list[tuple[int, float, int]] = []   # mhz, watts, gpu
        self._busy_gpus: set[int] = set()
        self._thread = None
        self._halt = None
        self._error: str | None = None

    def _sample_once(self, amdsmi, handles) -> None:
        busiest, busiest_act = None, -1
        for i, h in enumerate(handles):
            try:
                act = amdsmi.amdsmi_get_gpu_activity(h).get("gfx_activity")
            except Exception:
                continue
            if not isinstance(act, int) or act < _BUSY_ACTIVITY_PCT:
                continue
            self._busy_gpus.add(i)
            if act > busiest_act:
                busiest, busiest_act = (i, h), act
        if busiest is None:
            return
        i, h = busiest
        try:
            clk = amdsmi.amdsmi_get_clock_info(
                h, amdsmi.AmdSmiClkType.GFX).get("clk")
            pw = amdsmi.amdsmi_get_power_info(h).get("current_socket_power")
        except Exception:
            return
        if clk:
            self._samples.append((clk, pw or 0.0, i))

    def _loop(self) -> None:
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
        except Exception as e:                              # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            return
        import time as _t
        while not self._halt.is_set():
            self._sample_once(amdsmi, handles)
            self._halt.wait(1.0 / self.hz)
        del _t

    def __enter__(self) -> "ClockMonitor":
        import threading
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._halt is not None:
            self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def summary(self) -> dict:
        """What clock the measurement ran at, and whether that is F_LOCK."""
        import statistics

        expected = f_lock_mhz()
        clk = [s[0] for s in self._samples]
        pw = [s[1] for s in self._samples]
        out: dict = {
            "expected_f_lock_mhz": expected,
            "tolerance": self.tolerance,
            "n_busy_samples": len(clk),
            "busy_gpus": sorted(self._busy_gpus),
            "sampler_error": self._error,
        }
        if not clk:
            # Not a violation: a problem can be short enough, or compile-bound
            # enough, that no sample lands on GPU work. Recorded as unverified
            # rather than silently passed -- assert_clock_lock() still covers
            # the systematic case.
            out.update(median_mhz=None, min_mhz=None, max_mhz=None,
                       median_power_w=None, deviation=None,
                       within_tolerance=None)
            return out
        clk.sort()
        med = statistics.median(clk)
        dev = (med - expected) / expected if expected else None
        out.update(
            median_mhz=med,
            min_mhz=clk[0],
            max_mhz=clk[-1],
            p5_mhz=clk[max(0, int(0.05 * len(clk)) - 1)],
            median_power_w=statistics.median(pw),
            deviation=dev,
            within_tolerance=(dev is not None and abs(dev) <= self.tolerance),
        )
        return out


def stamp(task: str, extra: dict | None = None) -> dict:
    """Build a provenance block. Attach to every artifact."""
    return {
        "_provenance": {
            "task": task,
            "utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(Path(__file__).resolve().parent.parent),
            "host": platform.node(),
            "python": sys.version.split()[0],
            "torch": torch_info(),
            "rocm": rocm_info(),
            "part": part_name(),
            "kernel_stack": kernel_stack(),
            "env": env_mode(),
            "f_lock_mhz": f_lock_mhz(),
            "clock_lock": clock_lock_state(),
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            **(extra or {}),
        }
    }


def write_artifact(path: str | Path, task: str, payload: dict,
                   extra_provenance: dict | None = None) -> Path:
    """Write *payload* to *path* with a provenance block merged in."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {**stamp(task, extra_provenance), **payload}
    p.write_text(json.dumps(doc, indent=2, default=str))
    return p


if __name__ == "__main__":
    print(json.dumps(stamp(sys.argv[1] if len(sys.argv) > 1 else "adhoc"),
                     indent=2, default=str))
