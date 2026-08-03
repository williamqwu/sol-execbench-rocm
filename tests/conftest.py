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

from pathlib import Path
from typing import List

import pytest


def _gpu_sm_version() -> int:
    """Return the SM version of the current GPU (e.g. 90, 100), or 0 if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except ImportError:
        return 0


requires_sm100 = pytest.mark.skipif(
    _gpu_sm_version() < 100,
    reason=f"Requires sm_100+ (detected sm_{_gpu_sm_version()})",
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "timing_serial: GPU timing tests (skipped by default; run with: pytest tests -m timing_serial -n 0)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Skip tests based on hardware availability.

    Also skips timing_serial tests unless explicitly selected with -m.
    """
    sm_version = _gpu_sm_version()
    skip_timing = pytest.mark.skip(
        reason="timing_serial tests skipped by default; run with: pytest tests -m timing_serial -n 0"
    )

    # If the user passed -m that includes timing_serial, don't auto-skip them.
    markexpr = config.getoption("-m", default="")
    timing_selected = "timing_serial" in markexpr

    amd = _is_amd()
    for item in items:
        if sm_version < 100 and any(item.iter_markers(name="requires_cutile")):
            item.add_marker(
                pytest.mark.skip(
                    reason=f"cuTile requires sm_100+ (detected sm_{sm_version})"
                )
            )
        if "timing_serial" in item.keywords and not timing_selected:
            item.add_marker(skip_timing)
        # AMD: example solutions written in an NVIDIA-only language.
        if amd and item.fspath.basename == "test_examples.py":
            case_id = item.name[item.name.find("[") + 1 : -1]
            if case_id.endswith(_NVIDIA_ONLY_EXAMPLE_SUFFIXES):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{case_id}: NVIDIA-only solution language; the "
                        f"AMD-native counterpart is separate work"
                    )
                )


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated temporary cache directory for each test.

    Sets SOLEXECBENCH_CACHE_PATH so every builder writes build artifacts into a
    fresh temp directory, preventing pollution between tests.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SOLEXECBENCH_CACHE_PATH", str(cache_dir))
    return cache_dir

# --- SOL-ExecBench-AMD additions ------------------------------------------
# The CPU-verified activity package is imported by tests/test_gpu_activity.py
# as a top-level module, mirroring how the rocprofiler shim will import it.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(_Path(__file__).resolve().parent.parent / "src" / "solexbench_rocm" / "activity")
)


def _is_amd() -> bool:
    try:
        import torch

        return getattr(torch.version, "hip", None) is not None
    except ImportError:
        return False


# Test modules that import an NVIDIA-only wheel at module scope. On ROCm these
# raise ModuleNotFoundError during *collection*, which aborts the whole run
# rather than failing one test -- so they are skipped by path before import.
# They still run normally on an NVIDIA host, which is the point: the NVIDIA
# path stays the regression reference for the port.
_NVIDIA_ONLY_TEST_FILES = {
    "test_cupti_utils.py",       # cupti
    "test_cudnn_frontend.py",    # cudnn
    "test_cutedsl.py",           # cutlass / CuTe DSL
    "test_cutile.py",            # cuda-tile; no AMD equivalent by design
    # Toolchain smoke tests: nvcc/cuDNN/CUTLASS compile-and-link.
    "test_cuda.py",
    "test_cudnn.py",
    "test_cutlass.py",
}

# Example solutions written in an NVIDIA-only language. Languages are a
# property of solutions, not problems, so skipping these loses no problem
# coverage -- the AMD-native counterparts (hip_cpp, ck, miopen, hipblaslt) are
# separate work. The pytorch and triton examples DO run on ROCm and are not
# skipped: they are the end-to-end proof that the ported harness works.
_NVIDIA_ONLY_EXAMPLE_SUFFIXES = (
    "_cuda", "_cutlass", "_cudnn", "_cute_dsl", "_cutile",
)


def pytest_ignore_collect(collection_path, config):
    if _is_amd() and collection_path.name in _NVIDIA_ONLY_TEST_FILES:
        return True
    return None


# Upstream tests that assert NVIDIA-path behaviour directly: nvidia-smi clock
# locking, `--use_fast_math` / `-lcuda` defaults, the CUTLASS include dir, and
# sm_100a gencode injection. Run on an AMD host they would take the AMD branch
# and fail -- but they are the regression reference the port must not lose
# (tasks/02 guard rail), so instead of skipping them the vendor is pinned to
# "nvidia" for their duration. That keeps them testing what they were written
# to test, on either host.
_NVIDIA_PATH_TESTS = {
    "test_clock_lock.py": None,          # whole module
    "test_build_ext.py": {
        "test_no_sources_raises",
        "test_default_cuda_cflags",
        "test_default_cutlass_dir",
        "test_custom_cutlass_dir",
    },
    "test_problem_packager.py": {"test_gencode_injected_for_blackwell"},
}


@pytest.fixture(autouse=True)
def _pin_vendor_for_nvidia_path_tests(request, monkeypatch):
    """Pin detect_vendor() to nvidia for tests that assert NVIDIA behaviour."""
    selected = _NVIDIA_PATH_TESTS.get(Path(str(request.fspath)).name, False)
    if selected is False:
        return
    if selected is not None and request.node.originalname not in selected:
        return
    try:
        from sol_execbench.core.bench import device
    except ImportError:
        return
    monkeypatch.setattr(device, "detect_vendor", lambda: "nvidia")
