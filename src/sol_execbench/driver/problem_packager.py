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

"""Package a problem (definition + workloads + solution) into a staging directory.

Produces shell commands that the CLI can run directly via subprocess to compile
C++/CUDA solutions and evaluate them on the GPU.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

from ..core import (
    BenchmarkConfig,
    Definition,
    Solution,
    SupportedHardware,
    SupportedLanguages,
    Trace,
    Workload,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_CPP_LANGUAGES = {
    SupportedLanguages.CUDA_CPP,
    SupportedLanguages.CUTLASS,
    SupportedLanguages.CUDNN,
    SupportedLanguages.CUBLAS,
    # AMD: the C++-compiled languages have the same packaging needs.
    SupportedLanguages.HIP_CPP,
    SupportedLanguages.CK,
    SupportedLanguages.CK_TILE,
    SupportedLanguages.HIPBLASLT,
    SupportedLanguages.MIOPEN,
}

_BLACKWELL_HARDWARE = {SupportedHardware.B200}

# AMD: hardware target -> gfx offload target. MI350X and MI355X are the same
# CDNA4 die and the same ISA target; they differ in cooling and power budget,
# which affects the clock preset, not the compiled code.
_AMD_HARDWARE_ARCH = {
    SupportedHardware.MI355X: "gfx950",
    SupportedHardware.MI350X: "gfx950",
}


def _get_local_sm() -> str | None:
    """Detect the local GPU's SM version via nvidia-smi."""
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        return f"sm_{out.replace('.', '')}"
    except Exception:
        return None


def _get_local_gfx() -> str | None:
    """Detect the local GPU's gfx target.

    AMD: read from torch rather than shelling out to a vendor tool -- the ROCm
    equivalent of ``nvidia-smi --query-gpu=compute_cap`` is
    ``gcnArchName``, which arrives with feature flags attached
    (``gfx950:sramecc+:xnack-``). Only the base target is a valid
    ``--offload-arch`` value.
    """
    try:
        import torch

        name = torch.cuda.get_device_properties(0).gcnArchName
        return name.split(":", 1)[0] or None
    except Exception:
        return None


def _sm_to_gencode(sm: str) -> str:
    """Convert an SM version string (e.g. 'sm_90', 'sm_100a') to a gencode flag."""
    arch = sm.removeprefix("sm_")
    return f"-gencode=arch=compute_{arch},code={sm}"


def _gfx_to_offload_arch(gfx: str) -> str:
    """Convert a gfx target (e.g. 'gfx950') to a hipcc offload flag."""
    return f"--offload-arch={gfx}"


def gencode_flags_for_hardware(hardware=None) -> list[str]:
    """Architecture flags for *hardware*, or for the local GPU if None.

    Shared entry point used by the vendor device layer.
    """
    from sol_execbench.core.bench import device as device_layer

    if device_layer.detect_vendor() == "amd":
        gfx = _AMD_HARDWARE_ARCH.get(hardware) or _get_local_gfx()
        return [_gfx_to_offload_arch(gfx)] if gfx else []
    sm = "sm_100a" if hardware in _BLACKWELL_HARDWARE else _get_local_sm()
    return [_sm_to_gencode(sm)] if sm else []


class ProblemPackager:
    """Stage files for compilation and execution, returning commands for the CLI."""

    def __init__(
        self,
        definition: Definition,
        workloads: list[Workload],
        solution: Solution,
        config: BenchmarkConfig,
        output_dir: Path,
        keep_output_dir: bool = False,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_output_dir = keep_output_dir

        self.definition = definition
        self.workloads = workloads
        self.solution = solution
        self.config = config

        # Write problem files to staging directory up front.
        (self.output_dir / "definition.json").write_text(definition.model_dump_json())
        (self.output_dir / "workload.jsonl").write_text(
            "\n".join(w.model_dump_json() for w in workloads)
        )
        (self.output_dir / "solution.json").write_text(solution.model_dump_json())
        (self.output_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(config))
        )
        self._write_sources()

    def __del__(self):
        if not self.keep_output_dir:
            shutil.rmtree(self.output_dir, ignore_errors=True)

    @property
    def _is_cpp(self) -> bool:
        return any(lang in _CPP_LANGUAGES for lang in self.solution.spec.languages)

    def _inject_gencode_flags(self, sol_dict: dict) -> dict:
        """Auto-inject -gencode flags when no explicit arch flag is set.

        Blackwell targets get sm_100a (required for tcgen05/TMEM instructions).
        LOCAL target detects the compile machine's GPU.
        """
        from sol_execbench.core.bench import device as device_layer

        spec = sol_dict["spec"]
        compile_options = dict(spec.get("compile_options") or {})
        cuda_cflags = list(compile_options.get("cuda_cflags", []))
        target_hw = {h.upper() for h in spec.get("target_hardware", [])}

        # AMD: hipcc takes --offload-arch, and an explicit one must be
        # respected exactly as an explicit -gencode is on NVIDIA.
        if device_layer.detect_vendor() == "amd":
            if any("--offload-arch" in f or "-arch" in f for f in cuda_cflags):
                return sol_dict

            gfx_targets: list[str] = []
            for hw, gfx in _AMD_HARDWARE_ARCH.items():
                if hw.value in target_hw:
                    gfx_targets.append(gfx)
            if SupportedHardware.LOCAL.value in target_hw:
                local_gfx = _get_local_gfx()
                if local_gfx:
                    gfx_targets.append(local_gfx)

            seen_gfx: set[str] = set()
            unique_gfx = [
                g for g in gfx_targets if not (g in seen_gfx or seen_gfx.add(g))
            ]
            if unique_gfx:
                compile_options["cuda_cflags"] = [
                    _gfx_to_offload_arch(g) for g in unique_gfx
                ] + cuda_cflags
                spec["compile_options"] = compile_options
                sol_dict["spec"] = spec
            return sol_dict

        if any("-gencode" in f or "-arch" in f for f in cuda_cflags):
            return sol_dict

        gencode_sms: list[str] = []

        if any(h == hw.value for h in target_hw for hw in _BLACKWELL_HARDWARE):
            gencode_sms.append("sm_100a")

        if SupportedHardware.LOCAL.value in target_hw:
            local_sm = _get_local_sm()
            if local_sm:
                gencode_sms.append(local_sm)

        # Deduplicate preserving order
        seen: set[str] = set()
        unique = [s for s in gencode_sms if not (s in seen or seen.add(s))]
        if unique:
            compile_options["cuda_cflags"] = [
                _sm_to_gencode(sm) for sm in unique
            ] + cuda_cflags
            spec["compile_options"] = compile_options
            sol_dict["spec"] = spec

        return sol_dict

    def _write_sources(self) -> None:
        """Write solution source files to the staging directory."""
        for src in self.solution.sources:
            dest = self.output_dir / src.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.content)

    def compile(self) -> tuple[list[str], str]:
        """Stage compilation files and return (command, artifact_path).

        Writes build_ext.py, solution.json, and C++/CUDA source files to
        output_dir. Injects gencode flags for the target hardware.

        The CLI should run the command in output_dir.
        After success, the artifact (benchmark_kernel.so) will be at artifact_path.
        """
        assert self._is_cpp, (
            f"compile() only handles C++/CUDA solutions, "
            f"got languages={self.solution.spec.languages}"
        )

        sol_dict = json.loads(self.solution.model_dump_json())
        sol_dict = self._inject_gencode_flags(sol_dict)

        # Overwrite solution.json with injected gencode flags.
        (self.output_dir / "solution.json").write_text(json.dumps(sol_dict))
        (self.output_dir / "build_ext.py").write_text(
            (_TEMPLATES_DIR / "build_ext.py").read_text()
        )

        cmd = ["python", "build_ext.py"]
        artifact_path = str(self.output_dir / "benchmark_kernel.so")

        return cmd, artifact_path

    def execute(self) -> list[str]:
        """Stage execution files and return the command to run.

        Writes eval_driver.py, definition.json, workload.jsonl, solution.json
        to output_dir. For Python solutions, also writes source files. For C++
        solutions, expects benchmark_kernel.so to already exist in output_dir
        (produced by a prior compile() call).

        The CLI should run the command in output_dir.
        Trace JSON will be emitted on stdout (one JSON object per line).
        """
        if self._is_cpp:
            so_path = self.output_dir / "benchmark_kernel.so"
            if not so_path.exists():
                raise FileNotFoundError(
                    f"benchmark_kernel.so not found at {so_path} — "
                    "run compile() first for C++/CUDA solutions"
                )

        (self.output_dir / "eval_driver.py").write_text(
            (_TEMPLATES_DIR / "eval_driver.py").read_text()
        )

        return ["python", "eval_driver.py"]

    def convert_stdout_to_traces(self, stdout: str) -> list[Trace]:
        """Parse JSONL stdout from eval_driver.py into Trace objects.

        Each line starting with '{' is parsed as a Trace JSON object.
        Non-JSON lines (library noise redirected to stderr) are skipped.
        """
        traces = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                traces.append(Trace(**json.loads(line)))
        return traces
