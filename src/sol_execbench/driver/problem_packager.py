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
from ..core.bench.dsl_check import dsl_labels
from ..core.bench.reward_hack import check_static_source_screen

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

# AMD: the Python-hosted half, named so that `_is_cpp` returning False is a
# decision about a known language and not the absence of one. `flydsl` is
# imported and launched from `.py` like `cute_dsl`; `assembly` on its own is an
# ISA blob assembled and loaded from `.py` (the schema holds it to that entry
# point), while inline asm inside a compiled source declares that source's
# language too and so lands in the set above. Kept in step with the same pair
# of sets in `templates/eval_driver.py`, which the driver reads on the GPU
# side -- the duplication is deliberate, that template runs standalone.
_PYTHON_HOSTED_LANGUAGES = {
    SupportedLanguages.PYTORCH,
    SupportedLanguages.TRITON,
    SupportedLanguages.CUTE_DSL,
    SupportedLanguages.CUTILE,
    SupportedLanguages.CUDNN_FRONTEND,
    SupportedLanguages.AITER,
    SupportedLanguages.FLYDSL,
    SupportedLanguages.ASSEMBLY,
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

        # AMD: static source screen, before anything is written, compiled or
        # imported. The runtime guards live in Python and a compiled HIP
        # submission runs underneath them -- `install_smi_guard` cannot see a
        # `system("rocm-smi …")` inside an extension, and the stream check
        # cannot see a stream created and consumed inside a kernel launch.
        # Screening here rather than in the driver means it covers every entry
        # point (CLI, sweeps, submission intake) with one call.
        check_static_source_screen(solution.sources)

        # AMD (amdpilotv2#19): the DSL rules, immediately beside the screen
        # above so the two readings of "what this submission is" happen at one
        # chokepoint, on one object, before anything is written or compiled.
        #
        # RECORDING ONLY, and this is not a phase to be tidied away later:
        # `dsl_labels` cannot raise and nothing may reject a submission on it
        # until the census over the archived sweeps
        # (`scripts/dsl_census.py`) has been read by a human. The line above
        # rejects; this one describes. Wrapped anyway, because a *packager* that
        # dies inside a new descriptive read would turn a label into an outage.
        try:
            self.dsl_check = dsl_labels(solution.sources)
        except Exception as exc:                              # noqa: BLE001
            self.dsl_check = {"error": f"{type(exc).__name__}: {exc}"}

        # Write problem files to staging directory up front.
        (self.output_dir / "definition.json").write_text(definition.model_dump_json())
        (self.output_dir / "workload.jsonl").write_text(
            "\n".join(w.model_dump_json() for w in workloads)
        )
        (self.output_dir / "solution.json").write_text(solution.model_dump_json())
        (self.output_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(config))
        )
        # Beside solution.json rather than inside it: the DSL label is an
        # observation ABOUT the submission made by this repository, not a field
        # the submission declared, and putting it in the same document would
        # make those two indistinguishable to every later reader.
        (self.output_dir / "dsl_check.json").write_text(
            json.dumps(self.dsl_check, indent=1)
        )
        self._write_sources()

    def __del__(self):
        if not self.keep_output_dir:
            shutil.rmtree(self.output_dir, ignore_errors=True)

    @property
    def _is_cpp(self) -> bool:
        # AMD: refuse a language neither set knows before answering. False here
        # means "no compile step", and a language that fell through to it would
        # be staged, skipped past the compiler and run as Python -- a failure
        # that surfaces much later as a missing module on a submission that was
        # never built.
        unclassified = [
            lang
            for lang in self.solution.spec.languages
            if lang not in _CPP_LANGUAGES and lang not in _PYTHON_HOSTED_LANGUAGES
        ]
        if unclassified:
            raise ValueError(
                f"ProblemPackager has no rule for language(s) "
                f"{[lang.value for lang in unclassified]}: they are in neither "
                f"_CPP_LANGUAGES nor _PYTHON_HOSTED_LANGUAGES."
            )
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
            # AMD: `CompileOptions.cuda_cflags` defaults to
            # `["-O3", "--use_fast_math"]`. `--use_fast_math` is nvcc-only;
            # hipcc hands the flag to clang++, which rejects it outright:
            #
            #   clang++: error: unknown argument: '--use_fast_math'
            #
            # This is the `-lcuda` defect (below) in a second form, and it
            # bites the same way: the default only materializes once
            # `compile_options` exists, which is true for EVERY submission
            # that sets any build flag at all -- so a submission that asked
            # for `-lhipblaslt` could not compile, for a reason that has
            # nothing to do with what it asked for. `-ffast-math` is clang's
            # spelling of the same intent, so the field keeps meaning upstream
            # what it means here.
            cuda_cflags = [
                "-ffast-math" if f == "--use_fast_math" else f for f in cuda_cflags
            ]
            if unique_gfx:
                compile_options["cuda_cflags"] = [
                    _gfx_to_offload_arch(g) for g in unique_gfx
                ] + cuda_cflags
            elif cuda_cflags:
                compile_options["cuda_cflags"] = cuda_cflags
            # AMD: `CompileOptions.ld_flags` defaults to `["-lcuda"]`, which
            # does not exist on ROCm. The default only bites once
            # compile_options is materialized at all -- which the block above
            # does, for every LOCAL submission -- so before this, EVERY C++
            # submission on ROCm failed at the link step with
            # "/usr/bin/ld: cannot find -lcuda", on a kernel that had compiled
            # cleanly. torch adds -lamdhip64 itself; naming it here keeps the
            # field meaning the same thing it means upstream.
            #
            # It is a SUBSTITUTION, not a fill-in-if-empty. `-lcuda` is a
            # pydantic *default*, so it is present in the dumped dict whenever
            # the submission set any other build flag at all -- and an
            # emptiness test then skips exactly the submissions that asked for
            # a library. Observed on the `ck` seed, which set only
            # `cuda_cflags` and still died at the link step on `-lcuda` after
            # compiling a full CK GEMM instance cleanly.
            if compile_options:
                ld_flags = [
                    "-lamdhip64" if f == "-lcuda" else f
                    for f in (compile_options.get("ld_flags") or [])
                ]
                compile_options["ld_flags"] = ld_flags or ["-lamdhip64"]
            if compile_options:
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
