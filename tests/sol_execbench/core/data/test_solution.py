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


"""Tests for sol_execbench.core.data.solution — language and entry point validation."""

import pytest
from pydantic import ValidationError

from sol_execbench.core.data.solution import BuildSpec, SupportedLanguages


def _make_spec(**overrides):
    base = dict(
        languages=["triton"],
        target_hardware=["LOCAL"],
        entry_point="kernel.py::run",
    )
    base.update(overrides)
    return BuildSpec(**base)


# ── _validate_languages — mixing ─────────────────────────────────────────────


class TestLanguageMixingValidation:
    """BuildSpec must reject specs that mix C++ and Python languages."""

    # -- pure Python sets (should all pass) --

    @pytest.mark.parametrize(
        "langs",
        [
            ["pytorch"],
            ["triton"],
            ["cute_dsl"],
            ["cutile"],
            ["cudnn_frontend"],
            ["pytorch", "triton"],
            ["pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend"],
        ],
    )
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    # -- pure C++ sets (should all pass) --

    @pytest.mark.parametrize(
        "langs",
        [
            ["cuda_cpp"],
            ["cutlass"],
            ["cudnn"],
            ["cublas"],
            ["cuda_cpp", "cutlass"],
            ["cuda_cpp", "cutlass", "cudnn", "cublas"],
        ],
    )
    def test_pure_cpp_languages_accepted(self, langs):
        spec = _make_spec(languages=langs, entry_point="kernel.cu::run")
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    # -- mixed sets (should all fail) --

    @pytest.mark.parametrize(
        "langs",
        [
            ["pytorch", "cuda_cpp"],
            ["triton", "cutlass"],
            ["cute_dsl", "cudnn"],
            ["cutile", "cublas"],
            ["cudnn_frontend", "cuda_cpp"],
            ["pytorch", "triton", "cuda_cpp"],
            ["cuda_cpp", "cutlass", "pytorch"],
        ],
    )
    def test_mixed_languages_rejected(self, langs):
        with pytest.raises(ValidationError, match="C\\+\\+ and Python cannot be mixed"):
            _make_spec(languages=langs, entry_point="kernel.cu::run")

    # -- single language (every enum value should pass alone) --

    @pytest.mark.parametrize("lang", [lg.value for lg in SupportedLanguages])
    def test_every_language_accepted_alone(self, lang):
        # AMD: hip_cpp/ck/ck_tile/hipblaslt/miopen are the ROCm C++ languages
        # and need a C++ entry point, exactly as their CUDA counterparts do.
        # aiter is a Python package, so it stays on the .py side, as do flydsl
        # (Python-hosted DSL) and assembly (held to .py when declared alone).
        cpp_langs = (
            "cuda_cpp", "cutlass", "cudnn", "cublas",
            "hip_cpp", "ck", "ck_tile", "hipblaslt", "miopen",
        )
        ext = ".cu" if lang in cpp_langs else ".py"
        spec = _make_spec(languages=[lang], entry_point=f"kernel{ext}::run")
        assert spec.languages == [SupportedLanguages(lang)]


# ── _validate_languages — entry point suffix ─────────────────────────────────


class TestEntryPointSuffixValidation:
    """BuildSpec must reject entry points whose suffix doesn't match the language category."""

    # -- Python languages with wrong suffix --

    @pytest.mark.parametrize(
        "lang", ["pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend"]
    )
    def test_python_language_rejects_cu_entry(self, lang):
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=[lang], entry_point="kernel.cu::run")

    @pytest.mark.parametrize(
        "lang", ["pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend"]
    )
    def test_python_language_rejects_cpp_entry(self, lang):
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=[lang], entry_point="kernel.cpp::run")

    # -- C++ languages with wrong suffix --

    @pytest.mark.parametrize("lang", ["cuda_cpp", "cutlass", "cudnn", "cublas"])
    def test_cpp_language_rejects_py_entry(self, lang):
        with pytest.raises(ValidationError, match="require a C\\+\\+/CUDA entry point"):
            _make_spec(languages=[lang], entry_point="kernel.py::run")

    # -- C++ languages with valid suffixes --

    @pytest.mark.parametrize(
        "suffix", [".cu", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cuh"]
    )
    def test_cpp_language_accepts_valid_suffixes(self, suffix):
        spec = _make_spec(languages=["cuda_cpp"], entry_point=f"kernel{suffix}::run")
        assert spec.languages == [SupportedLanguages.CUDA_CPP]

    # -- Python languages with valid suffix --

    @pytest.mark.parametrize(
        "lang", ["pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend"]
    )
    def test_python_language_accepts_py_entry(self, lang):
        spec = _make_spec(languages=[lang], entry_point="kernel.py::run")
        assert spec.languages == [SupportedLanguages(lang)]

    # -- AMD additions --

    def test_hip_entry_point_accepted(self):
        """`.hip` is a first-class entry point, not something hipify produces.

        A submission written natively in HIP must not be forced through a
        `.cu` rename: hipify mangles inline asm and wavefront-width
        assumptions, so the renamed file is not the same program.
        """
        spec = _make_spec(languages=["hip_cpp"], entry_point="kernel.hip::run")
        assert spec.languages == [SupportedLanguages.HIP_CPP]

    @pytest.mark.parametrize("lang", ["hip_cpp", "ck", "ck_tile", "hipblaslt", "miopen"])
    def test_amd_cpp_language_rejects_py_entry(self, lang):
        with pytest.raises(ValidationError, match="require a C\\+\\+/CUDA entry point"):
            _make_spec(languages=[lang], entry_point="kernel.py::run")

    def test_aiter_is_a_python_language(self):
        """AITER is imported from Python, so it must not be in the C++ set."""
        spec = _make_spec(languages=["aiter"], entry_point="kernel.py::run")
        assert spec.languages == [SupportedLanguages.AITER]
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=["aiter"], entry_point="kernel.cu::run")

    @pytest.mark.parametrize("lang", ["hip_cpp", "ck", "hipblaslt"])
    def test_amd_cpp_cannot_mix_with_python(self, lang):
        with pytest.raises(ValidationError, match="C\\+\\+ and Python cannot be mixed"):
            _make_spec(languages=["pytorch", lang], entry_point="kernel.cu::run")


# ── the DSL-axis additions: flydsl and assembly ──────────────────────────────


class TestDSLAxisLanguages:
    """`flydsl` and `assembly` complete the axis the fleet ablates on.

    `core/bench/dsl_check.py::DSLS` is `triton, aiter, flydsl, assembly`. The
    first two were already members of `SupportedLanguages`; until the other two
    were, an arm constrained to one of them sent `--language flydsl` into
    `scripts/agent_eval.py::build_solution` and lost EVERY evaluation to schema
    validation -- a failure that reads downstream as the model being unable to
    write the DSL rather than as a value this enum did not accept.
    """

    @pytest.mark.parametrize("lang", ["flydsl", "assembly"])
    def test_axis_language_is_a_member(self, lang):
        assert SupportedLanguages(lang).value == lang

    def test_flydsl_is_a_python_language(self):
        """FlyDSL is imported and launched from `.py`, exactly like cute_dsl."""
        spec = _make_spec(languages=["flydsl"], entry_point="kernel.py::run")
        assert spec.languages == [SupportedLanguages.FLYDSL]
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=["flydsl"], entry_point="kernel.cu::run")

    def test_flydsl_cannot_mix_with_cpp(self):
        with pytest.raises(ValidationError, match="C\\+\\+ and Python cannot be mixed"):
            _make_spec(languages=["flydsl", "hip_cpp"], entry_point="kernel.cu::run")

    def test_assembly_alone_takes_a_py_entry_point(self):
        """Declared on its own it is the run-time-assembled form, hosted by Python."""
        spec = _make_spec(languages=["assembly"], entry_point="kernel.py::run")
        assert spec.languages == [SupportedLanguages.ASSEMBLY]

    @pytest.mark.parametrize("suffix", [".s", ".S", ".cu", ".hip", ".cpp"])
    def test_assembly_alone_rejects_a_non_python_entry_point(self, suffix):
        """The refusal names the language and says what to declare instead.

        `.s`/`.S` matter most: `driver/templates/build_ext.py` collects
        `.cu .hip .cpp .cc .cxx .c` and no assembler source, so a standalone
        assembly file has nowhere to be assembled. Accepting it here would
        stage a submission that then fails at "No CUDA/C++ source files found",
        which names neither assembly nor the missing build path.
        """
        with pytest.raises(ValidationError, match="'assembly' on its own requires a"):
            _make_spec(languages=["assembly"], entry_point=f"kernel{suffix}::run")

    def test_assembly_may_accompany_a_cpp_language(self):
        """Inline `asm volatile` in a compiled HIP source is the validated form."""
        spec = _make_spec(
            languages=["assembly", "hip_cpp"], entry_point="kernel.hip::run"
        )
        assert set(spec.languages) == {
            SupportedLanguages.ASSEMBLY,
            SupportedLanguages.HIP_CPP,
        }

    def test_assembly_may_accompany_a_python_language(self):
        """torch for the glue, an assembled blob for the kernel."""
        spec = _make_spec(
            languages=["assembly", "pytorch"], entry_point="kernel.py::run"
        )
        assert set(spec.languages) == {
            SupportedLanguages.ASSEMBLY,
            SupportedLanguages.PYTORCH,
        }

    def test_assembly_does_not_defeat_its_host_s_suffix_rule(self):
        """Pairing with a host does not turn the suffix check off."""
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=["assembly", "pytorch"], entry_point="kernel.cu::run")
        with pytest.raises(ValidationError, match="require a C\\+\\+/CUDA entry point"):
            _make_spec(languages=["assembly", "hip_cpp"], entry_point="kernel.py::run")

    def test_assembly_does_not_defeat_the_mixing_rule(self):
        """It is excluded from the rule, not an exemption from it for others."""
        with pytest.raises(ValidationError, match="C\\+\\+ and Python cannot be mixed"):
            _make_spec(
                languages=["assembly", "pytorch", "hip_cpp"],
                entry_point="kernel.cu::run",
            )

    def test_every_member_is_classified(self):
        """No member may reach the suffix check with no category.

        A value in none of the three lists silently skips both suffix rules,
        which is how a language ends up nominally supported while no code path
        knows what to do with it. Asserted over the enum so a future addition
        fails here rather than in a sweep.
        """
        for lang in SupportedLanguages:
            cpp = lang.value in (
                "cuda_cpp", "cutlass", "cudnn", "cublas",
                "hip_cpp", "ck", "ck_tile", "hipblaslt", "miopen",
            )
            spec = _make_spec(
                languages=[lang.value],
                entry_point="kernel.cu::run" if cpp else "kernel.py::run",
            )
            assert spec.languages == [lang]
