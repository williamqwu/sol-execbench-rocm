#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# NEW FILE, contributed from the AMDPilot v2 fleet side (issue amdpilotv2#19).

"""The four DSL rules, at the shapes that decide them.

`dsl_check` is label-only and can reject nothing, so what these tests protect is
not a gate — it is the *credibility* of the census that will decide whether a
gate is ever allowed. Two failure directions matter and they are not symmetric:

* a **false negative** understates how much Triton the fleet's agents wrote,
  which is a wrong number in a report;
* a **false positive** says a real Triton kernel is not Triton, and on the day
  anything scores on this label that becomes a correct kernel scored zero,
  indistinguishable downstream from a kernel that did not work.

So most of what is asserted here is false positives, and the shapes chosen are
the ones a first version of this rule (on the fleet side, over a 220-problem
sweep) actually got wrong: twelve packets, nine of them launching Triton through
`kernel.warmup(...)` or `CompiledKernel.run` rather than `k[grid](...)` — the
low-level path, chosen deliberately, because the subscript form costs about
20 us of Python. The submissions most likely to be misread are the ones written
by the strongest optimisers.

The module is loaded by path rather than imported through `sol_execbench`,
because the package `__init__` pulls in pydantic and torch and this detector
needs neither — a rule that only runs where a GPU stack is installed is a rule
that does not run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_SOURCE = ROOT / "src" / "sol_execbench" / "core" / "bench" / "dsl_check.py"

_spec = importlib.util.spec_from_file_location("dsl_check", _SOURCE)
dsl_check = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("dsl_check", dsl_check)
_spec.loader.exec_module(dsl_check)

CORPUS = ROOT / "artifacts" / "10" / "runs" / "full-01"
needs_corpus = pytest.mark.skipif(
    not CORPUS.is_dir(),
    reason=f"the archived packets are not in this checkout ({CORPUS}); the "
           f"two real cases cannot be replayed here and are SKIPPED rather "
           f"than quietly passing")


def labels(sources) -> list[str]:
    return dsl_check.dsl_labels(sources)["labels"]


# ── triton: present AND reached, through bindings and not spellings ───────────

def test_a_launched_jit_kernel_is_triton():
    assert labels({"kernel.py": (
        "import triton\n"
        "import triton.language as tl\n"
        "@triton.jit\n"
        "def k(x, BLOCK: tl.constexpr):\n"
        "    pass\n"
        "def run(x):\n"
        "    k[(1,)](x, BLOCK=64)\n"
        "    return x\n")}) == ["triton"]


@pytest.mark.parametrize("launch", [
    "k.warmup(x, grid=(1,))",
    "k.run(x, grid=(1,))",
    "_h = k[(1,)]\n    _h(x)",
    "from functools import partial\n    partial(k[(1,)], num_warps=1)(x)",
])
def test_the_low_level_launch_paths_are_launches(launch):
    """The nine-of-twelve shape. `k[grid](...)` costs ~20 us of Python and a
    submission optimising a small kernel will avoid it; a checker that only
    knows the subscript form calls the fastest submissions non-compliant."""
    assert labels({"kernel.py": (
        "import triton\n"
        "@triton.jit\n"
        "def k(x):\n"
        "    pass\n"
        "def run(x):\n"
        f"    {launch}\n"
        "    return x\n")}) == ["triton"]


def test_a_kernel_handed_to_torch_compile_still_counts():
    """`run = torch.compile(_run)` binds a real entry point that no
    name-to-name alias rule can see, and the Triton underneath still runs."""
    assert labels({"kernel.py": (
        "import torch, triton\n"
        "@triton.jit\n"
        "def k(x):\n"
        "    pass\n"
        "def _run(x):\n"
        "    k[(1,)](x)\n"
        "    return x\n"
        "run = torch.compile(_run)\n")}) == ["triton"]


def test_an_entry_point_defined_in_both_arms_of_an_if_is_found():
    seen = dsl_check.dsl_labels({"kernel.py": (
        "import triton\n"
        "_FAST = None\n"
        "@triton.jit\n"
        "def k(x):\n"
        "    pass\n"
        "if _FAST is not None:\n"
        "    def run(x):\n"
        "        k[(1,)](x)\n"
        "        return x\n"
        "else:\n"
        "    def run(x):\n"
        "        k[(1,)](x)\n"
        "        return x\n")})
    assert seen["entry_point_found"] is True
    assert seen["labels"] == ["triton"]


def test_a_torch_wrapper_that_merely_imports_triton_is_not_triton():
    """The headline false positive: presence is not authorship."""
    seen = dsl_check.dsl_labels({"kernel.py": (
        "import torch\n"
        "import triton\n"
        "def run(a, b):\n"
        "    return torch.matmul(a, b)\n")})
    assert seen["labels"] == []
    assert "no @triton.jit" in seen["rules"]["triton"]["why_not"]


def test_triton_language_alone_binds_no_jit():
    """`import triton.language as tl` binds the SUBMODULE. `tl.jit` is not a
    thing, and this is the real `L2__061` shape."""
    seen = dsl_check.dsl_labels({"kernel.py": (
        "import triton.language as tl\n"
        "def run(x):\n"
        "    return x\n")})
    assert seen["labels"] == []
    assert seen["rules"]["triton"]["imports_triton"] is True


def test_a_locally_defined_jit_decorator_does_not_count():
    """The rule reads bindings, never spellings — which is also the only reason
    it would survive contact with anything scoring on it."""
    assert labels({"kernel.py": (
        "def jit(f):\n"
        "    return f\n"
        "@jit\n"
        "def k(x):\n"
        "    pass\n"
        "def run(x):\n"
        "    k[(1,)](x)\n"
        "    return x\n")}) == []


def test_a_defined_but_never_launched_kernel_is_not_triton():
    seen = dsl_check.dsl_labels({"kernel.py": (
        "import torch, triton\n"
        "@triton.jit\n"
        "def k(x):\n"
        "    pass\n"
        "def run(a, b):\n"
        "    return torch.matmul(a, b)\n")})
    assert seen["labels"] == []
    assert seen["rules"]["triton"]["present"] is True
    assert seen["rules"]["triton"]["reached"] is False


def test_scratch_files_outside_the_import_graph_decide_nothing():
    """One archived submission staged twelve `exp*.py` files with live kernels
    beside a `kernel.py` that is pure torch and imports none of them."""
    seen = dsl_check.dsl_labels({
        "kernel.py": "import torch\ndef run(a, b):\n    return torch.matmul(a, b)\n",
        "exp1.py": ("import triton\n@triton.jit\ndef k(x):\n    pass\n"
                    "def go(x):\n    k[(1,)](x)\n")})
    assert seen["labels"] == []
    assert seen["rules"]["triton"]["jits_outside_the_import_graph"]
    assert seen["rules"]["triton"]["jits_defined"] == []


def test_a_sibling_module_is_part_of_the_submission():
    both = {
        "kernel.py": "from gn_triton import gn_silu\ndef run(x):\n    return gn_silu(x)\n",
        "gn_triton.py": ("import triton\n@triton.jit\ndef _k(x):\n    pass\n"
                         "def gn_silu(x):\n    _k[(1,)](x)\n    return x\n")}
    assert labels(both) == ["triton"]
    # The whole sibling-module argument in one assertion: the same submission
    # read as kernel.py alone labels differently, which is why a census taken
    # from a flat `kernels/*.py` directory reports a FLOOR and not a count.
    assert labels({"kernel.py": both["kernel.py"]}) == []


# ── aiter: called, not merely imported ────────────────────────────────────────

def test_aiter_called_from_the_entry_point_is_aiter():
    assert labels({"kernel.py": (
        "from aiter.ops.triton.moe import fused_moe\n"
        "def run(x):\n"
        "    return fused_moe(x)\n")}) == ["aiter"]


def test_aiter_imported_and_never_called_is_not_aiter():
    seen = dsl_check.dsl_labels({"kernel.py": (
        "import aiter\n"
        "import torch\n"
        "def run(a, b):\n"
        "    return torch.matmul(a, b)\n")})
    assert seen["labels"] == []
    assert seen["rules"]["aiter"]["present"] is True
    assert "leftover line" in seen["rules"]["aiter"]["why_not"]


# ── assembly: a constraint string and a mnemonic, or nothing ──────────────────

@pytest.mark.parametrize("body", [
    'asm volatile("");',
    '__asm__ __volatile__("" ::: "memory");',
    'asm("nop");',
])
def test_a_compiler_barrier_is_not_hand_written_assembly(body):
    """These appear in ordinary C++ that has nothing to do with hand assembly.
    Matching the word `asm` would label every submission containing one."""
    assert labels({"kernel.py": f'SRC = """{body}"""\ndef run(x):\n    return x\n'}) == []


def test_inline_asm_with_operands_and_a_gfx_mnemonic_is_assembly():
    assert labels({"kernel.py": (
        'SRC = """\n'
        'asm volatile("v_mfma_f32_16x16x16f16 %0, %1, %2, %0"\n'
        '             : "+v"(acc) : "v"(a), "v"(b));\n'
        '"""\n'
        "def run(x):\n"
        "    return x\n")}) == ["assembly"]


def test_an_assembly_source_file_is_assembly_even_though_nothing_builds_it():
    """`.s`/`.S` is written for a build path this repository does not have:
    `driver/templates/build_ext.py` collects `.cu .hip .cpp .cc .cxx .c` and
    nothing else. The rule says so, and this test pins the claim so the day the
    build path appears, somebody finds the sentence that has to change."""
    seen = dsl_check.dsl_labels({"kernel.py": "def run(x):\n    return x\n",
                                 "gemm.s": "v_mfma_f32_16x16x16f16 v0, v1, v2, v0\n"})
    assert seen["labels"] == ["assembly"]
    assert "assembly" in seen["unvalidated_rules"]


# ── flydsl: all four halves ───────────────────────────────────────────────────

def test_flydsl_needs_import_device_launcher_and_a_call_site():
    assert labels({"kernel.py": (
        "import flydsl as flyc\n"
        "@flyc.kernel\n"
        "def dev(x):\n"
        "    pass\n"
        "@flyc.jit\n"
        "def launcher(x):\n"
        "    return dev(x)\n"
        "def run(x):\n"
        "    h = flyc.compile(launcher)\n"
        "    return h.launch(x)\n")}) == ["flydsl"]


def test_importing_flydsl_alone_is_not_flydsl():
    seen = dsl_check.dsl_labels({"kernel.py":
                                 "import flydsl as flyc\ndef run(x):\n    return x\n"})
    assert seen["labels"] == []
    assert seen["rules"]["flydsl"]["present"] is True


def test_the_two_rules_with_no_corpus_say_so():
    """Written down in the result and not only in a docstring, so a reader of
    the census artifact cannot miss it."""
    seen = dsl_check.dsl_labels({"kernel.py": "def run(x):\n    return x\n"})
    assert seen["unvalidated_rules"] == ["flydsl", "assembly"]
    assert seen["rules"]["flydsl"]["validated"] is False
    assert seen["rules"]["assembly"]["validated"] is False
    assert seen["rules"]["triton"]["validated"] is True


# ── silence is never a finding ────────────────────────────────────────────────

def test_a_submission_this_reader_cannot_parse_is_reported_not_labelled():
    seen = dsl_check.dsl_labels({"kernel.py": "def run(:\n"})
    assert seen["labels"] == []
    assert seen["unparsed"], "a file that does not parse must be reported"


def test_a_non_python_submission_reads_as_unread_and_not_as_clean():
    """A `.hip` submission is staged, compiled and timed, and this reader does
    not parse it. `labels: []` alone would render that silence as a finding."""
    seen = dsl_check.dsl_labels({"kernel.hip": "int main(){ return 0; }"})
    assert seen["read"] == []
    assert seen["unread"] == ["kernel.hip"]


def test_reference_py_is_not_part_of_the_submission():
    """It is staged beside every submission and it is the problem's own code."""
    seen = dsl_check.dsl_labels({
        "kernel.py": "import torch\ndef run(a, b):\n    return torch.matmul(a, b)\n",
        "reference.py": "import triton\n@triton.jit\ndef k(x):\n    pass\n"})
    assert seen["labels"] == []
    assert "reference.py" not in seen["read"]


# ── the comparison, which is still not a verdict ──────────────────────────────

def test_an_undeclared_run_has_nothing_to_disagree_with():
    result = dsl_check.check_dsl_constraint(
        {"kernel.py": "def run(x):\n    return x\n"}, None)
    assert result["declared"] is None
    assert result["agrees"] is None, "absent is not a failure and not a pass"


def test_a_declared_dsl_that_is_not_visible_disagrees_and_rejects_nothing():
    result = dsl_check.check_dsl_constraint(
        {"kernel.py": "import torch\ndef run(a, b):\n    return torch.matmul(a, b)\n"},
        "triton")
    assert result["agrees"] is False
    assert "verdict" not in result and "outcome" not in result
    assert "rejects nothing" in result["note"]


def test_a_declared_dsl_with_no_rule_says_so_and_neither_passes_nor_fails():
    """The third outcome. `unruled` is not `unconstrained` and not `False`.

    Filtering an unknown value out of the declared list -- which is the obvious
    way to write this -- makes a run constrained to `cutlass` return
    `declared: None` and the note "no DSL was declared for this run", i.e. it
    reports a constrained run as an unconstrained one. That is the same defect
    as `_is_cpp` answering "this is Python" and "I have never heard of this
    language" with one `False`.
    """
    result = dsl_check.check_dsl_constraint(
        {"kernel.py": "import torch\ndef run(x):\n    return torch.relu(x)\n"},
        "cutlass")
    assert result["declared"] == ["cutlass"], "the declaration must survive"
    assert result["unruled"] == ["cutlass"]
    assert result["constrained"] is None
    assert result["per_declared"] == {"cutlass": None}
    assert result["agrees"] is None, "no rule is not a failure and not a pass"
    assert "no rule for" in result["note"]
    assert "rejects nothing" in result["note"]


def test_an_unruled_value_beside_a_ruled_one_is_not_silently_dropped():
    """A verdict over a subset must not be reported as a verdict over the whole.

    `triton` IS visible here, so answering `agrees: True` would be a pass this
    module did not earn -- it never checked `cutlass` at all.
    """
    source = ("import triton\nimport triton.language as tl\n"
              "@triton.jit\ndef k(x):\n    pass\n"
              "def run(x):\n    k[(1,)](x)\n    return x\n")
    result = dsl_check.check_dsl_constraint({"kernel.py": source},
                                            ["triton", "cutlass"])
    assert result["declared"] == ["triton", "cutlass"]
    assert result["unruled"] == ["cutlass"]
    assert result["per_declared"]["triton"] is True, "the ruled half is answered"
    assert result["per_declared"]["cutlass"] is None
    assert result["agrees"] is None, "the declaration as a whole was not checked"


def test_an_unruled_value_is_distinguishable_from_an_undeclared_run():
    """The two cases that a filter-and-drop implementation makes identical."""
    source = {"kernel.py": "def run(x):\n    return x\n"}
    undeclared = dsl_check.check_dsl_constraint(source, None)
    unruled = dsl_check.check_dsl_constraint(source, "cutlass")
    assert undeclared["agrees"] is unruled["agrees"] is None
    assert undeclared["declared"] != unruled["declared"]
    assert undeclared["unruled"] == [] and unruled["unruled"] == ["cutlass"]
    assert undeclared["note"] != unruled["note"]


def test_neither_entry_point_raises_on_any_shape():
    """The label-only promise, asserted rather than described. A screen that can
    raise is a screen that can stop a measurement."""
    for sources in ({}, None, {"kernel.py": "def run(:\n"},
                    {"kernel.hip": "int main(){}"},
                    [("kernel.py", "import triton\n")],
                    {"kernel.py": ""}):
        dsl_check.dsl_labels(sources)
        dsl_check.check_dsl_constraint(sources, "triton")


def test_the_three_source_shapes_normalise_the_same_way():
    """The same normaliser `static_source_screen` accepts, so the two screens at
    the one chokepoint never disagree about what the submission is."""
    text = "import triton\n@triton.jit\ndef k(x):\n    pass\ndef run(x):\n    k[(1,)](x)\n"

    class Src:
        def __init__(self, path, content):
            self.path, self.content = path, content

    assert labels({"kernel.py": text}) == ["triton"]
    assert labels([("kernel.py", text)]) == ["triton"]
    assert labels([Src("kernel.py", text)]) == ["triton"]


# ── the two real cases named in the issue ─────────────────────────────────────

def _packet(harness: str, problem: str) -> dict[str, str]:
    packet = CORPUS / harness / problem / "packet"
    return {str(p.relative_to(packet)): p.read_text(errors="replace")
            for p in sorted(packet.rglob("*")) if p.is_file()}


@needs_corpus
def test_L2__023_labels_as_triton_only_when_its_sibling_module_is_present():
    problem = "L2__023_video_latent_vae_encoder_downsampling"
    if not (CORPUS / "claude-code" / problem / "packet").is_dir():
        pytest.skip(f"{problem} is not in this checkout's claude-code packets")
    whole = _packet("claude-code", problem)
    assert "gn_triton.py" in whole, "the packet's shape has changed"
    assert labels(whole) == ["triton"]
    assert labels({"kernel.py": whole["kernel.py"]}) != ["triton"]


@needs_corpus
def test_L2__061_fails_triton_and_passes_aiter():
    """Imports `triton.language`, dispatches into `aiter.ops.triton.moe.*`, and
    defines zero `@triton.jit`. Authorship and dispatch are different axes and
    this submission is the case that proves it."""
    problem = "L2__061_sparse_moe_routing_and_experts"
    for harness in ("codex", "claude-code"):
        if (CORPUS / harness / problem / "packet").is_dir():
            break
    else:
        pytest.skip(f"{problem} is not in this checkout's packets")
    seen = dsl_check.dsl_labels(_packet(harness, problem))
    assert "triton" not in seen["labels"]
    assert "aiter" in seen["labels"]


@needs_corpus
def test_the_census_of_a_real_run_is_not_vacuous():
    """Anti-vacuity: a rule that labels nothing passes every test above that
    asserts a negative. At least one archived packet must label as Triton."""
    harness = next((h for h in ("codex", "claude-code")
                    if (CORPUS / h).is_dir()), None)
    if harness is None:
        pytest.skip("no archived packets in this checkout")
    problems = [p for p in sorted((CORPUS / harness).iterdir())
                if (p / "packet").is_dir()][:40]
    if not problems:
        pytest.skip("no packets to read")
    found = sum(1 for p in problems if "triton" in labels(_packet(harness, p.name)))
    assert found > 0, (
        f"none of {len(problems)} archived packets labelled as Triton. Either "
        f"the corpus changed shape or the rule stopped resolving; both are "
        f"reasons to look, and neither is a clean run.")
