# SPDX-License-Identifier: Apache-2.0
"""Tests for task packet construction. CPU only -- writes files, runs nothing."""

from __future__ import annotations

import json

import pytest

from solexbench_agents.task_packet import amd_workload_path, build_packet

REFERENCE = 'import torch\n\ndef run(x):\n    return x * 2\n'

DEFINITION = {
    "name": "069_rms_norm",
    "description": "doubles the input",
    "hf_id": "acme/model",
    "axes": {"n": {"type": "var", "description": "size"}},
    "inputs": {"x": {"shape": ["n"], "dtype": "bfloat16", "description": "in"}},
    "outputs": {"output": {"shape": ["n"], "dtype": "bfloat16", "description": "out"}},
    "reference": REFERENCE,
}

B200_WORKLOADS = [
    {"uuid": "u1", "axes": {"n": 8}, "inputs": {"x": {"type": "random"}},
     "tolerance": {"max_atol": 1e-9, "max_rtol": 1e-9}},
    {"uuid": "u2", "axes": {"n": 16}, "inputs": {"x": {"type": "random"}},
     "tolerance": {"max_atol": 1e-9, "max_rtol": 1e-9}},
]

AMD_WORKLOADS = [
    {**w, "tolerance": {"max_atol": 0.0078, "max_rtol": 0.0078,
                        "required_matched_ratio": 0.99,
                        "_provenance": "10 seeds x 1.25 margin"}}
    for w in B200_WORKLOADS
]


@pytest.fixture
def problem(tmp_path):
    d = tmp_path / "benchmark" / "L1" / "069_rms_norm"
    d.mkdir(parents=True)
    (d / "definition.json").write_text(json.dumps(DEFINITION))
    (d / "reference.py").write_text(REFERENCE)
    (d / "workload.jsonl").write_text(
        "\n".join(json.dumps(w) for w in B200_WORKLOADS) + "\n")
    return d


@pytest.fixture
def amd_root(tmp_path):
    d = tmp_path / "amd-workloads" / "L1" / "069_rms_norm"
    d.mkdir(parents=True)
    (d / "workload.jsonl").write_text(
        "\n".join(json.dumps(w) for w in AMD_WORKLOADS) + "\n")
    return tmp_path / "amd-workloads"


class TestWorkloadSelection:
    def test_prefers_amd_derived_tolerances(self, problem, amd_root):
        assert amd_workload_path(problem, amd_root) == \
            amd_root / "L1" / "069_rms_norm" / "workload.jsonl"

    def test_refuses_to_fall_back_silently(self, problem, tmp_path):
        """A missing AMD entry must raise, not quietly use B200 tolerances.

        Silent fallback would judge an AMD kernel by a tolerance measured on
        other silicon, for exactly the problems whose calibration is most
        interesting. Prime directive 2.
        """
        with pytest.raises(FileNotFoundError, match="no AMD-derived workloads"):
            amd_workload_path(problem, tmp_path / "does-not-exist")

    def test_explicit_none_uses_the_shipped_file(self, problem):
        assert amd_workload_path(problem, None) == problem / "workload.jsonl"


class TestBuildPacket:
    def test_contains_the_problem_and_the_tools(self, problem, amd_root, tmp_path):
        packet = tmp_path / "packet"
        manifest = build_packet(problem, packet, max_attempts=5, author="agent",
                                workloads_root=amd_root)
        for name in ("definition.json", "reference.py", "workload.jsonl",
                     "TASK.md", "verify", ".packet.json"):
            assert (packet / name).exists(), name
        assert manifest["n_workloads"] == 2
        assert manifest["tolerances"] == "amd-derived"
        assert manifest["problem"] == "L1/069_rms_norm"

    def test_carries_amd_tolerances_not_b200(self, problem, amd_root, tmp_path):
        packet = tmp_path / "packet"
        build_packet(problem, packet, max_attempts=5, author="a",
                     workloads_root=amd_root)
        first = json.loads((packet / "workload.jsonl").read_text().splitlines()[0])
        assert first["tolerance"]["max_atol"] == pytest.approx(0.0078)

    def test_leaks_no_derivations_or_other_problems(self, problem, amd_root, tmp_path):
        """The packet is the problem and nothing else.

        An agent that could read the tolerance derivation could tune to the
        tolerance instead of to the semantics.
        """
        packet = tmp_path / "packet"
        build_packet(problem, packet, max_attempts=5, author="a",
                     workloads_root=amd_root)
        present = {p.name for p in packet.iterdir()}
        assert present == {"definition.json", "reference.py", "workload.jsonl",
                           "TASK.md", "verify", ".packet.json"}

    def test_verify_is_executable(self, problem, amd_root, tmp_path):
        packet = tmp_path / "packet"
        build_packet(problem, packet, max_attempts=5, author="a",
                     workloads_root=amd_root)
        import os
        assert os.access(packet / "verify", os.X_OK)

    def test_reference_comes_from_the_definition_field(self, problem, amd_root,
                                                       tmp_path):
        """definition.json's `reference` is what the harness executes.

        The materializer writes the source to both places; if they ever drift,
        the field is the authoritative one and the agent should read that.
        """
        (problem / "reference.py").write_text("# a stale copy\n")
        packet = tmp_path / "packet"
        build_packet(problem, packet, max_attempts=5, author="a",
                     workloads_root=amd_root)
        assert (packet / "reference.py").read_text() == REFERENCE

    def test_task_md_states_the_budget_and_workload_count(self, problem, amd_root,
                                                          tmp_path):
        packet = tmp_path / "packet"
        build_packet(problem, packet, max_attempts=3, author="a",
                     workloads_root=amd_root)
        task = (packet / "TASK.md").read_text()
        assert "3 times" in task
        assert "2\nworkloads" in task or "2 workloads" in task
        assert "MI355X" in task
        # The menu must actually offer the exotic backends, or "I could not
        # express this" becomes a legitimate excuse for an unsolved problem.
        for lang in ("triton", "hip_cpp", "ck_tile", "hipblaslt", "aiter"):
            assert lang in task, lang
        assert "Gluon" in task
        assert "asm" in task

    def test_rebuild_clears_stale_state(self, problem, amd_root, tmp_path):
        packet = tmp_path / "packet"
        build_packet(problem, packet, max_attempts=5, author="a",
                     workloads_root=amd_root)
        (packet / "solution.json").write_text('{"stale": true}')
        (packet / ".attempts.json").write_text('{"used": 5}')
        build_packet(problem, packet, max_attempts=5, author="a",
                     workloads_root=amd_root)
        assert not (packet / "solution.json").exists()
        assert not (packet / ".attempts.json").exists()


def _make_category(root, category, names):
    d = root / "benchmark" / category
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = d / n
        p.mkdir()
        (p / "definition.json").write_text("{}")
    return root / "benchmark"


class TestDiscoverProblems:
    def test_even_sampling_spans_the_category(self, tmp_path):
        """Taking the first N would sample one model family.

        Problem numbering follows the source model, so L1/001..006 come from the
        same family and a pilot over them measures that family rather than the
        category.
        """
        from solexbench_agents.runner import discover_problems

        bench = _make_category(tmp_path, "L1",
                               [f"{i:03d}_problem" for i in range(1, 21)])
        picked, excluded = discover_problems(bench, ["L1"], limit_per_category=4)
        names = [p.name for p in picked]
        assert len(names) == 4
        assert names[0] == "001_problem"
        assert names[-1] != "004_problem", "should not be the first four"
        assert excluded == []

    def test_missing_category_is_loud(self, tmp_path):
        from solexbench_agents.runner import discover_problems

        (tmp_path / "benchmark").mkdir()
        with pytest.raises(FileNotFoundError, match="no such category"):
            discover_problems(tmp_path / "benchmark", ["L1"])

    def test_limit_above_available_returns_all(self, tmp_path):
        from solexbench_agents.runner import discover_problems

        bench = _make_category(tmp_path, "Quant", ["000_q", "001_q", "002_q"])
        picked, _ = discover_problems(bench, ["Quant"], 10)
        assert len(picked) == 3

    def test_deferred_problems_are_excluded_and_named(self, tmp_path):
        """Excluded, not silently dropped.

        The NVFP4 problems' own references fail on ROCm, so there is no solvable
        task; including them would depress every rate by a constant. But a rate
        over 220 is not a rate over 235, so the exclusion has to be reported.
        """
        from solexbench_agents.runner import discover_problems

        bench = _make_category(tmp_path, "Quant",
                               ["001_fp8_a", "008_nvfp4_b", "020_nvfp4_c"])
        picked, excluded = discover_problems(
            bench, ["Quant"],
            exclude={"Quant__008_nvfp4_b": "nvfp4-no-rocm-path",
                     "Quant__020_nvfp4_c": "nvfp4-no-rocm-path"},
        )
        assert [p.name for p in picked] == ["001_fp8_a"]
        assert excluded == ["Quant__008_nvfp4_b", "Quant__020_nvfp4_c"]

    def test_sampling_happens_after_exclusion(self, tmp_path):
        """Otherwise a pilot slot is spent on a problem that cannot be run."""
        from solexbench_agents.runner import discover_problems

        names = [f"{i:03d}_p" for i in range(1, 11)]
        bench = _make_category(tmp_path, "Quant", names)
        picked, excluded = discover_problems(
            bench, ["Quant"], limit_per_category=3,
            exclude={f"Quant__{n}": "deferred" for n in names[:5]},
        )
        assert len(picked) == 3
        assert all(p.name in names[5:] for p in picked)
        assert len(excluded) == 5


class TestLoadDeferred:
    def test_reads_reasons(self, tmp_path):
        from solexbench_agents.runner import load_deferred

        art = tmp_path / "artifacts"
        art.mkdir()
        (art / "deferred.json").write_text(json.dumps({
            "problems": {"Quant__008_x": {"reason": "nvfp4-no-rocm-path",
                                          "mechanism": "long text"}}
        }))
        assert load_deferred(tmp_path) == {"Quant__008_x": "nvfp4-no-rocm-path"}

    def test_absent_file_is_empty_not_an_error(self, tmp_path):
        from solexbench_agents.runner import load_deferred

        assert load_deferred(tmp_path) == {}

    def test_corrupt_file_does_not_crash_the_sweep(self, tmp_path):
        from solexbench_agents.runner import load_deferred

        art = tmp_path / "artifacts"
        art.mkdir()
        (art / "deferred.json").write_text("{truncated")
        assert load_deferred(tmp_path) == {}


class TestResumeSemantics:
    """A recorded failure is a result; an infrastructure failure is not."""

    def _sweep(self, tmp_path, retry_transient):
        from solexbench_agents.runner import Sweep, Unit

        bench = _make_category(tmp_path, "L1", ["001_p"])
        sweep = Sweep(
            run_root=tmp_path / "run",
            harness_specs={"claude-code": {}},
            gpus=[1],
            max_attempts=5,
            timeout_s=60,
            workloads_root=None,
            retry_transient=retry_transient,
        )
        return sweep, Unit(harness="claude-code", problem_dir=bench / "L1" / "001_p")

    def _write_session(self, sweep, unit, payload):
        out = unit.out_dir(sweep.run_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "session.json").write_text(json.dumps(payload))

    def test_missing_session_is_not_done(self, tmp_path):
        sweep, unit = self._sweep(tmp_path, retry_transient=False)
        assert not sweep.already_done(unit)

    def test_recorded_model_failure_counts_as_done(self, tmp_path):
        """Otherwise the sweep retries the same hard problem forever."""
        sweep, unit = self._sweep(tmp_path, retry_transient=False)
        self._write_session(sweep, unit, {"produced_solution": False,
                                          "transient_failure": False})
        assert sweep.already_done(unit)

    def test_transient_failure_is_kept_by_default(self, tmp_path):
        sweep, unit = self._sweep(tmp_path, retry_transient=False)
        self._write_session(sweep, unit, {"transient_failure": True})
        assert sweep.already_done(unit)

    def test_transient_failure_is_rerun_when_asked(self, tmp_path):
        sweep, unit = self._sweep(tmp_path, retry_transient=True)
        self._write_session(sweep, unit, {"transient_failure": True})
        assert not sweep.already_done(unit)

    def test_api_error_is_rerun_when_asked(self, tmp_path):
        sweep, unit = self._sweep(tmp_path, retry_transient=True)
        self._write_session(sweep, unit, {"harness_api_error": True})
        assert not sweep.already_done(unit)

    def test_real_result_is_never_rerun(self, tmp_path):
        sweep, unit = self._sweep(tmp_path, retry_transient=True)
        self._write_session(sweep, unit, {"produced_solution": True,
                                          "transient_failure": False})
        assert sweep.already_done(unit)

    def test_corrupt_session_is_rerun(self, tmp_path):
        sweep, unit = self._sweep(tmp_path, retry_transient=True)
        out = unit.out_dir(sweep.run_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "session.json").write_text("{truncated")
        assert not sweep.already_done(unit)


class TestVerifyRoot:
    def test_holds_the_harness_but_no_answer_key(self, tmp_path):
        """The reduced tree an agent's ./verify points at.

        It must carry the evaluation code and must NOT carry artifacts/ -- the
        tolerance derivations and analytic bounds are the answer key (D17).
        """
        from solexbench_agents.task_packet import REPO_ROOT, build_verify_root

        dest = build_verify_root(REPO_ROOT, tmp_path / "verify-root")
        assert (dest / "scripts" / "agent_verify.py").is_file()
        assert (dest / "scripts" / "runners" / "_common.py").is_file()
        assert (dest / "src" / "sol_execbench").is_dir()
        assert not (dest / "artifacts").exists()
        assert not (dest / "data" / "SOL-ExecBench").exists()

    def test_is_idempotent(self, tmp_path):
        from solexbench_agents.task_packet import REPO_ROOT, build_verify_root

        first = build_verify_root(REPO_ROOT, tmp_path / "vr")
        build_verify_root(REPO_ROOT, tmp_path / "vr")
        assert (first / "scripts" / "agent_verify.py").is_file()

    def test_packet_verify_points_at_the_reduced_tree(self, problem, amd_root,
                                                     tmp_path):
        from solexbench_agents.task_packet import build_packet

        verify_root = tmp_path / "vr"
        verify_root.mkdir()
        packet = tmp_path / "packet"
        manifest = build_packet(problem, packet, max_attempts=5, author="a",
                                workloads_root=amd_root, verify_root=verify_root)
        assert manifest["verify_root"] == str(verify_root)
        assert str(verify_root) in (packet / "verify").read_text()
