# SPDX-License-Identifier: Apache-2.0
"""Tests for harness command construction and usage parsing.

Usage parsing gets its own tests because it is the one place a silent bug is
expensive: it would misreport cost across every session in a sweep, and a wrong
number in a currency column is indistinguishable from a right one. The fixtures
below are the real output shapes, captured from `claude -p --output-format json`
and `codex exec --json` on this node.
"""

from __future__ import annotations

import json

from solexbench_agents.harnesses import (
    AgentSession,
    ClaudeCodeHarness,
    CodexHarness,
)

# Trimmed from a real `claude -p --output-format json` result object.
CLAUDE_RESULT = json.dumps({
    "is_error": False,
    "num_turns": 14,
    "stop_reason": "end_turn",
    "session_id": "25f1fdfd-2621-4f18-b836-50d117dc5d5b",
    "total_cost_usd": 1.874321,
    "usage": {
        "input_tokens": 285123,
        "cache_creation_input_tokens": 40000,
        "cache_read_input_tokens": 190000,
        "output_tokens": 12044,
    },
    "modelUsage": {"claude-opus-5": {"inputTokens": 285123}},
    "subtype": "success",
    "result": "done",
    "type": "result",
})

# Real `codex exec --json` stream: JSON lines, usage on turn.completed.
CODEX_STREAM = "\n".join([
    '{"type":"thread.started","thread_id":"019fcb9b-48d3-7f81-8959-1f70bb5eb0e8"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"ok"}}',
    '{"type":"turn.completed","usage":{"input_tokens":11992,"cached_input_tokens":500,'
    '"cache_write_input_tokens":0,"output_tokens":27,"reasoning_output_tokens":18}}',
])


def _session() -> AgentSession:
    return AgentSession(harness="t", model=None, problem="L1__x",
                        packet_dir="/tmp/x", gpu=3)


class TestClaudeParsing:
    def test_reads_cost_tokens_and_turns(self):
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage(CLAUDE_RESULT, s)
        assert s.cost_usd == 1.874321
        assert s.input_tokens == 285123
        assert s.output_tokens == 12044
        assert s.cached_input_tokens == 190000
        assert s.num_turns == 14
        assert s.error is None

    def test_infers_model_when_not_pinned(self):
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage(CLAUDE_RESULT, s)
        assert s.model == "claude-opus-5"

    def test_explicit_model_is_not_overwritten(self):
        s = AgentSession(harness="t", model="pinned-model", problem="p",
                         packet_dir="/tmp", gpu=3)
        ClaudeCodeHarness(timeout_s=60, model="pinned-model").parse_usage(
            CLAUDE_RESULT, s)
        assert s.model == "pinned-model"

    def test_tolerates_leading_noise(self):
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage(
            "warning: something\n" + CLAUDE_RESULT, s)
        assert s.cost_usd == 1.874321

    def test_missing_result_object_is_an_error_not_a_zero(self):
        """Silence must not read as a free session."""
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage("no json here", s)
        assert s.cost_usd is None
        assert s.error and "no JSON result object" in s.error

    def test_reported_error_prefers_the_message_over_the_subtype(self):
        """`subtype` can say "success" on a failed run.

        A real gateway 403 arrived as is_error=true with subtype="success", so
        reporting the subtype described a clean finish that produced nothing.
        """
        blob = json.loads(CLAUDE_RESULT)
        blob["is_error"] = True
        blob["subtype"] = "success"
        blob["result"] = ("Failed to authenticate. API Error: 403 AMD gateway "
                          "error: Access denied due to Virtual Network rules.")
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage(json.dumps(blob), s)
        assert s.error and "403" in s.error
        assert "success" not in s.error


class TestTransientDetection:
    def test_gateway_403_is_transient(self):
        """Infrastructure, not the model. Counting it as a failure to solve
        would understate the score by however often the gateway wobbled."""
        s = _session()
        s.error = ("claude reported error: Failed to authenticate. API Error: 403 "
                   "AMD gateway error: Access denied due to Virtual Network rules.")
        assert ClaudeCodeHarness.is_transient_failure(s, "", "")

    def test_rate_limit_is_transient(self):
        s = _session()
        assert ClaudeCodeHarness.is_transient_failure(s, "", "HTTP 429 rate limit")

    def test_a_solution_is_never_transient(self):
        """If the agent delivered, whatever else happened is not worth retrying
        -- and a retry would discard a real answer."""
        s = _session()
        s.produced_solution = True
        s.error = "502 bad gateway"
        assert not ClaudeCodeHarness.is_transient_failure(s, "", "")

    def test_a_wrong_kernel_is_not_transient(self):
        s = _session()
        s.error = "claude reported error: could not make the kernel pass"
        assert not ClaudeCodeHarness.is_transient_failure(s, "", "")


class TestCodexParsing:
    def test_reads_tokens_and_turns(self):
        s = _session()
        CodexHarness(timeout_s=60).parse_usage(CODEX_STREAM, s)
        assert s.input_tokens == 11992
        assert s.output_tokens == 27
        assert s.cached_input_tokens == 500
        assert s.reasoning_tokens == 18
        assert s.num_turns == 1

    def test_cost_stays_null_rather_than_guessed(self):
        """This proxy publishes no prices; a guessed rate would look measured."""
        s = _session()
        CodexHarness(timeout_s=60).parse_usage(CODEX_STREAM, s)
        assert s.cost_usd is None

    def test_accumulates_across_turns(self):
        stream = CODEX_STREAM + "\n" + json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 1000, "output_tokens": 50,
                      "cached_input_tokens": 10, "reasoning_output_tokens": 5},
        })
        s = _session()
        CodexHarness(timeout_s=60).parse_usage(stream, s)
        assert s.input_tokens == 11992 + 1000
        assert s.output_tokens == 27 + 50
        assert s.num_turns == 2

    def test_failed_turn_is_recorded(self):
        stream = CODEX_STREAM + "\n" + json.dumps({
            "type": "turn.failed", "error": {"message": "upstream 502"},
        })
        s = _session()
        CodexHarness(timeout_s=60).parse_usage(stream, s)
        assert s.error and "upstream 502" in s.error

    def test_no_usage_event_is_an_error(self):
        s = _session()
        CodexHarness(timeout_s=60).parse_usage(
            '{"type":"thread.started","thread_id":"x"}', s)
        assert s.error and "no turn.completed" in s.error

    def test_malformed_lines_are_skipped_not_fatal(self):
        s = _session()
        CodexHarness(timeout_s=60).parse_usage(
            "{not json\n" + CODEX_STREAM + "\ntrailing garbage", s)
        assert s.input_tokens == 11992


class TestCommands:
    def test_claude_runs_without_asking_permission(self, tmp_path):
        """A sweep has no interactive channel; an unapproved tool call hangs.

        `--permission-mode bypassPermissions` rather than
        `--dangerously-skip-permissions`, which refuses outright under root and
        exits before any model call.
        """
        cmd = ClaudeCodeHarness(timeout_s=60).command(tmp_path, "go")
        assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
        assert "--dangerously-skip-permissions" not in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"

    def test_claude_declares_itself_sandboxed(self, tmp_path, monkeypatch):
        """The other half of running as root: without this, the permission
        bypass is refused regardless of the mode flag."""
        monkeypatch.setenv("SOLEXBENCH_SCRATCH", str(tmp_path))
        assert ClaudeCodeHarness(timeout_s=60).env(1)["IS_SANDBOX"] == "1"

    def test_codex_skips_the_git_repo_check(self, tmp_path):
        """Packets are not git repos and Codex otherwise refuses to start."""
        cmd = CodexHarness(timeout_s=60).command(tmp_path, "go")
        assert "--skip-git-repo-check" in cmd
        assert "--json" in cmd
        assert cmd[cmd.index("-C") + 1] == str(tmp_path)

    def test_model_is_passed_when_pinned(self, tmp_path):
        assert "--model" in ClaudeCodeHarness(timeout_s=60, model="m").command(
            tmp_path, "go")
        assert "--model" in CodexHarness(timeout_s=60, model="m").command(
            tmp_path, "go")


class TestEnvIsolation:
    def test_agent_sees_exactly_one_gpu(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOLEXBENCH_SCRATCH", str(tmp_path))
        env = ClaudeCodeHarness(timeout_s=60).env(3)
        assert env["HIP_VISIBLE_DEVICES"] == "3"
        assert env["CUDA_VISIBLE_DEVICES"] == "3"

    def test_compile_caches_are_per_gpu(self, tmp_path, monkeypatch):
        """A shared Triton cache across concurrent agents serves one agent's
        stale kernel to another, and turns a cache hit into a fast compile."""
        monkeypatch.setenv("SOLEXBENCH_SCRATCH", str(tmp_path))
        h = ClaudeCodeHarness(timeout_s=60)
        assert h.env(1)["TRITON_CACHE_DIR"] != h.env(2)["TRITON_CACHE_DIR"]
        assert h.env(1)["MIOPEN_USER_DB_PATH"] != h.env(2)["MIOPEN_USER_DB_PATH"]


class TestHarvest:
    def test_harvest_reads_attempts_and_solution(self, tmp_path):
        (tmp_path / "solution.json").write_text("{}")
        (tmp_path / ".attempts.json").write_text(json.dumps({
            "used": 3,
            "log": [{"passed": 2, "workloads": 16}, {"passed": 14, "workloads": 16}],
        }))
        s = _session()
        ClaudeCodeHarness._harvest(tmp_path, s)
        assert s.produced_solution
        assert s.verify_attempts == 3
        assert s.verify_best_passed == 14
        assert s.verify_workloads == 16

    def test_harvest_without_any_artifacts(self, tmp_path):
        s = _session()
        ClaudeCodeHarness._harvest(tmp_path, s)
        assert not s.produced_solution
        assert s.verify_attempts == 0
        assert s.verify_best_passed is None

    def test_harvest_survives_corrupt_attempts_file(self, tmp_path):
        (tmp_path / ".attempts.json").write_text("{truncated")
        s = _session()
        ClaudeCodeHarness._harvest(tmp_path, s)
        assert s.verify_attempts == 0


class TestApiErrorTerminalReason:
    def test_terminal_reason_api_error_marks_transient(self):
        """The reliable signal. The *message* varies -- "403 gateway error",
        "Stream idle timeout - no chunks received" -- in ways no signature list
        keeps up with, but the terminal state is structured."""
        blob = json.loads(CLAUDE_RESULT)
        blob["is_error"] = True
        blob["subtype"] = "success"
        blob["terminal_reason"] = "api_error"
        blob["result"] = "API Error: Stream idle timeout - no chunks received"
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage(json.dumps(blob), s)
        assert s.harness_api_error
        assert ClaudeCodeHarness.is_transient_failure(s, "", "")

    def test_clean_completion_is_not_an_api_error(self):
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage(CLAUDE_RESULT, s)
        assert not s.harness_api_error

    def test_api_error_with_a_solution_is_still_kept(self):
        """A retry would discard a real answer."""
        blob = json.loads(CLAUDE_RESULT)
        blob["terminal_reason"] = "api_error"
        s = _session()
        ClaudeCodeHarness(timeout_s=60).parse_usage(json.dumps(blob), s)
        s.produced_solution = True
        assert not ClaudeCodeHarness.is_transient_failure(s, "", "")


class TestOrphanReaping:
    """An agent-spawned process that outlives its session holds a GPU and
    inflates every later measurement on it, silently. See D20."""

    def test_reaps_a_process_whose_cwd_is_the_packet(self, tmp_path):
        import subprocess
        import time

        from solexbench_agents.harnesses import _reap_orphans

        packet = tmp_path / "packet"
        packet.mkdir()
        # Double-fork via setsid so it is NOT in our process group -- the case
        # process-group killing alone would miss.
        proc = subprocess.Popen(["sleep", "300"], cwd=str(packet),
                                start_new_session=True)
        time.sleep(0.3)
        killed = _reap_orphans(packet)
        assert any(str(proc.pid) in k for k in killed), killed
        proc.wait(timeout=10)

    def test_leaves_processes_outside_the_packet_alone(self, tmp_path):
        import subprocess
        import time

        from solexbench_agents.harnesses import _reap_orphans

        packet = tmp_path / "packet"
        packet.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        proc = subprocess.Popen(["sleep", "5"], cwd=str(elsewhere),
                                start_new_session=True)
        time.sleep(0.3)
        try:
            assert _reap_orphans(packet) == []
            assert proc.poll() is None, "must not have been killed"
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_no_orphans_is_an_empty_list_not_an_error(self, tmp_path):
        from solexbench_agents.harnesses import _reap_orphans

        packet = tmp_path / "empty"
        packet.mkdir()
        assert _reap_orphans(packet) == []

    def test_does_not_kill_a_prefix_match(self, tmp_path):
        """`/x/packet-other` must not match `/x/packet`."""
        import subprocess
        import time

        from solexbench_agents.harnesses import _reap_orphans

        packet = tmp_path / "packet"
        packet.mkdir()
        sibling = tmp_path / "packet-other"
        sibling.mkdir()
        proc = subprocess.Popen(["sleep", "5"], cwd=str(sibling),
                                start_new_session=True)
        time.sleep(0.3)
        try:
            assert _reap_orphans(packet) == []
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait(timeout=10)
