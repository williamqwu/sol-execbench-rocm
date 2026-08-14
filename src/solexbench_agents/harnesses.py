# SPDX-License-Identifier: Apache-2.0
"""Adapters that run one coding agent against one task packet.

Two harnesses, one interface. Each takes a packet directory and a GPU, runs its
CLI non-interactively until it stops or the wallclock cap fires, and returns an
:class:`AgentSession` recording what it cost.

The cost fields are not decoration. "Which model solves more problems" is not
answerable on its own -- a model given twice the budget should solve more -- so
the scoreboard reports solved-per-dollar and solved-per-minute beside the raw
count, and that requires the harness to report usage rather than the runner to
estimate it. Where a harness does not report cost (Codex reports tokens only),
the field is left null rather than filled with a guess.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Substrings that mark an infrastructure failure rather than a model answer.
# Observed on this node: the AMD LLM gateway intermittently returns
# "403 ... Access denied due to Virtual Network/Firewall rules" mid-session,
# which killed a smoke run after 27 productive turns and $1.36.
TRANSIENT_SIGNATURES = (
    "failed to authenticate",
    "gateway error",
    "access denied due to virtual network",
    "stream idle timeout",
    "no chunks received",
    "429",
    "rate limit",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "connection reset by peer",
    "temporarily unavailable",
)


@dataclass
class AgentSession:
    """Everything one agent invocation cost and produced."""

    harness: str
    model: str | None
    problem: str
    packet_dir: str
    gpu: int

    returncode: int | None = None
    timed_out: bool = False
    wallclock_s: float = 0.0

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    num_turns: int | None = None

    produced_solution: bool = False
    verify_attempts: int = 0
    verify_best_passed: int | None = None
    verify_workloads: int | None = None

    error: str | None = None
    orphans_killed: list[str] = field(default_factory=list)
    transient_failure: bool = False
    # Set from the harness's own structured terminal state, when it has one.
    # More reliable than matching strings in a log: Claude reports
    # `terminal_reason: "api_error"`, and the *message* varies ("403 gateway
    # error", "Stream idle timeout - no chunks received", ...) in ways no
    # signature list keeps up with.
    harness_api_error: bool = False
    attempt: int = 1
    stderr_tail: str = ""
    raw_usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class Harness:
    """Base class. Subclasses build a command line and parse usage."""

    name: str = "abstract"

    def __init__(self, *, timeout_s: int, model: str | None = None) -> None:
        self.timeout_s = timeout_s
        self.model = model

    # -- subclass interface ------------------------------------------------
    def command(self, packet_dir: Path, prompt: str) -> list[str]:
        raise NotImplementedError

    def parse_usage(self, stdout: str, session: AgentSession) -> None:
        raise NotImplementedError

    def recover_usage(self, packet_dir: Path, session: AgentSession) -> None:
        """Last-resort accounting when the CLI died before reporting any.

        Default is to do nothing, which leaves the token and cost fields null.
        Null is the correct answer when a harness genuinely leaves no record;
        overriding this is only right when there is a real second source.
        """
        session.cost_source = session.cost_source or "unavailable"

    # -- shared ------------------------------------------------------------
    def prompt(self, packet_dir: Path) -> str:
        return (
            "Read TASK.md in this directory and do what it says.\n\n"
            "You are in the working directory already. Everything you need is "
            "here: definition.json, reference.py, workload.jsonl.\n\n"
            "Write kernel.py and solution.json here EARLY -- as soon as you have "
            "a first attempt, before you start tuning. solution.json is the only "
            "file collected, and if the session ends without one (wallclock cap, "
            "dropped connection) a working kernel scores the same as nothing. "
            "Then run ./verify and iterate on kernel.py until every workload "
            "passes or your attempts run out.\n\n"
            "Stay in this directory. The rest of the repository is not part of "
            "the task and reading it is recorded."
        )

    def env(self, gpu: int) -> dict:
        env = dict(os.environ)
        # The agent sees exactly one GPU and it is renumbered to 0, so anything
        # the agent writes addresses cuda:0 and cannot reach a sibling -- which
        # is what keeps one agent's exploratory load off another's timings.
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Keep per-agent compile caches apart. A shared Triton or MIOpen cache
        # across concurrent agents is both a correctness hazard (one agent's
        # stale kernel served to another) and a timing one (a cache hit looks
        # like a fast compile).
        scratch = Path(os.environ.get("SOLEXBENCH_SCRATCH", "/var/tmp/solbench"))
        cache = scratch / "agent-caches" / f"gpu{gpu}"
        (cache / "triton").mkdir(parents=True, exist_ok=True)
        (cache / "torchinductor").mkdir(parents=True, exist_ok=True)
        (cache / "miopen").mkdir(parents=True, exist_ok=True)
        env["TRITON_CACHE_DIR"] = str(cache / "triton")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
        env["MIOPEN_USER_DB_PATH"] = str(cache / "miopen")
        env["MIOPEN_CUSTOM_CACHE_DIR"] = str(cache / "miopen")
        return env

    def run(self, packet_dir: Path, problem: str, gpu: int,
            log_dir: Path | None = None) -> AgentSession:
        session = AgentSession(
            harness=self.name,
            model=self.model,
            problem=problem,
            packet_dir=str(packet_dir),
            gpu=gpu,
        )
        cmd = self.command(packet_dir, self.prompt(packet_dir))
        started = time.time()
        stdout = stderr = ""
        proc = None
        try:
            # start_new_session puts the agent in its own process group so the
            # whole tree can be signalled. Without it, a timeout kills only the
            # CLI and every process the agent launched keeps running -- observed:
            # an autotuning script the agent wrote was still saturating a GPU an
            # hour after its session was capped. See STATE.md D22.
            proc = subprocess.Popen(
                cmd,
                cwd=str(packet_dir),
                env=self.env(gpu),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
                session.returncode = proc.returncode
                session.stderr_tail = (stderr or "")[-4000:]
                try:
                    self.parse_usage(stdout or "", session)
                except Exception as exc:  # noqa: BLE001
                    session.error = f"usage parse failed: {type(exc).__name__}: {exc}"
            except subprocess.TimeoutExpired:
                # A timeout is a result, not an absence of one: the solution.json
                # on disk at the cap is what the agent had, and it is scored as
                # such. But the processes it spawned must not outlive it.
                session.timed_out = True
                session.error = f"wallclock cap {self.timeout_s}s exceeded"
                _kill_group(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    stdout = stderr = ""
                session.stderr_tail = (stderr or "")[-4000:]
        except Exception as exc:  # noqa: BLE001
            session.error = f"{type(exc).__name__}: {exc}"
            if proc is not None:
                _kill_group(proc)
        finally:
            # Belt and braces: a process that escaped the group (double-forked, or
            # re-parented) is found by its working directory instead. An orphan
            # holding a GPU inflates every later measurement on that device and
            # nothing in the output would say so.
            session.orphans_killed = _reap_orphans(packet_dir)

        session.wallclock_s = time.time() - started
        if session.input_tokens is None:
            self.recover_usage(packet_dir, session)
        # Kept on disk, not in the session JSON: it is large, and it is the only
        # way to find out *why* an agent behaved as it did. The smoke run that
        # first exercised this path failed on a gateway 403 whose message existed
        # nowhere except here.
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "agent-stdout.log").write_text(stdout)
            (log_dir / "agent-stderr.log").write_text(stderr)
        self._harvest(packet_dir, session)
        session.transient_failure = self.is_transient_failure(session, stdout, stderr)
        return session

    @staticmethod
    def is_transient_failure(session: AgentSession, stdout: str, stderr: str) -> bool:
        """Is this an infrastructure failure rather than the agent's answer?

        A gateway 403, a rate limit or an upstream 5xx says nothing about the
        model, and counting one as "did not solve it" would understate the score
        by however often the gateway happened to wobble. Retried rather than
        recorded as a result -- the exception to prime directive 1's "a failure is
        a result", because this is not a failure *of the thing being measured*.
        """
        if session.produced_solution:
            return False
        if session.harness_api_error:
            return True
        haystack = f"{session.error or ''}\n{stderr[-8000:]}\n{stdout[-8000:]}".lower()
        return any(sig in haystack for sig in TRANSIENT_SIGNATURES)

    @staticmethod
    def _harvest(packet_dir: Path, session: AgentSession) -> None:
        """Read what the packet ended up holding, however the run terminated."""
        session.produced_solution = (packet_dir / "solution.json").exists()
        attempts = packet_dir / ".attempts.json"
        if attempts.exists():
            try:
                state = json.loads(attempts.read_text())
            except json.JSONDecodeError:
                return
            session.verify_attempts = int(state.get("used", 0))
            passes = [e.get("passed") for e in state.get("log", [])
                      if isinstance(e.get("passed"), int)]
            counts = [e.get("workloads") for e in state.get("log", [])
                      if isinstance(e.get("workloads"), int)]
            session.verify_best_passed = max(passes) if passes else None
            session.verify_workloads = max(counts) if counts else None


class ClaudeCodeHarness(Harness):
    """Anthropic's Claude Code, headless.

    ``--output-format json`` gives a single result object carrying token counts,
    cost in USD and turn count, which is exactly the accounting the scoreboard
    needs and is the reason this harness reports cost while Codex does not.
    """

    name = "claude-code"

    def command(self, packet_dir: Path, prompt: str) -> list[str]:
        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
            # The agent must compile, run python and execute ./verify without a
            # human to approve each call. There is no interactive channel in a
            # sweep, so an unapproved tool call is a hung session.
            #
            # `--permission-mode bypassPermissions` rather than
            # `--dangerously-skip-permissions`: the latter refuses outright when
            # the process is root, which it is here, exiting non-zero before any
            # model call. See `env()` for the other half of this.
            "--permission-mode", "bypassPermissions",
        ]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def env(self, gpu: int) -> dict:
        env = super().env(gpu)
        # Claude Code blocks permission bypass for root unless it is told it is
        # already inside a sandbox. This container runs as uid 0 with no option
        # to drop privileges (the ROCm devices and /opt/venv are root-owned), and
        # the node itself is the isolation boundary. Setting this is an assertion
        # about the environment, and it is true: the container is disposable and
        # the repo is version-controlled.
        env["IS_SANDBOX"] = "1"
        return env

    def recover_usage(self, packet_dir: Path, session: AgentSession) -> None:
        """Rebuild token counts from Claude's own transcript.

        Needed because the result object -- which is where cost and tokens live --
        is only emitted when the CLI exits normally. Killing it at the wallclock
        cap therefore loses the entire accounting for that session, and in the
        first pilot that was 12 of 14 sessions: a budget cap that silently sees
        almost nothing is worse than no cap, because it reads as enforced.

        Tokens are recoverable. **Cost is not** -- the transcript records no
        price -- so ``cost_usd`` stays null and ``cost_source`` says why. Deriving
        it by multiplying tokens by a rate inferred from another session would put
        an estimate in a currency column, indistinguishable from a measurement.
        """
        transcript = _find_claude_transcript(packet_dir)
        if transcript is None:
            session.cost_source = "unavailable: no transcript found"
            return

        totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0}
        turns = 0
        for line in transcript.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = (event.get("message") or {}).get("usage")
            if not usage:
                continue
            turns += 1
            for key in totals:
                totals[key] += int(usage.get(key) or 0)

        if not turns:
            session.cost_source = "unavailable: transcript carried no usage"
            return
        session.raw_usage = {**totals, "recovered_from_transcript": str(transcript)}
        session.input_tokens = totals["input_tokens"]
        session.output_tokens = totals["output_tokens"]
        session.cached_input_tokens = totals["cache_read_input_tokens"]
        session.num_turns = turns
        session.cost_source = (
            "tokens recovered from transcript; cost unavailable because the CLI "
            "was killed at the wallclock cap before it reported one"
        )

    def parse_usage(self, stdout: str, session: AgentSession) -> None:
        blob = _last_json_object(stdout)
        if blob is None:
            session.error = session.error or "no JSON result object in claude output"
            return
        session.cost_source = "harness result object"
        session.raw_usage = blob.get("usage", {}) or {}
        usage = session.raw_usage
        session.input_tokens = usage.get("input_tokens")
        session.output_tokens = usage.get("output_tokens")
        session.cached_input_tokens = usage.get("cache_read_input_tokens")
        session.cost_usd = blob.get("total_cost_usd")
        session.num_turns = blob.get("num_turns")
        models = blob.get("modelUsage") or {}
        if models and not self.model:
            session.model = next(iter(models))
        # The authoritative signal that the run died in the transport rather than
        # in the model's reasoning. Both a gateway 403 and a stream idle timeout
        # arrive this way, with different messages and with subtype "success".
        if blob.get("terminal_reason") == "api_error":
            session.harness_api_error = True
        if blob.get("is_error"):
            # `result` carries the actual message; `subtype` can read "success"
            # even on a failed run, which made a gateway 403 look like a clean
            # finish that simply produced nothing.
            detail = str(blob.get("result") or blob.get("subtype") or "unknown")
            session.error = session.error or f"claude reported error: {detail[:400]}"


class CodexHarness(Harness):
    """OpenAI Codex CLI, headless via ``codex exec --json``.

    Emits JSON *lines* rather than one object, and the token counts arrive on a
    ``turn.completed`` event -- possibly several, one per turn -- so usage is
    accumulated across events rather than read from a single field.

    Codex reports no cost. That field stays null: this proxy's pricing is not
    published to the CLI, and a scoreboard column filled by multiplying tokens by
    a guessed rate would look like a measurement.
    """

    name = "codex"

    def command(self, packet_dir: Path, prompt: str) -> list[str]:
        cmd = [
            "codex", "exec",
            "--json",
            # Same reason as Claude's flag: no human is available to approve a
            # compile or a ./verify call.
            "--dangerously-bypass-approvals-and-sandbox",
            # The packet is not a git repo, and Codex refuses to run outside one
            # unless told the check is unnecessary.
            "--skip-git-repo-check",
            "-C", str(packet_dir),
        ]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append(prompt)
        return cmd

    def parse_usage(self, stdout: str, session: AgentSession) -> None:
        totals = {"input_tokens": 0, "output_tokens": 0,
                  "cached_input_tokens": 0, "reasoning_output_tokens": 0}
        turns = 0
        saw_usage = False
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed":
                turns += 1
                usage = event.get("usage") or {}
                saw_usage = True
                for key in totals:
                    totals[key] += int(usage.get(key) or 0)
            elif event.get("type") == "turn.failed":
                turns += 1
                err = (event.get("error") or {}).get("message") or "turn.failed"
                session.error = session.error or f"codex turn failed: {err}"

        if not saw_usage:
            session.error = session.error or "no turn.completed event in codex output"
            return
        session.raw_usage = totals
        session.input_tokens = totals["input_tokens"]
        session.output_tokens = totals["output_tokens"]
        session.cached_input_tokens = totals["cached_input_tokens"]
        session.reasoning_tokens = totals["reasoning_output_tokens"]
        session.num_turns = turns


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the agent's whole process group."""
    import os
    import signal

    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _reap_orphans(packet_dir: Path) -> list[str]:
    """Kill any surviving process whose working directory is inside *packet_dir*.

    The cwd is the reliable marker: the agent is launched with the packet as its
    working directory, so anything it spawns inherits it, including through a
    double fork that escapes the process group. Matching on cwd rather than on a
    command name also means it catches scripts the agent invented -- the observed
    orphan was called ``explore10.py``, a name nothing could have predicted.

    Returns the command lines it killed, so a session record says whether this
    happened rather than leaving it to be inferred from a busy GPU.
    """
    return _reap_under(packet_dir)


def reap_eval_scratch(scratch_root: Path | None = None) -> list[str]:
    """Kill orphans left in the evaluation scratch tree.

    Separate from `_reap_orphans` because an eval subprocess does NOT run in the
    packet directory: `runners/_common.py` gives it a fresh `solb_run_*` staging dir
    under ``SOLEXBENCH_SCRATCH``, so the packet-scoped sweep cannot see it however
    thoroughly it searches.

    That gap stalled a full pipeline run for 15 hours. Two `eval_driver.py` processes
    outlived their agents, were reparented to init, and sat holding GPU memory for
    ~19 h. `require_idle()` correctly refused to start the scoring stage while an
    `eval_driver` was alive, `reap()` could not find them to kill, and so the driver
    exited immediately after the sweep and every later stage simply never ran. The
    guard was right and the cleanup was blind, which is the worst pairing: the run
    fails safe and silent.
    """
    import os
    import tempfile

    root = scratch_root or Path(
        os.environ.get("SOLEXBENCH_SCRATCH", tempfile.gettempdir()))
    killed: list[str] = []
    if not root.is_dir():
        return killed
    # Only the staging dirs, not the whole scratch root: sibling trees under
    # /var/tmp/solbench include the dataset and SOLAR's own scratch, and something
    # legitimately working there must not be killed.
    for staging in sorted(root.glob("solb_run_*")):
        killed += _reap_under(staging)
    return killed


def _reap_under(target_dir: Path) -> list[str]:
    """Kill every process whose cwd is at or below *target_dir*. Returns cmdlines."""
    import os
    import signal

    target = str(target_dir.resolve())
    killed: list[str] = []
    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if cwd != target and not cwd.startswith(target + os.sep):
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace").strip()
            os.kill(pid, signal.SIGKILL)
            killed.append(f"{pid}: {cmdline[:200]}")
        except OSError:
            continue
    return killed


def _find_claude_transcript(packet_dir: Path) -> Path | None:
    """Claude Code's own session log for a run in *packet_dir*.

    Claude stores transcripts under ``~/.claude/projects/<mangled-cwd>/``, where
    the mangling replaces every non-alphanumeric character with ``-``. That is
    tried first because it is O(1); if the mangling ever changes, the fallback
    scans transcripts for one whose recorded ``cwd`` matches, which is the
    property actually being relied on.
    """
    import os
    import re

    root = Path(os.path.expanduser("~/.claude/projects"))
    if not root.is_dir():
        return None

    mangled = re.sub(r"[^a-zA-Z0-9-]", "-", str(packet_dir))
    candidates = sorted((root / mangled).glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True) \
        if (root / mangled).is_dir() else []
    if candidates:
        return candidates[0]

    target = str(packet_dir)
    for path in sorted(root.glob("*/*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open(errors="replace") as fh:
                for _ in range(20):
                    line = fh.readline()
                    if not line:
                        break
                    if f'"cwd": "{target}"' in line or f'"cwd":"{target}"' in line:
                        return path
        except OSError:
            continue
    return None


def _as_text(value) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _last_json_object(text: str) -> dict | None:
    """The last complete top-level JSON object in *text*.

    Claude prints one result object, but a wrapper or a warning can prepend
    lines, so the tail is parsed rather than the whole stream.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Fall back to the whole payload, for the case where the object is
    # pretty-printed across lines.
    try:
        blob = json.loads(text)
        return blob if isinstance(blob, dict) else None
    except json.JSONDecodeError:
        return None


HARNESSES: dict[str, type[Harness]] = {
    ClaudeCodeHarness.name: ClaudeCodeHarness,
    CodexHarness.name: CodexHarness,
}
