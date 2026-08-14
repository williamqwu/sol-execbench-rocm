# SPDX-License-Identifier: Apache-2.0
"""Run coding agents against SOL-ExecBench problems and collect what it cost.

Task 10. Upstream's task 09 step 2 calls for "an agent baseline sweep" -- one
agent, one number, the analogue of its median-0.732 figure. This package
generalizes that into a comparison across several (model, harness) pairs, which
is a different measurement and needs saying out loud: a harness is not a model.
Claude Code and Codex differ in how many tool calls they will make, how they
recover from a compile error, and how much context they carry between attempts,
so a difference in score is a difference between *agents*, not between the
underlying models on their own.

The pieces:

    task_packet   the sandboxed per-problem directory an agent works in
    harnesses     adapters for `claude -p` and `codex exec`, plus usage accounting
    gpu_pool      GPU leasing, so two agents never share a device
    runner        the resumable sweep across (harness, problem) pairs

Scoring lives outside this package on purpose: ``scripts/score_solutions.py``
re-evaluates each harvested solution from a pristine tree on the authoritative
GPU. Nothing an agent produced is trusted to score itself.
"""

from .gpu_pool import AUTHORITATIVE_GPU, GpuPool, default_agent_gpus
from .harnesses import HARNESSES, AgentSession, ClaudeCodeHarness, CodexHarness, Harness
from .runner import Sweep, Unit, discover_problems, load_deferred, preflight
from .task_packet import build_packet

__all__ = [
    "AUTHORITATIVE_GPU",
    "AgentSession",
    "ClaudeCodeHarness",
    "CodexHarness",
    "GpuPool",
    "HARNESSES",
    "Harness",
    "Sweep",
    "Unit",
    "build_packet",
    "default_agent_gpus",
    "discover_problems",
    "load_deferred",
    "preflight",
]
