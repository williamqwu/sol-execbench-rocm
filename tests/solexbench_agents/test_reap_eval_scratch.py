# SPDX-License-Identifier: Apache-2.0
"""Orphaned eval subprocesses must be reapable, or the pipeline stalls silently.

An eval subprocess runs in its own `solb_run_*` staging dir rather than in the agent
packet, so the packet-scoped reaper cannot see it. That combination cost a full run 15
hours: two `eval_driver.py` processes outlived their agents, were reparented to init,
and held GPU memory for ~19 h; `require_idle()` correctly refused to begin scoring
while an `eval_driver` lived, `reap()` could not find them to kill, and the driver
exited straight after the sweep with every later stage never running.

The guard was right and the cleanup was blind. These tests cover the blindness.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solexbench_agents.harnesses import reap_eval_scratch  # noqa: E402


def _sleeper(cwd: Path) -> subprocess.Popen:
    """A process parked in *cwd*, standing in for a stranded eval_driver."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                         cwd=str(cwd),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):                      # wait for cwd to be observable
        try:
            if os.readlink(f"/proc/{p.pid}/cwd") == str(cwd.resolve()):
                return p
        except OSError:
            pass
        time.sleep(0.05)
    return p


def _gone(p: subprocess.Popen, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.poll() is not None:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
def test_kills_a_process_stranded_in_a_staging_dir(tmp_path):
    staging = tmp_path / "solb_run_abc123"
    staging.mkdir()
    p = _sleeper(staging)
    try:
        killed = reap_eval_scratch(tmp_path)
        assert any(str(p.pid) in k for k in killed), killed
        assert _gone(p)
    finally:
        if p.poll() is None:
            p.kill()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
def test_kills_deeper_descendants_of_a_staging_dir(tmp_path):
    """eval_driver writes subdirectories and may be running inside one."""
    deep = tmp_path / "solb_run_abc123" / "nested" / "deeper"
    deep.mkdir(parents=True)
    p = _sleeper(deep)
    try:
        assert any(str(p.pid) in k for k in reap_eval_scratch(tmp_path))
        assert _gone(p)
    finally:
        if p.poll() is None:
            p.kill()


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
def test_leaves_sibling_scratch_trees_alone(tmp_path):
    """/var/tmp/solbench also holds the dataset and SOLAR's scratch. Reaping the
    whole root would kill a T_SOL derivation, which is a 27-minute job."""
    other = tmp_path / "sol-scratch"
    other.mkdir()
    p = _sleeper(other)
    try:
        assert reap_eval_scratch(tmp_path) == []
        assert p.poll() is None, "a process outside solb_run_* must survive"
    finally:
        p.kill()


def test_missing_scratch_root_is_not_an_error(tmp_path):
    """reap() runs before every timing stage, including on a fresh machine."""
    assert reap_eval_scratch(tmp_path / "does-not-exist") == []


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
def test_does_not_kill_the_caller(tmp_path):
    """The reaper's own cwd can sit inside the tree it is sweeping."""
    staging = tmp_path / "solb_run_self"
    staging.mkdir()
    here = Path.cwd()
    os.chdir(staging)
    try:
        reap_eval_scratch(tmp_path)          # must return, not suicide
    finally:
        os.chdir(here)
