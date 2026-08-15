#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The worker must be told which manifest to score against.

The defect this pins was silent and expensive. `worker.py` invoked
`scripts/agent_score.py` with no `--manifest`, that script defaulted to
`artifacts/09/manifest-v1.json` -- MI350X's frozen release manifest -- and the
worker holds GPU 0 on an MI355X card. Every submission scored through the board
was compared against another part's bounds. Nothing in the output said so; the
symptom was a mean S about 0.08 too high.

`agent_score.py` now requires the flag and refuses on a part mismatch, which
turns the silent version into an exit code. These tests are about the other
half: that the worker *supplies* it, and that it does not invent one.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")


def test_the_worker_refuses_to_start_without_a_manifest(monkeypatch):
    """No default, on purpose.

    A default resolved from the visible cards would fix the cross-part accident
    and leave the remaining decision -- which VERSION of this part's manifest the
    board publishes against -- to whichever filename sorted last. Scores are
    comparable only within one manifest version.
    """
    import worker

    monkeypatch.delenv("SOLBENCH_MANIFEST", raising=False)
    with pytest.raises(SystemExit) as e:
        worker.resolve_manifest(None)
    msg = str(e.value)
    # The refusal has to be actionable: it names both parts' manifests, because
    # the reader is an operator on one node who needs the other one to be wrong.
    assert "SOLBENCH_MANIFEST" in msg
    for path in worker.MANIFEST_BY_PART.values():
        assert path in msg


def test_an_explicit_manifest_wins_over_the_environment(monkeypatch, tmp_path):
    m = tmp_path / "manifest-v4.json"
    m.write_text("{}")
    other = tmp_path / "manifest-v1.json"
    other.write_text("{}")
    import worker

    monkeypatch.setenv("SOLBENCH_MANIFEST", str(other))
    assert worker.resolve_manifest(str(m)) == m
    assert worker.resolve_manifest(None) == other


def test_a_manifest_that_does_not_exist_is_refused_before_the_gpu_lock(tmp_path,
                                                                      monkeypatch):
    """Refusing after `acquire_lock()` would leave a lock file behind for a
    worker that never ran a job, and the lock is deliberately not auto-cleared.
    """
    import worker

    monkeypatch.delenv("SOLBENCH_MANIFEST", raising=False)
    with pytest.raises(SystemExit) as e:
        worker.resolve_manifest(str(tmp_path / "nope.json"))
    assert "does not exist" in str(e.value)


def test_the_scorer_is_invoked_with_the_manifest(monkeypatch, tmp_path):
    """The flag reaches `agent_score.py`'s argv, not just the worker's own state.

    Asserted on the command line rather than on a return value: the bug was that
    the argument was absent from exactly this list.
    """
    import worker

    seen = {}

    class _Done:
        returncode = 0
        stdout = stderr = ""

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _Done()

    monkeypatch.setattr(worker.subprocess, "run", _fake_run)
    manifest = tmp_path / "manifest-v4.json"
    manifest.write_text("{}")
    worker.score(tmp_path / "run", 60, manifest)

    cmd = seen["cmd"]
    assert "--manifest" in cmd
    assert cmd[cmd.index("--manifest") + 1] == str(manifest)
    assert cmd[cmd.index("--run") + 1] == str(tmp_path / "run")
