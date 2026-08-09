#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The GPU lease broker: exclusivity, crash release, and refusing to guess.

Why this exists at all. Measured over `glm-sweep-2` -- 220 problems, 1,574
evaluations -- the run spent **9.9 GPU-hours computing inside 217 GPU-hours of
allocation**, 4.6% utilisation. The median evaluation is 9.3 s and a session
runs six of them, so each card was held for an hour to serve about three
minutes. The card is pinned for the session because that is the only way the
fleet can guarantee the one property that may not be given up: a timing run has
the device to itself. The broker keeps that property and drops the pinning.

Everything downstream of a measurement assumes the measurement was exclusive.
So the tests that matter are not "does it hand out numbers" but:

  * **Two clients never hold one card.** Checked by having clients report the
    interval they held and asserting no two intervals on a card overlap --
    not by trusting the broker's own bookkeeping, which is the thing under test.
  * **A dead client releases.** The lease is an open socket, so the kernel does
    it. There is no heartbeat to tune and no reaper to wedge, and this asserts
    that the design actually behaves that way under SIGKILL rather than that
    the intent was written down.
  * **A refusal is never a lease.** A client that cannot be served must fail
    loudly and distinctly. D33 is the precedent: a scorer timeout that read on
    the board as a kernel producing nothing, indistinguishable from a real
    failure for as long as nobody looked.
  * **GPU 0 cannot be leased.** Every published score is a re-time there. A
    broker able to hand it out could invalidate the whole board, not just its
    own run, so the refusal is at startup and not a convention.

Not asserted here: that leasing preserves the *measured latency*. That needs
real silicon and is verified separately -- 14 concurrent evaluations on 7 cards
land within 2.5% of the same measurement on an idle card, while 4 concurrent on
one card without the broker inflate it up to 2.14x. That number is in
`sbt/sandbox.py`'s note and in STATE.md; it cannot be a unit test because there
is no GPU in CI.
"""

from __future__ import annotations

import collections
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BROKER = ROOT / "scripts" / "gpu_broker.py"
LEASE = ROOT / "scripts" / "gpu_lease.py"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def broker():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(BROKER), "--gpus", "1,2,3", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("broker never came up")
    yield port
    proc.kill()
    proc.wait()


def status(port: int) -> dict:
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    f = s.makefile("rw", encoding="utf-8", newline="\n")
    f.write("STATUS\n")
    f.flush()
    out = json.loads(f.readline())
    s.close()
    return out


def _hold(port, who, secs, sink, lock):
    r = subprocess.run(
        [sys.executable, str(LEASE), "--port", str(port), "--who", who,
         "--require-broker", "--",
         sys.executable, "-c",
         f"import os,time;t=time.time();time.sleep({secs});"
         f"print(os.environ['SOLB_LEASED_GPU'],t,time.time())"],
        capture_output=True, text=True)
    with lock:
        sink.append((r.returncode, r.stdout.strip(), r.stderr.strip()))


def test_two_clients_never_hold_the_same_card(broker):
    """The property everything downstream assumes. Asserted from what the
    clients observed, not from the broker's own view of who holds what."""
    got, lock = [], threading.Lock()
    ts = [threading.Thread(target=_hold, args=(broker, f"c{i}", 0.30, got, lock))
          for i in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert all(rc == 0 for rc, _, _ in got), [g for g in got if g[0]]
    by_card = collections.defaultdict(list)
    for _, out, _ in got:
        gpu, start, end = out.split()
        by_card[int(gpu)].append((float(start), float(end)))
    assert set(by_card) <= {1, 2, 3}
    for card, spans in by_card.items():
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            assert a[1] <= b[0] + 1e-6, (
                f"card {card} was held by two clients at once: {a} and {b}")


def test_all_twelve_were_actually_served(broker):
    """Exclusivity is trivial to achieve by serving nobody."""
    got, lock = [], threading.Lock()
    ts = [threading.Thread(target=_hold, args=(broker, f"c{i}", 0.05, got, lock))
          for i in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(got) == 12 and all(rc == 0 for rc, _, _ in got)
    assert status(broker)["granted"] >= 12


def test_a_killed_holder_releases_without_a_reaper(broker):
    before = status(broker)
    assert len(before["free"]) == 3
    p = subprocess.Popen(
        [sys.executable, str(LEASE), "--port", str(broker), "--require-broker",
         "--", sys.executable, "-c", "import time;time.sleep(60)"])
    for _ in range(100):
        if len(status(broker)["free"]) == 2:
            break
        time.sleep(0.05)
    else:
        p.kill()
        pytest.fail("the lease was never taken")
    p.kill()
    p.wait()
    for _ in range(100):
        if len(status(broker)["free"]) == 3:
            return
        time.sleep(0.05)
    pytest.fail("SIGKILL leaked a lease; the socket is supposed to BE the lease")


def test_the_authoritative_gpu_cannot_be_leased():
    """Startup refusal, not a convention. GPU 0 carries every score on the
    board, so a broker that could hand it out is a board-wide hazard."""
    r = subprocess.run([sys.executable, str(BROKER), "--gpus", "0,1",
                        "--port", str(_free_port())],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
    assert "authoritative" in (r.stdout + r.stderr)


def test_a_queue_timeout_is_a_refusal_and_never_a_lease(broker):
    """The client must be able to tell "could not measure" from "measured
    badly". D33 shipped the version where it could not."""
    blockers = [subprocess.Popen(
        [sys.executable, str(LEASE), "--port", str(broker), "--require-broker",
         "--", sys.executable, "-c", "import time;time.sleep(30)"])
        for _ in range(3)]
    try:
        for _ in range(100):
            if not status(broker)["free"]:
                break
            time.sleep(0.05)
        r = subprocess.run(
            [sys.executable, str(LEASE), "--port", str(broker), "--timeout", "1",
             "--require-broker", "--", sys.executable, "-c",
             "print('MUST NOT RUN')"],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 76, (r.returncode, r.stdout, r.stderr)
        assert "MUST NOT RUN" not in r.stdout
    finally:
        for b in blockers:
            b.kill()
            b.wait()


def test_no_broker_is_fatal_only_when_the_caller_says_so():
    """A sweep must refuse; a developer running one evaluation by hand on a
    laptop with no fleet must not be blocked by a service they do not have."""
    port = _free_port()          # nothing is listening here
    hard = subprocess.run(
        [sys.executable, str(LEASE), "--port", str(port), "--require-broker",
         "--", sys.executable, "-c", "print('MUST NOT RUN')"],
        capture_output=True, text=True, timeout=60)
    assert hard.returncode == 77 and "MUST NOT RUN" not in hard.stdout

    soft = subprocess.run(
        [sys.executable, str(LEASE), "--port", str(port), "--",
         sys.executable, "-c", "print('ran')"],
        capture_output=True, text=True, timeout=60)
    assert soft.returncode == 0 and "ran" in soft.stdout


def test_something_that_is_not_a_broker_is_diagnosed_as_such(broker):
    """Found by pointing the client at an occupied port: the JSONDecodeError
    escaped and the process exited 1, which a caller reads as "the child
    failed" -- the exact confusion the exit codes exist to prevent."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]

    def talk():
        c, _ = srv.accept()
        c.sendall(b"<!DOCTYPE HTML>\n")
        time.sleep(0.5)
        c.close()
    threading.Thread(target=talk, daemon=True).start()
    try:
        r = subprocess.run(
            [sys.executable, str(LEASE), "--port", str(port), "--require-broker",
             "--", sys.executable, "-c", "print('MUST NOT RUN')"],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 76, (r.returncode, r.stderr)
        assert "not a gpu-broker" in r.stderr
        assert "MUST NOT RUN" not in r.stdout
    finally:
        srv.close()


def test_the_wait_is_recorded_so_the_budget_cost_is_measurable(broker, tmp_path):
    """Queueing is spent out of the agent's own session hour. If it is not
    recorded, the run's constraint is 'one hour' with an unknown deduction."""
    wf = tmp_path / "gpu-wait.txt"
    r = subprocess.run(
        [sys.executable, str(LEASE), "--port", str(broker), "--require-broker",
         "--wait-file", str(wf), "--", sys.executable, "-c", "pass"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert wf.exists()
    assert float(wf.read_text().strip()) >= 0.0
