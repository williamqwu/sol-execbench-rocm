#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A GPU lease broker, so agent concurrency stops being bounded by GPU count.

    python scripts/gpu_broker.py --gpus 1,2,3,4,5,6,7 --port 7210

Measured on `glm-sweep-2` (220 problems, 1,574 evaluations): the whole run spent
**9.9 GPU-hours** actually computing, inside **217 GPU-hours** of allocation --
7 cards held for 31 wall-hours. **4.6% utilisation.** The median `./evaluate`
takes 9.3 s and a session runs 6 of them, so each card sat idle for roughly 59
minutes of every hour to serve about three minutes of work.

The reason is structural, not wasteful scheduling: an agent thinks, reads, and
writes code for almost all of its hour, and touches the GPU only to measure. The
fleet gives it a whole card for the whole session because that is the only way
to guarantee **the one thing that must not be given up** -- that a timing run
has the device to itself. Two agents timing on one card produce two wrong
numbers, and the agent then optimises against the wrong one.

So: keep exclusivity, drop the pinning. A card is leased for the seconds an
evaluation runs and returned. Sessions outnumber cards; the queue absorbs it.
Simulated against the empirical service-time and think-time distributions, 40
concurrent sessions finish the benchmark in 6.1 h instead of 32 h at 23% GPU
utilisation, with a median wait of 0 s and a p99 of about 60 s.

**A lease is an open TCP connection, and that is the whole crash-safety story.**
The broker holds the socket for the life of the lease; if the client segfaults,
is SIGKILLed, or its container dies, the kernel closes the socket and the card
is free on the next poll. No heartbeat, no TTL to tune, no reaper that can
itself wedge -- the failure mode of every lease daemon that tracks liveness
out-of-band. `--network=host` is already required of these containers (job
daemon §3.1), so loopback needs no new mount and no new capability.

What this deliberately does NOT do:

* **Kill a slow holder.** A lease past its expected duration is logged and
  left alone. The longest evaluation in the corpus is 961 s and it is a real
  measurement; interrupting one to keep a queue moving would corrupt the thing
  the queue exists to protect.
* **Hand out GPU 0.** It is refused at startup rather than by convention. Every
  published score is a re-time on an idle GPU 0 (CLAUDE.md §4), so a broker that
  could lease it would be able to invalidate the scores of every run on the
  board, not merely the one that borrowed it.
* **Fabricate a lease on timeout.** A client that cannot be served fails loudly.
  A measurement taken without the lease is worse than no measurement, because
  nothing downstream can tell it apart from a good one.
"""

from __future__ import annotations

import argparse
import collections
import json
import socket
import threading
import time
from dataclasses import dataclass, field

PROTOCOL = 1
AUTHORITATIVE_GPU = 0


@dataclass
class Lease:
    gpu: int
    who: str
    granted_at: float
    waited_s: float
    conn: socket.socket


@dataclass
class Stats:
    granted: int = 0
    refused: int = 0
    waits: list = field(default_factory=list)
    holds: list = field(default_factory=list)
    peak_queue: int = 0


class Broker:
    """FIFO over a fixed set of cards. One lock, no async, no dependencies.

    FIFO is stated rather than emergent: a poll-and-retry scheme has no queue
    and therefore no bound on how long an unlucky client waits. At 23% expected
    utilisation starvation is unlikely, but "unlikely" is not a property you can
    report, and the wait per session has to be reportable -- it is spent out of
    the agent's own hour.
    """

    def __init__(self, gpus: list[int], verbose: bool = False):
        bad = [g for g in gpus if g == AUTHORITATIVE_GPU]
        if bad:
            raise SystemExit(
                f"refusing to lease GPU {AUTHORITATIVE_GPU}: it is the "
                "authoritative timing device (CLAUDE.md §4). Every score on "
                "the board is a re-time there, so leasing it out would put "
                "every published number at risk, not just this run's.")
        if not gpus:
            raise SystemExit("no GPUs to lease")
        self.free: collections.deque[int] = collections.deque(sorted(set(gpus)))
        self.all = sorted(set(gpus))
        self.held: dict[int, Lease] = {}
        self.waiting: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.stats = Stats()
        self.verbose = verbose
        self.started = time.time()

    # -- the queue ---------------------------------------------------------
    def acquire(self, who: str, conn: socket.socket, timeout: float):
        """Block until a card is free. Returns (gpu, waited_s) or (None, ...)."""
        ticket = threading.Event()
        box: dict = {}
        t0 = time.monotonic()
        with self.lock:
            self.waiting.append((ticket, box, who, conn))
            self.stats.peak_queue = max(self.stats.peak_queue, len(self.waiting))
            self._pump_locked()
        if not ticket.wait(timeout):
            with self.lock:
                # Still queued: remove it. It may have been granted between the
                # wait expiring and this lock, in which case the card is ours
                # and must go back rather than leak.
                self.waiting = collections.deque(
                    w for w in self.waiting if w[0] is not ticket)
                if "gpu" in box:
                    self._release_locked(box["gpu"])
                else:
                    self.stats.refused += 1
                    return None, time.monotonic() - t0
        return box.get("gpu"), time.monotonic() - t0

    def _pump_locked(self):
        while self.free and self.waiting:
            ticket, box, who, conn = self.waiting.popleft()
            gpu = self.free.popleft()
            box["gpu"] = gpu
            self.held[gpu] = Lease(gpu, who, time.time(), 0.0, conn)
            self.stats.granted += 1
            ticket.set()

    def release(self, gpu: int):
        with self.lock:
            self._release_locked(gpu)

    def _release_locked(self, gpu: int):
        lease = self.held.pop(gpu, None)
        if lease is not None:
            self.stats.holds.append(time.time() - lease.granted_at)
        if gpu not in self.free:
            self.free.append(gpu)
        self._pump_locked()

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            w = sorted(self.stats.waits)
            h = sorted(self.stats.holds)
            pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0
            return {
                "protocol": PROTOCOL,
                "uptime_s": round(now - self.started, 1),
                "gpus": self.all,
                "free": sorted(self.free),
                "held": {str(g): {"who": l.who, "held_s": round(now - l.granted_at, 1)}
                         for g, l in sorted(self.held.items())},
                "queued": len(self.waiting),
                "peak_queue": self.stats.peak_queue,
                "granted": self.stats.granted,
                "refused_timeout": self.stats.refused,
                "utilisation": round(sum(h) / max(1e-9, (now - self.started)
                                                  * len(self.all)), 4),
                "wait_s": {"n": len(w), "median": round(pct(w, .5), 2),
                           "p90": round(pct(w, .9), 2), "p99": round(pct(w, .99), 2),
                           "max": round(max(w), 2) if w else 0.0,
                           "total": round(sum(w), 1)},
                "hold_s": {"n": len(h), "median": round(pct(h, .5), 2),
                           "p99": round(pct(h, .99), 2),
                           "max": round(max(h), 2) if h else 0.0,
                           "total": round(sum(h), 1)},
            }


def serve_client(broker: Broker, conn: socket.socket, addr, timeout: float):
    gpu = None
    try:
        conn.settimeout(timeout + 60)
        f = conn.makefile("rw", encoding="utf-8", newline="\n")
        line = f.readline().strip()
        if not line:
            return
        verb, _, rest = line.partition(" ")
        verb = verb.upper()

        if verb == "STATUS":
            f.write(json.dumps(broker.snapshot()) + "\n")
            f.flush()
            return

        if verb != "ACQUIRE":
            f.write(json.dumps({"error": f"unknown verb {verb!r}"}) + "\n")
            f.flush()
            return

        # `ACQUIRE <timeout> <who...>`. The client's deadline is carried on the
        # wire and enforced HERE, because the broker owns the queue: a
        # client-side timer racing a server-side one produces the case where a
        # card is granted to a caller that has already given up, and the lease
        # then leaks until the socket happens to close.
        #
        # It has to be on the wire at all because the first version simply did
        # not use the client's `--timeout` -- accepted it, and queued against
        # the broker's own 900s regardless. The same shape as D33: a flag that
        # names a behaviour it never reaches. `who` is last so it may contain
        # spaces.
        ask, _, who = rest.partition(" ")
        try:
            wanted = float(ask)
        except ValueError:              # legacy `ACQUIRE <who>` form
            wanted, who = timeout, rest
        # Clamped: a client may ask to wait less, never more. Otherwise one
        # caller's generous deadline becomes the broker's problem.
        deadline = max(0.0, min(wanted, timeout))
        who = who.strip() or f"{addr}"
        gpu, waited = broker.acquire(who, conn, deadline)
        with broker.lock:
            broker.stats.waits.append(waited)
        if gpu is None:
            # Loud, and never a lease. See the module docstring.
            f.write(json.dumps({
                "error": "timeout",
                "waited_s": round(waited, 1),
                "detail": (f"no GPU became free within {timeout:.0f}s. This is "
                           "a refusal, not a lease: measuring without one "
                           "would produce a number indistinguishable from a "
                           "good one.")}) + "\n")
            f.flush()
            return
        f.write(json.dumps({"gpu": gpu, "waited_s": round(waited, 3),
                            "protocol": PROTOCOL}) + "\n")
        f.flush()
        if broker.verbose:
            print(f"[grant] gpu {gpu} -> {who} (waited {waited:.1f}s)", flush=True)
        # Hold until the peer goes away. A read that returns b"" is the socket
        # closing, which is the release -- whether the client did it politely
        # or its container was destroyed underneath it.
        conn.settimeout(None)
        while True:
            try:
                if not conn.recv(4096):
                    break
            except OSError:
                break
    except Exception as exc:            # noqa: BLE001 - a client must not kill the broker
        if broker.verbose:
            print(f"[error] {addr}: {exc!r}", flush=True)
    finally:
        if gpu is not None:
            broker.release(gpu)
            if broker.verbose:
                print(f"[free ] gpu {gpu}", flush=True)
        try:
            conn.close()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpus", default="1,2,3,4,5,6,7",
                    help="comma-separated torch indices; GPU 0 is refused")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7210)
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="how long a client may queue before it is refused")
    ap.add_argument("--status-every", type=float, default=300.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    gpus = [int(x) for x in a.gpus.split(",") if x.strip() != ""]
    broker = Broker(gpus, verbose=a.verbose)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(512)
    print(f"gpu-broker up on {a.host}:{a.port}, leasing {gpus}, "
          f"queue timeout {a.timeout:.0f}s", flush=True)

    def ticker():
        while True:
            time.sleep(a.status_every)
            print("[status] " + json.dumps(broker.snapshot()), flush=True)
    threading.Thread(target=ticker, daemon=True).start()

    while True:
        conn, addr = srv.accept()
        threading.Thread(target=serve_client,
                         args=(broker, conn, addr, a.timeout),
                         daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
