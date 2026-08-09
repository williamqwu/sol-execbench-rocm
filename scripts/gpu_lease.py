#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a command holding an exclusive GPU lease from `gpu_broker.py`.

    python scripts/gpu_lease.py --who L1__085 -- ./some-timing-command

Sets `HIP_VISIBLE_DEVICES` to the leased card for the child and nothing else,
so a process that runs outside a lease sees whatever it was already given -- it
does not silently inherit someone else's device.

**Stdlib only, on purpose.** This runs inside the agent container as
`/opt/venv/bin/python`, and the one thing that must never fail here is the part
that decides whether a measurement is allowed to happen. A dependency would put
a pip install between the benchmark and its own correctness.

Exit codes are the child's, except:

  76  the broker refused (queue timeout). Distinct from any child status so a
      caller can tell "the benchmark could not be measured" from "the kernel is
      broken" -- which is exactly the confusion D33 shipped, where a scorer
      timeout was indistinguishable on the board from a kernel that produced
      nothing.
  77  no broker reachable, and `--require-broker` was set.

Without `--require-broker` an unreachable broker is a warning and the command
runs with the environment it already had. That is the right default for a
developer running one evaluation by hand on a machine with no fleet, and the
wrong one for a sweep -- so a sweep passes the flag.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

EXIT_REFUSED = 76
EXIT_NO_BROKER = 77


class Refused(RuntimeError):
    pass


def acquire(host: str, port: int, who: str, timeout: float):
    """Open the lease. The returned socket IS the lease -- do not close it
    until the work is done, and do not hand it to the child."""
    sock = socket.create_connection((host, port), timeout=30)
    # No read timeout: the queue wait is bounded by the broker, which answers
    # with an explicit refusal. A client-side deadline racing the server-side
    # one produces the case where the broker grants a card to a client that has
    # already walked away, which is how a lease leaks.
    sock.settimeout(None)
    f = sock.makefile("rw", encoding="utf-8", newline="\n")
    # The deadline goes ON THE WIRE and the broker enforces it. This argument
    # used to be accepted here and then never used -- every client queued
    # against the broker's own default no matter what it asked for, which is
    # D33's shape exactly: a `--timeout` that does not reach the wait it names.
    f.write(f"ACQUIRE {timeout:.3f} {who}\n")
    f.flush()
    line = f.readline()
    if not line:
        raise Refused("broker closed the connection without answering")
    try:
        reply = json.loads(line)
    except ValueError:
        # Connected, but the peer is not a broker. Found by pointing this at a
        # port that happened to be occupied: the JSONDecodeError escaped and
        # the process died with status 1, which a caller reads as "the child
        # failed" -- the one thing this exit-code scheme exists to prevent.
        raise Refused(
            f"whatever is listening on this port is not a gpu-broker; it "
            f"answered {line[:120]!r}. Check the port before treating this as "
            f"a GPU shortage.") from None
    if not isinstance(reply, dict):
        raise Refused(f"broker answered {type(reply).__name__}, expected an object")
    if "error" in reply:
        raise Refused(reply.get("detail") or reply["error"])
    if not isinstance(reply.get("gpu"), int):
        raise Refused(f"broker granted no usable device: {reply!r}")
    return sock, reply


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("SOLB_GPU_BROKER_HOST",
                                                     "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("SOLB_GPU_BROKER_PORT", "7210")))
    ap.add_argument("--who", default=os.environ.get("JOB_ID", "anonymous"))
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--require-broker", action="store_true",
                    default=os.environ.get("SOLB_GPU_BROKER_REQUIRED") == "1")
    ap.add_argument("--wait-file", default=None,
                    help="append the wait in seconds here, so the budget cost "
                         "of queueing is measurable after the fact")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("nothing to run")

    env = dict(os.environ)
    sock = None
    t0 = time.monotonic()
    try:
        sock, reply = acquire(a.host, a.port, a.who, a.timeout)
        env["HIP_VISIBLE_DEVICES"] = str(reply["gpu"])
        # The child is told what it holds and for whom, so an artifact written
        # inside it can record the device without guessing from the environment
        # it happens to have inherited.
        env["SOLB_LEASED_GPU"] = str(reply["gpu"])
        env["SOLB_LEASE_WAIT_S"] = str(reply.get("waited_s", 0.0))
        waited = float(reply.get("waited_s", 0.0))
    except Refused as exc:
        print(f"gpu_lease: refused after {time.monotonic() - t0:.0f}s: {exc}",
              file=sys.stderr)
        return EXIT_REFUSED
    except OSError as exc:
        if a.require_broker:
            print(f"gpu_lease: no broker at {a.host}:{a.port} ({exc}). "
                  "Refusing to measure: --require-broker is set because an "
                  "unbrokered timing on a shared node is a number nothing "
                  "downstream can tell apart from a good one.", file=sys.stderr)
            return EXIT_NO_BROKER
        print(f"gpu_lease: no broker at {a.host}:{a.port} ({exc}); running "
              "with the environment as given", file=sys.stderr)
        waited = 0.0

    if a.wait_file:
        try:
            with open(a.wait_file, "a", encoding="utf-8") as fh:
                fh.write(f"{waited:.3f}\n")
        except OSError:
            pass

    try:
        return subprocess.call(cmd, env=env)
    finally:
        # Releases the lease. Also happens automatically if we are killed --
        # that is the point of the socket being the lease.
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
