#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run `sbt port` with the fleet's write tokens attached. Adds a header, nothing else.

    JQ_API_TOKEN=... python scripts/port_via_fleet.py \
        --run-id gpt56-180 --model gpt-5.6-sol \
        --problems-file artifacts/10/gpt56-220/remaining-problems.txt

**Why this exists rather than a patch to `sbt`.** On 2026-08-09 23:55 the fleet's
`secrets.env` had `JQ_API_TOKEN`, `DB_API_TOKEN` and `MAPI_ADMIN_TOKEN`
uncommented, which switches on each service's write guard. That file's own
comment says not to, and says why:

    COMMENTED OUT, AND THAT IS NOT AN OVERSIGHT. Exporting these turns the
    guards on, and the fleet's own internal callers do not present them yet --
    five sites need wiring first [...] Uncomment only after those five send the
    header, or the next `fleet.sh restart` 401s the fleet against itself.

`sbt` is one of the callers that does not send it, so `sbt port` now dies on
`401 POST /v1/tasks needs a valid X-JQ-Token header`. That is somebody else's
work in progress and this repo does not touch `dash-overlay/`.

So: import their package, wrap `httpx.Client` to attach the header, call their
own `sbt.cli.main`. Every decision about what a task is, how the prompt is
built, which spec is stated and how the authoritative GPU is held stays theirs.
Nothing here reimplements it, so nothing here can drift from it.

**Read the warning that comes with that.** `gs/runner.py -> JQ_API_TOKEN` is on
the same list of five, which means the SCHEDULER may not be able to call the
queue either. If so, jobs written here are accepted and then never placed. That
failure is silent -- a queued J2 waiting for a gap looks identical to a queued
J2 nobody can start -- so port one problem first and watch it actually run
before porting a sweep. `--canary` does exactly that.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SBT_ROOT = Path("/home/qinwu/dev-imported/amdpilot-v2/dash-overlay/solbench-tasks")

#: Header name -> the environment variable holding its value. Both services'
#: names come from their own config, not from a guess here:
#: `jq/config.py: API_TOKEN_HEADER = os.environ.get("JQ_API_TOKEN_HEADER", "X-JQ-Token")`
#: `gs/config.py: JOBQUEUE_TOKEN_HEADER = _env_str("JQ_API_TOKEN_HEADER", "X-JQ-Token")`
TOKENS = {
    "X-JQ-Token": "JQ_API_TOKEN",
    "X-GS-Token": "GS_API_TOKEN",
}


def install_headers() -> list[str]:
    """Attach every token present in the environment to outgoing requests.

    Sent to every host the client dials, which is loopback-only here (7201,
    7202, 7205). Worth being explicit that this is a convenience and not a
    security boundary: do not point this at anything off the node.
    """
    import httpx

    headers = {h: os.environ[v] for h, v in TOKENS.items() if os.environ.get(v)}
    if not headers:
        return []

    original = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        merged = dict(headers)
        merged.update(dict(kwargs.pop("headers", None) or {}))
        original(self, *args, headers=merged, **kwargs)

    httpx.Client.__init__ = patched
    return sorted(headers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="port",
                    choices=["port", "collect", "status", "report", "guard"],
                    help="which sbt subcommand to run with the tokens "
                         "attached. `collect` needs them too: it reads the "
                         "database, and the write guards that broke `port` are "
                         "per service, not per verb.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default=None,
                    help="required for --cmd port. "
                         "Reaches the container as the model to ask for; sbt "
                         "records it as `model_requested` and `sbt collect` "
                         "reads back which upstream actually answered.")
    ap.add_argument("--problems-file", type=Path, default=None,
                    help="required for --cmd port")
    ap.add_argument("--canary", action="store_true",
                    help="port only the FIRST problem in the file. Use this "
                         "before a sweep: a J2 the scheduler cannot start "
                         "looks exactly like a J2 waiting for a gap")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.cmd != "port":
        if not SBT_ROOT.is_dir():
            raise SystemExit(f"{SBT_ROOT} is not there; the fleet moved")
        sys.path.insert(0, str(SBT_ROOT))
        sent = install_headers()
        print(f"tokens attached: {sent or 'none'}")
        from sbt.cli import main as sbt_main
        return sbt_main([a.cmd, a.run_id])

    if a.problems_file is None or a.model is None:
        raise SystemExit("--problems-file and --model are required for --cmd port")
    problems = [ln.strip() for ln in a.problems_file.read_text().splitlines()
                if ln.strip()]
    if not problems:
        raise SystemExit(f"{a.problems_file}: no problems listed")
    if a.canary:
        problems = problems[:1]

    if not SBT_ROOT.is_dir():
        raise SystemExit(f"{SBT_ROOT} is not there; the fleet moved")
    sys.path.insert(0, str(SBT_ROOT))

    sent = install_headers()
    print(f"tokens attached: {sent or 'none -- the guards had better be off'}")

    os.environ["SBT_MODEL"] = a.model
    from sbt.cli import main as sbt_main  # noqa: E402

    argv = ["port", "--run-id", a.run_id, "--problems", *problems]
    if a.dry_run:
        argv.append("--dry-run")
    print(f"sbt {' '.join(argv[:4])} ... ({len(problems)} problems)")
    return sbt_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
