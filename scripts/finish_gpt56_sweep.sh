#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Finish the gpt-5.6-sol 220 sweep unattended: wait out the fleet, collect,
# re-time what the first pass could not, merge, score, check coverage, publish.
#
#   scripts/finish_gpt56_sweep.sh            # from the repo root
#
# Written as a script rather than run by hand because the last four steps are
# the ones `TODO.md` D-list calls out as looking like a pipeline and not being
# one -- `sbt collect`, `agent_score.py` and `ingest.py` are three manual steps,
# and until all three run the kernels exist only in ~/.jobd/jobs/<id>/. A sweep
# that finishes at 3am and stops there is a sweep nobody published.
#
# Every step is idempotent and re-runnable. `--reuse-retimed` means an
# interrupted run resumes rather than re-measuring, and `ingest.py` rebuilds
# from artifacts, so running this twice costs a few minutes and changes nothing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SBT_PY=/home/qinwu/dev-imported/amdpilot-v2/dash-overlay/solbench-tasks/.venv/bin/python
GS_DB=/home/qinwu/dev-imported/amdpilot-v2/dash-overlay/gpu-scheduler/.state/gs.sqlite3
MANIFEST=artifacts/09/manifest-v1.2.json
LOG_DIR="${LOG_DIR:-/home/qinwu/.claude/jobs/764f49ce/tmp}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# The secrets carry the queue and scheduler tokens. sbt does not send them --
# it is one of the five callers the fleet's own secrets.env warns about -- so
# `scripts/port_via_fleet.py` attaches them and calls sbt's own entry point.
set -a; . /home/qinwu/.config/amdpilot/secrets.env; set +a

# ── 1. wait for the fleet ─────────────────────────────────────────────────────
# Polls the scheduler's ledger, read from a COPY: opening the live sqlite file
# while the scheduler is writing it is how a reader corrupts somebody else's
# WAL. Nothing here ever touches the original.
say "waiting for the fleet to drain ..."
while :; do
    cp "$GS_DB"* "$TMP"/ 2>/dev/null
    read -r RUNNING <<<"$(python3 - "$TMP/$(basename "$GS_DB")" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
print(c.execute("SELECT COUNT(*) FROM job_runs WHERE node LIKE 'gbt350%' "
                "AND started_at > 1786335500 AND state = 'running'").fetchone()[0])
PY
)"
    [ "${RUNNING:-1}" = "0" ] && break
    sleep 120
done
say "fleet drained"

# ── 2. collect ────────────────────────────────────────────────────────────────
# Overwrites each run.json with the FINAL sandbox contents. The first pass
# re-timed only the jobs that had finished by then; those results live in
# retimed/*.json and are reused below, so re-collecting cannot undo them.
for run in gpt56-180 gpt56-canary; do
    say "collecting $run"
    "$SBT_PY" scripts/port_via_fleet.py --cmd collect --run-id "$run" \
        >>"$LOG_DIR/finish.log" 2>&1 || say "  collect $run returned $?"
done

# ── 3. re-time whatever is not already done ───────────────────────────────────
# One at a time on GPU 0, which is what makes a number authoritative, and each
# measurement checks the card is exclusively ours first (STATE.md D29).
for run in gpt56-180 gpt56-canary; do
    say "re-timing $run (reusing what the first pass measured)"
    python3 scripts/agent_score.py --run "artifacts/10/$run" --gpu 0 \
        --reuse-retimed --manifest "$MANIFEST" >>"$LOG_DIR/finish.log" 2>&1 \
        || say "  agent_score $run returned $?"
done

# ── 4. merge into one board entry ─────────────────────────────────────────────
# gpt56-40 is the base: it keeps the run id, and the other two fill in. One
# submission, 220 problems, rather than three rows that invite a comparison
# between a sample and the sweep it is part of.
for fill in gpt56-canary gpt56-180; do
    say "merging $fill into gpt56-40"
    python3 scripts/merge_agent_runs.py --base artifacts/10/gpt56-40 \
        --fill "artifacts/10/$fill" >>"$LOG_DIR/finish.log" 2>&1 \
        || say "  merge $fill returned $?"
done

say "scoring the merged run"
python3 scripts/agent_score.py --run artifacts/10/gpt56-40 --gpu 0 \
    --reuse-retimed --manifest "$MANIFEST" 2>&1 | tail -5

# ── 5. coverage, which is not optional ────────────────────────────────────────
# A sweep that died partway and got marked done looks exactly like one that
# finished. This is the only thing that says otherwise (CLAUDE.md s0).
say "coverage"
python3 scripts/check_coverage.py --artifacts artifacts/10/gpt56-40 2>&1 | tail -15
COVERAGE_RC=$?

say "publishing"
leaderboard/.venv/bin/python leaderboard/ingest.py 2>&1 | tail -12

say "done. coverage exit=$COVERAGE_RC (non-zero means a gap with no reason attached)"
say ""
say "LEFT FOR A HUMAN: the merged run still has run_id 'gpt56-40' while holding"
say "220 problems. Renaming it is a supervised step, not an unattended one --"
say "ingest keys the board on scored.json's run_id and refuses to publish a"
say "board that lost a submission, so the rename and the --allow-drop that"
say "goes with it should happen where someone can read the output."
