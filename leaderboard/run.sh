#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve the leaderboard locally.
#
#   leaderboard/run.sh              # http://127.0.0.1:8088
#   PORT=9000 leaderboard/run.sh
#
# Runs on the HOST in its own venv, deliberately not in the pinned measurement
# container: adding fastapi to the image would change the environment every
# baseline in this repo was measured under.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8088}"
HOST_ADDR="${HOST_ADDR:-127.0.0.1}"

if [ ! -x "${HERE}/.venv/bin/python" ]; then
  echo "creating venv ..." >&2
  uv venv --python 3.11 "${HERE}/.venv" >&2
  VIRTUAL_ENV="${HERE}/.venv" uv pip install fastapi 'uvicorn[standard]' jinja2 >&2
fi

# Which database will the app serve? Ask the app; do not spell a path here.
# This guard used to test `leaderboard/solbench.db` while `ingest.py` wrote
# `db/solbench-<PART>.db`, so it was true on every run, the build it fired
# never satisfied it, and nothing said so.
#
# `ingest.py` builds for whatever part its manifest names -- it cannot conjure
# a part that has not been measured -- so the trigger is "no database at all".
# A resolved part that still has none after that is the app's honest empty
# state (DESIGN-v2 s6), not something a rebuild can fix.
#
# Fails closed: if the probe raises -- a bogus SOLBENCH_PART is the way it
# happens -- `read` gets nothing, returns non-zero, and `set -e` stops here
# rather than serving a board whose part the app is about to reject on every
# request.
IFS='|' read -r PART SERVED NDB < <(
  LB_HERE="${HERE}" "${HERE}/.venv/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["LB_HERE"])
import app
dbs = app.part_databases()
part = app.resolve_part()
print(f"{part}|{dbs.get(part, '')}|{len(dbs)}")
PY
)

if [ "${NDB}" = "0" ]; then
  echo "building database ..." >&2
  # An operator who pins SOLBENCH_DB has named the file the app will serve, so
  # build that one: the alternative is ingesting to the per-part path and then
  # serving an empty pin.
  DB_ARG=()
  if [ -n "${SOLBENCH_DB:-}" ]; then DB_ARG=(--db "${SOLBENCH_DB}"); fi
  "${HERE}/.venv/bin/python" "${HERE}/ingest.py" \
      ${DB_ARG[@]+"${DB_ARG[@]}"} >&2
elif [ -z "${SERVED}" ]; then
  echo "serving ${PART}, which has no database -- the board will show its" \
       "empty state. Measured parts: $(ls "${HERE}/db" 2>/dev/null | tr '\n' ' ')" >&2
fi

exec "${HERE}/.venv/bin/python" -m uvicorn app:app \
  --app-dir "${HERE}" --host "${HOST_ADDR}" --port "${PORT}" "$@"
