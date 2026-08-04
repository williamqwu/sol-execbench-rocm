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

if [ ! -f "${HERE}/solbench.db" ]; then
  echo "building database ..." >&2
  "${HERE}/.venv/bin/python" "${HERE}/ingest.py" >&2
fi

exec "${HERE}/.venv/bin/python" -m uvicorn app:app \
  --app-dir "${HERE}" --host "${HOST_ADDR}" --port "${PORT}" "$@"
