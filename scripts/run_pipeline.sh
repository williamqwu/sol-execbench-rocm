#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Task 10 end to end, unattended. Meant to be run inside tmux:
#
#   tmux new -s solb -d 'bash scripts/run_pipeline.sh'
#   tmux attach -t solb
#
# Every stage is idempotent: a completed stage leaves a marker under
# artifacts/10/pipeline/ and is skipped on restart, so this can be killed and
# relaunched at any point without losing work or repeating GPU time.
#
# THE ORDERING IS THE POINT. Stages 1 and 4 measure timings on the authoritative
# GPU and must have the node to themselves; stage 5 saturates seven GPUs and 120 CPUs
# with agents. Running them concurrently is what voided the first T_b measurement:
# re-running the identical variant came out 5x slower than the recorded anchor,
# because Triton autotuning is CPU-bound and seven compile-heavy agents starve it.
# GPU-to-GPU interference is negligible (+0.02%, task 01); CPU contention is not.
set -uo pipefail   # deliberately not -e: a failing stage must be RECORDED and
                   # the remaining independent stages still attempted.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MARKERS="${ROOT}/artifacts/10/pipeline"
LOGS="${ROOT}/artifacts/10/pipeline/logs"
mkdir -p "${MARKERS}" "${LOGS}"

MANIFEST="artifacts/09/manifest-MI355X-v1.json"
RUN_ID="${SOLB_RUN_ID:-full-01}"

say() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$1"; }
done_marker() { echo "${MARKERS}/$1.done"; }
is_done() { [ -f "$(done_marker "$1")" ]; }
mark_done() { date -u +%FT%TZ > "$(done_marker "$1")"; }

# Every ancestor of this process, self included. A wrapper shell's command line
# contains the text of whatever it was asked to run, so any ancestor can match a
# pattern we are searching for and look like a running job.
ancestors() {
  local pid=$$ next
  # PPid from /proc/<pid>/status, not from field 4 of /proc/<pid>/stat: the comm
  # field there is parenthesized but may itself contain spaces, which shifts every
  # later field and yields the state character instead of the parent pid.
  while [ -n "${pid}" ] && [ "${pid}" -gt 1 ] 2>/dev/null; do
    echo "${pid}"
    next="$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)"
    [ -n "${next}" ] || break
    pid="${next}"
  done
}

# PIDs matching a pattern, excluding this process and its ancestors. pgrep rather
# than `ps aux | grep "[a]bc"`: the bracket trick only hides grep itself and does
# nothing about the ancestor case above.
matching_pids() {
  local exclude
  exclude="$(ancestors | paste -sd'|' -)"
  pgrep -f "$1" 2>/dev/null | grep -vxE "${exclude:-^$}" || true
}

running() { [ -n "$(matching_pids "$1")" ]; }

# Wait for a process already in flight to exit, rather than starting a second
# copy. Two timing passes on one GPU inflate each other and the artifact records
# the device it was told to use either way (D11).
wait_for() {
  local pattern="$1" label="$2"
  if running "${pattern}"; then
    say "waiting for ${label} already in flight (pids: $(matching_pids "${pattern}" | tr '\n' ' '))"
    while running "${pattern}"; do sleep 60; done
    say "${label} finished"
  fi
}

# Kill anything an agent left behind that is still holding a GPU. An orphan
# inflates every later measurement on that device and nothing in the output says
# so (STATE.md D22).
reap() {
  env/solb-native python - <<'PY' 2>/dev/null || true
import sys
from pathlib import Path
sys.path.insert(0, "src")
from solexbench_agents.harnesses import _reap_orphans
killed = []
for run in Path("artifacts/10/runs").glob("*/*/*/packet"):
    killed += _reap_orphans(run)
if killed:
    print(f"reaped {len(killed)} orphan process(es) still holding a packet")
PY
}

require_idle() {
  reap
  local busy=""
  for p in "claude -p" "codex exec" "eval_driver" "run_agents"; do
    if running "${p}"; then busy="${busy} ${p}"; fi
  done
  if [ -n "${busy}" ]; then
    say "REFUSING to start a timing stage:${busy} still running. A timing stage
         needs the node to itself -- CPU contention from compile-heavy agents is
         what voided the first T_b measurement. Wait or kill, then re-run."
    return 1
  fi
  return 0
}

say "pipeline start — run_id=${RUN_ID}"
env/solb-native python -c "
import sys; sys.path.insert(0,'scripts')
from provenance import stamp
p = stamp('10-pipeline')['_provenance']
print(f\"part={p['part']} f_lock={p['f_lock_mhz']} torch={p['torch']['version']}\")"

# ---------------------------------------------------------------- stage 0
# T_SOL for this part. CPU and meta-device only, so it can overlap anything --
# but it must land before the manifest, and it must be derived with the CURRENT
# sol_bounds.py. The version before master's fp32-rate fix priced 160 of 235
# problems at the vector rate, 16x low, and put bounds above measured times.
if is_done t_sol; then
  say "stage 0 T_SOL — already done, skipping"
else
  wait_for "sol_bounds.py" "a T_SOL derivation"
  say "stage 0 T_SOL for MI355X at the measured F_LOCK"
  # Deliberately NOT --resume. The scratch directory is keyed by problem, not by
  # (problem, code version) -- deviation D10 -- so resuming after a change to how
  # the bound is computed silently mixes bounds from both versions, and the
  # mixture is indistinguishable from a clean run. An interrupted stage 0 leaves
  # no marker and re-derives all 235 from scratch, which costs ~30 CPU-minutes
  # and is the cheap side of that trade.
  env/solb-native python -u scripts/sol_bounds.py \
      --part MI355X --freq-mhz 1640 \
      --out artifacts/03/t_sol-MI355X.json --jobs 16 --timeout 900 \
      2>&1 | tee -a "${LOGS}/00-t-sol.log" | tail -20
  mark_done t_sol
fi

# ---------------------------------------------------------------- stage 1
# Authoritative T_b. One GPU, serially, on an otherwise idle node.
if is_done tb_authoritative; then
  say "stage 1 T_b authoritative — already done, skipping"
else
  wait_for "authoritative_tb" "an authoritative T_b pass"
  require_idle || exit 1
  say "stage 1 T_b authoritative — $(ls artifacts/06/authoritative/*.json 2>/dev/null | wc -l) \
of $(ls artifacts/06/candidates/*.json 2>/dev/null | wc -l) candidates already re-timed"
  # Run it unconditionally. authoritative_tb.py skips any problem whose artifact
  # already exists, so this costs nothing when the work is done and cannot skip
  # work that is not. Guessing completeness from a file count would do both wrong:
  # a pass that legitimately ends below the candidate count (some problems have no
  # winner) would look unfinished forever.
  env/solb-native python -u scripts/authoritative_tb.py \
      --candidates artifacts/06/candidates \
      --out artifacts/06/authoritative \
      --gpu 0 --top-k 2 --within 0.25 --timeout 900 \
      2>&1 | tee -a "${LOGS}/01-tb-authoritative.log"
  mark_done tb_authoritative
fi

# ---------------------------------------------------------------- stage 2
# The second bound tier: traffic the definition itself declares, over DRAM
# bandwidth. It takes T_b as a GATE -- a derived bound above the measured time is
# rejected rather than shipped -- so it has to run after stage 1, not before.
say "stage 2 traffic-floor bound tier"
env/solb-native python scripts/sol_traffic_floor.py \
    --t-sol artifacts/03/t_sol-MI355X.json \
    --arch SOLAR/configs/arch/MI355X.yaml \
    --t-b artifacts/06/authoritative \
    --out artifacts/03/t_sol_traffic-MI355X.json \
    2>&1 | tee "${LOGS}/02-traffic-floor.log" | tail -12

# ---------------------------------------------------------------- stage 3
# Freeze the manifest. Scores are only meaningful inside one manifest version.
# Both tiers go in; build_manifest takes the max of the two that survive being
# checked against the measurement, symmetrically, and counts a workload as not
# scoreable where neither does.
say "stage 3 manifest"
env/solb-native python scripts/build_manifest.py \
    --out "${MANIFEST}" --version MI355X-v1 \
    --t-sol artifacts/03/t_sol-MI355X.json \
    --t-sol-traffic artifacts/03/t_sol_traffic-MI355X.json \
    --t-b artifacts/06/authoritative --force \
    2>&1 | tee "${LOGS}/03-manifest.log" | head -10

# ---------------------------------------------------------------- stage 4
# The check that decides whether S may be published at all. Task 06 step 4.
if is_done anchor; then
  say "stage 4 anchor — already done, skipping"
else
  say "stage 4 anchor verification (needs an idle node)"
  require_idle || exit 1
  HIP_VISIBLE_DEVICES=0 env/solb-native python -u scripts/verify_anchor.py \
      --manifest "${MANIFEST}" \
      --out artifacts/06/anchor-verification.md --sample 12 \
      2>&1 | tee "${LOGS}/04-anchor.log"
  # A failing report must not be mistaken for a passing one by a later session,
  # and must not be mistaken for a clean measurement either.
  if env/solb-native python -c "
import json, sys
d = json.load(open('artifacts/06/anchor-verification.md'))
p = d.get('anchor_property') or {}
sys.exit(0 if p.get('total') and p['passing'] == p['total'] else 1)"; then
    say "anchor property HOLDS — S may be published"
    mark_done anchor
  else
    say "anchor property FAILS — S will NOT be published. Recording and continuing;
         the agent sweep does not depend on it."
    mv artifacts/06/anchor-verification.md \
       "artifacts/06/anchor-verification-FAILED-$(date -u +%Y%m%dT%H%M%SZ).md"
  fi
fi

# ---------------------------------------------------------------- stage 5
# The agent sweep. Seven GPUs, hours. Resumable: a unit with a session.json is
# done, and --retry-transient re-runs only the ones that failed on infrastructure
# rather than on the model's answer.
if is_done sweep; then
  say "stage 5 agent sweep — already done, skipping"
else
  say "stage 5 agent sweep (${RUN_ID})"
  reap
  env/solb-native python -u scripts/run_agents.py \
      --run-id "${RUN_ID}" \
      --categories L1 L2 FlashInfer-Bench \
      --max-attempts 5 --timeout-min 30 \
      --budget-usd "${SOLB_BUDGET_USD:-1500}" \
      --retry-transient \
      2>&1 | tee -a "${LOGS}/05-sweep.log"
  mark_done sweep
fi

# ---------------------------------------------------------------- stage 6
# Scoring. Serial, authoritative GPU, and only once the sweep is finished --
# two evaluations sharing a GPU inflate each other and the artifact cannot tell
# you it happened (D11).
say "stage 6 scoring ${RUN_ID} on the authoritative GPU"
require_idle || exit 1
env/solb-native python -u scripts/score_solutions.py \
    --run-id "${RUN_ID}" --manifest "${MANIFEST}" --timeout 2400 \
    2>&1 | tee -a "${LOGS}/06-score.log"

# ---------------------------------------------------------------- stage 8
say "stage 7 backfill scores from the manifest"
env/solb-native python scripts/backfill_scores.py \
    --run-id "${RUN_ID}" --manifest "${MANIFEST}" \
    2>&1 | tee "${LOGS}/07-backfill.log"

# ---------------------------------------------------------------- stage 7
say "stage 8 scoreboard + coverage + acceptance"
env/solb-native python scripts/build_scoreboard.py --all-runs \
    2>&1 | tee "${LOGS}/08-scoreboard.log"

for h in claude-code codex; do
  env/solb-native python scripts/check_coverage.py \
      --artifacts "artifacts/10/scores/${RUN_ID}/${h}" \
      2>&1 | tee -a "${LOGS}/08-coverage.log"
done

env/solb-native python scripts/verify_artifacts.py --task 10 \
    2>&1 | tee "${LOGS}/08-acceptance.log"

say "pipeline complete"
echo
echo "  artifacts/10/dashboard.html    the scoreboard"
echo "  artifacts/10/scoreboard.json   the same, machine-readable"
echo "  ${LOGS}/                       per-stage logs"
echo
echo "Nothing here is committed. Review, then commit."
