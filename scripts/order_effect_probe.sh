#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Task 04 diagnostic — is the hip_events/rocprof divergence an ARM-ORDER effect?
#
# `compare_methodology.py` times both arms back to back in one process,
# hip_events first. Under MI355X's unlocked clock basis the second arm runs on
# a card that the first arm just heated, so a systematic bias would appear in
# the divergence that belongs to the thermal state and not to either
# methodology.
#
# The test: run the SAME problems both ways round, PAIRED and back to back, so
# the two orders see as nearly the same node conditions as possible. If the
# median divergence flips sign with the order, the bias is positional. If it
# does not, the shim genuinely reads low and that is a different problem.
#
# Not a sweep runner and not wired into shard_sweep: this answers one question
# once. The timed region is untouched (upstream warmup 10 + iterations 50).
#
#   bash scripts/order_effect_probe.sh <out-dir> [problem ...]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: order_effect_probe.sh <out-dir> [problem ...]}"
shift
PROBLEMS=("$@")

mkdir -p "${OUT}/A_hip_first" "${OUT}/B_rocprof_first"

n=0
for p in "${PROBLEMS[@]}"; do
  n=$((n + 1))
  key="L1__$(basename "${p}")"
  echo "=== [${n}/${#PROBLEMS[@]}] ${key} ==="

  # A then B, adjacent in time, so drift in node conditions is shared rather
  # than assigned to one order.
  for arm in "A_hip_first:hip_events,rocprof" "B_rocprof_first:rocprof,hip_events"; do
    dir="${arm%%:*}"; ord="${arm##*:}"
    dest="${OUT}/${dir}/${key}.json"
    [ -f "${dest}" ] && { echo "  ${dir}: already done"; continue; }
    echo "  ${dir}: order=${ord}"
    "${ROOT}/env/solb" python "${ROOT}/scripts/runners/compare_methodology.py" \
      --problem "${p}" --out "${dest}" --order "${ord}" >/dev/null 2>&1
    echo "    rc=$? $( [ -f "${dest}" ] && echo written || echo MISSING )"
  done
done

echo
echo "done. summarise with: python scripts/order_effect_report.py --probe ${OUT}"
