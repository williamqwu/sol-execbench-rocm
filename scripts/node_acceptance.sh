#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Task 00 — node acceptance. Run first, before anything depends on this node.
#
# Prevents the day-three discovery that the node has seven healthy GPUs and one
# that throttles, and that two days of measurements are contaminated.
#
# Emits artifacts/00/node-report.json and a human-readable log.

set -uo pipefail   # deliberately not -e: a missing tool should be RECORDED,
                   # not fatal. Prime directive 1.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/artifacts/00"
mkdir -p "${OUT}"

say() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "host"
hostname; uname -a; date -u

say "amd-smi / rocm-smi"
if have amd-smi; then
  amd-smi version
  amd-smi static 2>/dev/null | head -100
  amd-smi metric 2>/dev/null | head -60
elif have rocm-smi; then
  echo "amd-smi absent, falling back to rocm-smi"
  rocm-smi --showallinfo 2>/dev/null | head -120
else
  echo "MISSING: neither amd-smi nor rocm-smi found — record this in STATE.md"
fi

say "rocm version"
cat /opt/rocm/.info/version 2>/dev/null || echo "MISSING /opt/rocm/.info/version"

say "amdgpu driver"
cat /sys/module/amdgpu/version 2>/dev/null || echo "MISSING amdgpu module version"

say "topology"
if have rocm-smi; then rocm-smi --showtopo 2>/dev/null | head -60; fi
numactl --hardware 2>/dev/null | head -20 || echo "numactl absent"

say "torch"
python3 - <<'PY'
try:
    import torch
    print("torch      ", torch.__version__)
    print("hip        ", getattr(torch.version, "hip", None))
    print("cuda       ", getattr(torch.version, "cuda", None))
    print("available  ", torch.cuda.is_available())
    n = torch.cuda.device_count()
    print("devices    ", n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  arch={getattr(p,'gcnArchName','?')}  "
              f"mem={p.total_memory/2**30:.0f}GiB  CUs={p.multi_processor_count}")
except Exception as e:
    print("torch probe FAILED:", e)
PY

say "repo self-test (no GPU required)"
( cd "${ROOT}" && python3 -m pytest tests/ -q 2>&1 | tail -5 )

say "dataset"
if [ -d "${ROOT}/data" ]; then
  # Real layout is data/SOL-ExecBench/benchmark/<Category>/<problem>/, one
  # level deeper than first assumed; -L because data/ may be a symlink.
  find -L "${ROOT}/data" -name definition.json | wc -l | \
    xargs printf 'problems found: %s\n'
  for c in L1 L2 Quant FlashInfer-Bench; do
    n=$(find -L "${ROOT}/data" -path "*/${c}/*" -name definition.json 2>/dev/null | wc -l)
    printf '  %-18s %s\n' "${c}" "${n}"
  done
  echo "expected: L1=94 L2=82 Quant=33 FlashInfer-Bench=26"
else
  echo "data/ absent. Fetch with:"
  echo "  huggingface-cli download nvidia/SOL-ExecBench --repo-type dataset --local-dir data/"
  echo "This was never verified from the build environment — if gated or"
  echo "differently laid out, record what you find in STATE.md."
fi

say "building node-report.json"
python3 "${ROOT}/scripts/build_node_report.py" --out "${OUT}/node-report.json"

echo
echo "Next: python scripts/verify_artifacts.py --task 00"
