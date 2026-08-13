# SPDX-License-Identifier: Apache-2.0
"""What `scripts/apply_tolerances.py` carries from artifacts/05 into a workload.

CPU-only, and it writes nothing outside `tmp_path`: every output path is a CLI
flag, so the real `artifacts/05` and `reference/b200-tolerances.json` are not
touched.

The defect covered here (STATE.md D52b) is an omission rather than a wrong
number: the script rebuilt `tolerance` from a fixed set of keys, so the
per-output facts task 05 derives -- which outputs are compared exactly, and how
wide the single band is for an output whose own dtype earns a tighter one --
stopped at `artifacts/05/*.json` and never reached the workload the harness and
the board read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

UUID = "11111111-2222-3333-4444-555555555555"


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    """A one-problem dataset and a one-problem calibration artifact."""
    data = tmp_path / "data"
    prob = data / "L2" / "049_group_limited_topk_routing"
    prob.mkdir(parents=True)
    (prob / "workload.jsonl").write_text(json.dumps({
        "uuid": UUID,
        "axes": {"n": 4},
        "inputs": {},
        "tolerance": {"max_atol": 1e-2, "max_rtol": 1e-2,
                      "required_matched_ratio": 0.99},
    }) + "\n")

    cal = tmp_path / "cal"
    cal.mkdir()
    (cal / "L2__049.json").write_text(json.dumps({
        "problem": "L2__049_group_limited_topk_routing",
        "per_workload": [{
            "workload_uuid": UUID,
            "ok": True,
            "deterministic": False,
            "run_to_run": {"max_abs": 0.0, "max_rel": 0.0,
                           "exact_outputs_max_abs": 3.0},
            "tolerance": {
                "max_atol": 1.2e-07,
                "max_rtol": 1.1920929e-07,
                "required_matched_ratio": 0.99,
                "_exact_outputs": [0],
                "_dtype_floors": [{"dtype": "torch.float32", "n_outputs": 1,
                                   "rms": 1.0, "atol": 1.1920929e-07,
                                   "rtol": 1.1920929e-07}],
                "_floor_over_grant": {"atol": 1.0, "rtol": 1.0},
                "_derivation": "test fixture, not a measurement",
            },
        }],
    }))
    return data, cal


def _run(tmp_path: Path) -> tuple[dict, str]:
    data, cal = _tree(tmp_path)
    out_wl = tmp_path / "workloads"
    triage = tmp_path / "triage.md"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "apply_tolerances.py"),
         "--calibration", str(cal), "--data", str(data),
         "--out-workloads", str(out_wl), "--out-triage", str(triage),
         "--out-b200", str(tmp_path / "b200.json")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    line = (out_wl / "L2" / "049_group_limited_topk_routing"
            / "workload.jsonl").read_text().strip()
    return json.loads(line)["tolerance"], triage.read_text()


class TestPerOutputFactsSurvive:
    def test_exact_outputs_reaches_the_shipped_workload(self, tmp_path):
        tol, _ = _run(tmp_path)
        assert tol["_exact_outputs"] == [0], (
            "the workload must state which outputs the band does not apply to"
        )

    def test_the_per_dtype_floors_reach_it_too(self, tmp_path):
        tol, _ = _run(tmp_path)
        assert tol["_dtype_floors"][0]["dtype"] == "torch.float32"
        assert tol["_floor_over_grant"] == {"atol": 1.0, "rtol": 1.0}

    def test_the_enforced_fields_are_unchanged(self, tmp_path):
        """Carrying extra keys must not move a number the harness reads."""
        tol, _ = _run(tmp_path)
        assert tol["max_atol"] == 1.2e-07
        assert tol["max_rtol"] == 1.1920929e-07
        assert tol["required_matched_ratio"] == 0.99

    def test_the_extra_keys_are_ignored_by_the_schema(self, tmp_path):
        """They are records, not fields: ToleranceSpec must parse and drop them."""
        from sol_execbench.core.data.workload import ToleranceSpec

        tol, _ = _run(tmp_path)
        spec = ToleranceSpec(**tol)
        assert set(spec.model_dump()) == {
            "max_atol", "max_rtol", "required_matched_ratio",
            "max_error_cap", "allow_negative_inf",
        }
        assert spec.max_atol == tol["max_atol"]


class TestNondeterminismTriageIsNotSelfContradictory:
    def test_integer_variance_is_its_own_column(self, tmp_path):
        """`max_abs 0.0` on a row headed "non-deterministic" reads as a bug.

        Since D52 the run-to-run `max_abs` covers float outputs only, so a
        reference that is non-deterministic purely in its indices lands in this
        table with 0.0. The integer figure has to be next to it or the table
        contradicts itself.
        """
        _tol, triage = _run(tmp_path)
        header = [ln for ln in triage.splitlines()
                  if ln.startswith("| problem | workload |")][0]
        assert "int/bool outputs" in header
        row = [ln for ln in triage.splitlines() if UUID[:8] in ln][0]
        assert "| 3 |" in row, row
