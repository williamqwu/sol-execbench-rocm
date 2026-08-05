# SPDX-License-Identifier: Apache-2.0
"""The coverage checker has to read every layout the pipeline actually writes.

It could not read the agent sweep's. `--pattern` assumed
`<Category>/<problem>/<file>`, while the sweep writes `<Category>__<problem>/<file>`,
so every one of 202 present problems was reported missing. A checker that cries wolf
at that volume stops being consulted, which defeats the point of having one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_coverage import covered  # noqa: E402


def test_joined_layout_as_the_agent_sweep_writes_it(tmp_path):
    for key in ("L1__001_foo", "Quant__002_bar", "FlashInfer-Bench__003_baz"):
        d = tmp_path / key
        d.mkdir()
        (d / "session.json").write_text("{}")
    assert covered(tmp_path, "session.json") == {
        "L1__001_foo", "Quant__002_bar", "FlashInfer-Bench__003_baz"}


def test_nested_layout_still_works(tmp_path):
    """artifacts/05/workloads and friends use this form; it must not regress."""
    for cat, prob in (("L1", "001_foo"), ("Quant", "002_bar")):
        d = tmp_path / cat / prob
        d.mkdir(parents=True)
        (d / "workload.jsonl").write_text("{}")
    assert covered(tmp_path, "workload.jsonl") == {"L1__001_foo", "Quant__002_bar"}


def test_a_category_with_a_hyphen_is_not_mistaken_for_a_joined_name(tmp_path):
    """FlashInfer-Bench contains a hyphen but no double underscore, so the nested
    form must still be recognised for it."""
    d = tmp_path / "FlashInfer-Bench" / "003_baz"
    d.mkdir(parents=True)
    (d / "workload.jsonl").write_text("{}")
    assert covered(tmp_path, "workload.jsonl") == {"FlashInfer-Bench__003_baz"}


def test_both_layouts_side_by_side(tmp_path):
    """Nothing requires a tree to be homogeneous, and a mixed one must not silently
    lose whichever form is not guessed first."""
    j = tmp_path / "L1__001_foo"
    j.mkdir()
    (j / "session.json").write_text("{}")
    n = tmp_path / "L2" / "002_bar"
    n.mkdir(parents=True)
    (n / "session.json").write_text("{}")
    assert covered(tmp_path, "session.json") == {"L1__001_foo", "L2__002_bar"}


def test_flat_json_layout_unchanged(tmp_path):
    """No --pattern: <artifacts>/<Category>__<problem>.json, as artifacts/06 uses."""
    (tmp_path / "L1__001_foo.json").write_text("{}")
    (tmp_path / "L2__002_bar.json").write_text("{}")
    assert covered(tmp_path, None) == {"L1__001_foo", "L2__002_bar"}
