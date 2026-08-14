# SPDX-License-Identifier: Apache-2.0
"""Task 01's acceptance, on a part whose clock cannot be pinned.

`check_01` as written presupposes a lock at every step, and on MI355X
`get_clock_preset(...).f_lock_mhz` is `None` by design — so it does not merely
fail, it reports "no preset for this device" about an entry that exists and
cannot pass however well the node behaves. The maintainer's replacement is that
the **clock basis is characterised**: the per-card distribution under sustained
load is recorded, the eight-card spread is inside a stated band, and the bracket
refusal rate is below a stated bound.

What these tests mostly guard is the difference between *unmeasured* and *fine*.
A gate that reports a missing refusal rate as a pass would certify an unlocked
node on the strength of a file nobody wrote.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import verify_artifacts as va  # noqa: E402


def _dist(spread: float, n_cards: int = 8, host: str = "mia1-p02-g46") -> dict:
    base = 1800.0
    med = [base * (1 - spread / 2), *([base] * (n_cards - 2)),
           base * (1 + spread / 2)][:n_cards]
    return {"_provenance": {"host": host},
            "blocks": [{"label": "gemm_dense, all 8 loaded",
                        "per_gpu": {str(i): {"clock_median_mhz": m}
                                    for i, m in enumerate(med)}}]}


def _run(tmp_path, monkeypatch, dist=None, tb: dict | None = None) -> list[tuple]:
    art = tmp_path / "artifacts"
    (art / "01").mkdir(parents=True, exist_ok=True)
    if dist is not None:
        (art / "01" / "unlocked-clock.json").write_text(json.dumps(dist))
    if tb is not None:
        d = art / "06" / "authoritative"
        d.mkdir(parents=True, exist_ok=True)
        for name, doc in tb.items():
            (d / name).write_text(json.dumps(doc))
    monkeypatch.setattr(va, "ART", art)
    c = va.Checks()
    va._check_01_unlocked(c)
    return c.results


def _named(results, needle):
    return [r for r in results if needle in r[1]]


def test_a_missing_distribution_fails_rather_than_being_skipped(tmp_path, monkeypatch):
    """Unlocked, this artifact IS the clock calibration. Its absence is the
    failure, not a reason to check nothing."""
    r = _run(tmp_path, monkeypatch, dist=None)
    assert r[0][0] == va.FAIL and "per-card clock distribution" in r[0][1]
    assert len(r) == 1, "nothing downstream may report on data that is not there"


def test_a_spread_inside_the_band_passes(tmp_path, monkeypatch):
    """6.46% is g46's measured eight-card spread (1739-1855 MHz)."""
    r = _named(_run(tmp_path, monkeypatch, _dist(0.0646)), "eight-card clock spread")
    assert r[0][0] == va.PASS


def test_a_spread_outside_the_band_fails(tmp_path, monkeypatch):
    r = _named(_run(tmp_path, monkeypatch, _dist(0.15)), "eight-card clock spread")
    assert r[0][0] == va.FAIL


def test_the_band_is_above_both_measured_nodes():
    """5.23%/5.28% on g10, 6.46% on g46. A band below either would fail a node
    the benchmark was characterised on; one far above would check nothing."""
    assert 0.0646 < va.MAX_EIGHT_CARD_CLOCK_SPREAD < 0.10


def test_fewer_than_eight_cards_cannot_produce_a_spread(tmp_path, monkeypatch):
    """One card at a time says nothing about card-to-card variation, and
    averaging separate single-card blocks would look like an answer."""
    r = _named(_run(tmp_path, monkeypatch, _dist(0.05, n_cards=4)),
               "all eight cards sampled")
    assert r[0][0] == va.FAIL


def test_the_host_is_surfaced_for_judgement(tmp_path, monkeypatch):
    """A distribution from another MI355X node describes that node's chassis.
    The gate cannot decide that; it must not hide it either."""
    r = _named(_run(tmp_path, monkeypatch, _dist(0.05, host="mia1-p02-g10")),
               "THIS node's")
    assert r[0][0] == va.JUDGE and "mia1-p02-g10" in r[0][2]


def test_an_unmeasured_refusal_rate_is_not_a_low_one(tmp_path, monkeypatch):
    """The failure mode that would matter most: certifying a node on the
    strength of a statistic nobody computed."""
    r = _named(_run(tmp_path, monkeypatch, _dist(0.05), tb={}), "refusal rate")
    assert r[0][0] == va.JUDGE and "has not been measured" in r[0][2]


def test_a_low_refusal_rate_passes_and_a_high_one_fails(tmp_path, monkeypatch):
    low = {"a.json": {"clock_bracket_summary": {"n_bracketed": 1000,
                                                "n_refused": 5,
                                                "refusal_rate": 0.005}}}
    high = {"a.json": {"clock_bracket_summary": {"n_bracketed": 1000,
                                                 "n_refused": 300,
                                                 "refusal_rate": 0.3}}}
    assert _named(_run(tmp_path, monkeypatch, _dist(0.05), tb=low),
                  "refusal rate")[0][0] == va.PASS
    r = _named(_run(tmp_path, monkeypatch, _dist(0.05), tb=high), "refusal rate")
    assert r[0][0] == va.FAIL
    assert "re-derived" in r[0][2], \
        "the remedy for a high rate is re-deriving the threshold, not raising it"


def test_the_rate_is_pooled_across_artifacts_not_averaged(tmp_path, monkeypatch):
    """Averaging per-problem rates would let a one-workload problem with a single
    refusal outweigh a thousand clean measurements."""
    tb = {"a.json": {"clock_bracket_summary": {"n_bracketed": 999, "n_refused": 0,
                                               "refusal_rate": 0.0}},
          "b.json": {"clock_bracket_summary": {"n_bracketed": 1, "n_refused": 1,
                                               "refusal_rate": 1.0}}}
    r = _named(_run(tmp_path, monkeypatch, _dist(0.05), tb=tb), "refusal rate")
    assert r[0][0] == va.PASS and "1/1000" in r[0][2]


def test_the_locked_basis_gate_is_untouched(monkeypatch):
    """MI350X must still be adjudicated by the lock checks, unchanged."""
    monkeypatch.delenv("SOLEXBENCH_CLOCK_BASIS", raising=False)
    c = va.Checks()
    va.check_01(c)
    names = [n for _, n, _ in c.results]
    assert "F_LOCK present in CLOCK_LOCK_PRESETS" in names
    assert not any("eight-card clock spread" in n for n in names)


def test_the_unlocked_basis_drops_the_lock_checks(monkeypatch):
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "unlocked")
    c = va.Checks()
    va.check_01(c)
    names = [n for _, n, _ in c.results]
    assert "F_LOCK present in CLOCK_LOCK_PRESETS" not in names
    assert "every GPU is at the preset's determinism setpoint" not in names
    assert "F_LOCK at or below lowest observed floor" not in names
    # ...but the checks that are about the NODE rather than the lock stay.
    for kept in ("clock floor sampled on >=3 GPUs", "stability measured",
                 "sibling interference measured"):
        assert kept in names, kept
