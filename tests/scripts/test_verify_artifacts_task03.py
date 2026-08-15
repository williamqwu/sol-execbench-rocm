# SPDX-License-Identifier: Apache-2.0
"""Task 03's two report-reading gates: check A-published, and check D.

Both of these were wrong in the same way -- they read a number that was not the
number they claimed to read -- and both were wrong for months with no test
under them.

**check A-published** searched the whole cross-checks report for
``A-published[\\s\\S]{0,4000}?\\*\\*(\\d+) VIOLATIONS``. `sol_cross_checks.py`
emitted no `VIOLATIONS` clause when A-published was clean, so on the shipped
MI355X report the window ran 3123 characters past the section, through B and C,
and matched **section D's** count. The gate reported "120 published bounds sit
below the declared-traffic floor" from a report whose own A-published line says
3688/3717 sit at or above it -- and D is the opposite error direction, on a
different tier. The first test here is that exact shape: A-published clean,
section D dirty.

**check D** keyed on the manifest's ``t_sol_ms``, which on MI355X is the
reference-clock column that the two tiers wrote at two different clocks (D63).
The naive repair -- read ``t_sol_ms_published`` instead -- is what these tests
exist to prevent: MI350X's manifests carry that field on **0 of 3717** scoreable
workloads, so the bounds map empties, `n_measured` goes to 0, and the real
144-of-7840 failure becomes a silent pass over nothing. That is the failure mode
check D's own docstring is about, so a fix that reintroduces it is worse than
the defect it fixes.

**check A-published, again, and worse.** Scoping the count to its own section
made the gate green -- and nothing bound the report to the manifest being gated.
Measured before the fix: ``--manifest manifest-v1.json`` gave ``[PASS] check
A-published`` alongside ``[FAIL] check D 54 of 1801``, and ``manifest-v2.json``
gave ``[PASS]`` alongside ``10 of 2078``; both PASSes came from a report
generated against ``manifest-v4.json``. A red gate that is wrong gets read; a
green gate that cannot fail does not. So ``sol_cross_checks.py`` now records
what it was generated from, and the gate REFUSES a report it cannot bind.

The generator's half of that contract is tested here too, in the same file as
the gate's, because the two halves are only correct with respect to each other:
one spelling of the marker, one digest, and a test on either side alone would
pass while the pair failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import sol_cross_checks as scc  # noqa: E402
import verify_artifacts as va  # noqa: E402


@pytest.fixture
def art(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setattr(va, "ART", root)
    return root


def _use(monkeypatch, part: str = va.DEFAULT_PART, manifest: str | None = None):
    monkeypatch.setattr(va, "TREE", va.ArtifactTree(part=part))
    if manifest:
        monkeypatch.setattr(va, "MANIFEST_NAME", manifest)


def _statuses(c: va.Checks, name_contains: str) -> list[tuple[str, str]]:
    return [(s, d) for s, n, d in c.results if name_contains in n]


# -- the section reader ----------------------------------------------------

REPORT_CLEAN_A_DIRTY_D = """# Task 03 — T_SOL cross-checks

## A — SOLAR's memory term vs the problem's own declared traffic

* checked: **2998** workloads
* below declared minimum: **1021**

## A-published — the bound a score is computed against, vs that floor

3688/3717 PUBLISHED workloads sit at or above their declared-traffic floor; \
29 have a floor refuted by measurement (floor > T_b); 0 not checkable

## B — rates implied by T_SOL

* implied bandwidth above DRAM peak: **0**
* implied FLOPS above the precision's peak: **0**

## C — hand-derived MAC counts

* MISMATCHes: **0**

## D — T_SOL <= best measured time

2574/2694 workloads satisfy T_SOL <= T_b — **120 VIOLATIONS**, each one a config error
"""


def test_section_stops_at_the_next_heading():
    sec = va._section(REPORT_CLEAN_A_DIRTY_D, "## A-published")
    assert "sit at or above" in sec
    assert "VIOLATIONS" not in sec
    assert "## B" not in sec


def test_section_of_the_last_section_keeps_its_last_character():
    """`text[i:text.find("\\n## ", i+1)]` is `text[i:-1]` when there is no next
    section, which drops the final character -- and the final character is
    inside the count when the section ends on one."""
    text = "## A-published — x\n\n7 VIOLATIONS**"
    assert va._section(text, "## A-published").endswith("VIOLATIONS**")


def test_section_absent_is_empty_not_the_whole_document():
    assert va._section(REPORT_CLEAN_A_DIRTY_D, "## Z-nonexistent") == ""


# -- check A-published -----------------------------------------------------

def _bind(report: str, manifest: Path | None, extra: dict | None = None) -> str:
    """*report* with the input record `sol_cross_checks.py` writes.

    Built through the generator's own `file_binding`, not by hand, so that a
    change to the record's shape breaks these tests instead of silently making
    the gate unbindable.

    *extra* adds the report's other inputs (`t_sol`, `t_sol_traffic`, `arch`),
    which the freshness gate reads. They are absent by default because most of
    these tests are about the manifest binding alone, and an input record that
    named files a test never created would fail for the wrong reason.
    """
    rec = {"manifest": scc.file_binding(manifest) if manifest
           else {"path": None, "present": False}}
    rec.update(extra or {})
    marker = f"<!-- {va.INPUTS_MARKER} {json.dumps(rec)} -->"
    head, sep, rest = report.partition("\n")
    return f"{head}{sep}\n{marker}\n{rest}"


def _task_03_tree(art, part: str, report: str,
                  bind_to: Path | None = None) -> Path:
    """The minimum task-03 tree `check_03` will read without erroring out.

    `bind_to` is the manifest the report claims to have been generated against.
    None leaves the report unbound, which is what every report written before
    the record existed looks like -- and which the gate must refuse.
    """
    d = art / ("03" if part == va.DEFAULT_PART else f"03-{part}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "t_sol.json").write_text(json.dumps({
        "_provenance": {"utc": "2026-08-15T00:00:00+00:00", "git_sha": "abc",
                        "part": part, "torch": {"devices": [f"AMD Instinct {part}"]}},
        "problems": {"L1__001_x": {"workloads": {
            "u0": {"t_sol_cycles": 100, "t_sol_ms": 0.05}}}},
    }))
    (d / "cross-checks.md").write_text(
        _bind(report, bind_to) if bind_to is not None else report)
    return d


def _bound_tree(art, report: str, part: str = va.DEFAULT_PART,
                name: str = "manifest-v1.json") -> Path:
    """A task-03 tree whose report is bound to a manifest that exists."""
    man = _manifest(art, part, name,
                    {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    _task_03_tree(art, part, report, bind_to=man)
    return man


def test_clean_a_published_does_not_capture_section_ds_count(art, monkeypatch):
    """The regression, in the exact shape that produced it.

    A-published clean and section D at 120 violations: the gate must report
    A-published as PASS. Before the section scoping it reported
    "120 published bounds sit below the declared-traffic floor", which is
    section D's number, about the SOLAR tier, in the opposite direction.
    """
    _bound_tree(art, REPORT_CLEAN_A_DIRTY_D)
    _use(monkeypatch)

    c = va.Checks()
    va.check_03(c)
    assert _statuses(c, "check A-published: ") == [(va.PASS, "")]
    # ... and section D's number is still visible to the check that owns it.
    assert "**120 VIOLATIONS" in (art / "03" / "cross-checks.md").read_text()


def test_explicit_zero_violations_is_a_pass(art, monkeypatch):
    """The new writer states the zero; "0 VIOLATIONS" must not read as a hit."""
    report = REPORT_CLEAN_A_DIRTY_D.replace(
        "0 not checkable", "0 not checkable — **0 VIOLATIONS**.")
    _bound_tree(art, report)
    _use(monkeypatch)

    c = va.Checks()
    va.check_03(c)
    assert _statuses(c, "check A-published: ") == [(va.PASS, "")]


def test_a_published_violation_in_its_own_section_still_fails(art, monkeypatch):
    """The gate must keep gating: a real count, inside the section, fails."""
    report = REPORT_CLEAN_A_DIRTY_D.replace(
        "0 not checkable",
        "0 not checkable — **7 VIOLATIONS across 2 problems**. Those bounds "
        "are below the unavoidable minimum, so their scores are inflated.")
    _bound_tree(art, report)
    _use(monkeypatch)

    c = va.Checks()
    va.check_03(c)
    (status, detail), = _statuses(c, "check A-published: ")
    assert status == va.FAIL
    assert detail.startswith("7 published bounds sit below")


def test_a_published_absent_is_a_judgement_not_a_pass(art, monkeypatch):
    """MI350X's report predates the check. Absent must not read as clean."""
    report = REPORT_CLEAN_A_DIRTY_D.replace(
        "## A-published — the bound a score is computed against, vs that floor",
        "## A-elsewhere")
    _task_03_tree(art, va.DEFAULT_PART, report)
    _use(monkeypatch)

    c = va.Checks()
    va.check_03(c)
    (status, _), = _statuses(c, "check A-published absent")
    assert status == va.JUDGE
    # No binding check either: there is no count to bind to anything.
    assert not _statuses(c, "bound to the manifest under audit")


# -- check A-published must be bound to the manifest it audits --------------

def test_a_report_generated_against_another_manifest_makes_the_gate_refuse(
        art, monkeypatch):
    """THE regression, measured: A-published PASSed for every manifest.

    On the real tree, `--manifest manifest-v1.json` and `--manifest
    manifest-v2.json` both reported `[PASS] check A-published` from a report
    generated against `manifest-v4.json`, while check D -- which reads the
    manifest directly -- failed on 54 of 1801 and 10 of 2078 respectively. Here
    the report is bound to `manifest-v4.json` and `manifest-v1.json` is under
    audit: the gate must REFUSE, not pass.
    """
    other = _manifest(art, va.DEFAULT_PART, "manifest-v4.json",
                      {"u9": {"scoreable": True, "t_sol_ms": 0.05}})
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    _task_03_tree(art, va.DEFAULT_PART, REPORT_CLEAN_A_DIRTY_D, bind_to=other)
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va.check_03(c)
    (status, detail), = _statuses(c, "bound to the manifest under audit")
    assert status == va.FAIL
    assert "manifest-v4.json" in detail and "manifest-v1.json" in detail
    assert "a different manifest entirely" in detail
    # And the count itself is refused rather than believed.
    (status, detail), = _statuses(c, "check A-published: ")
    assert status == va.FAIL
    assert detail.startswith("REFUSED")


def test_a_report_with_no_input_record_makes_the_gate_refuse(art, monkeypatch):
    """Every report written before the record exists is unbindable.

    Absence is not innocence here: an unbound report is exactly the artifact
    that passed for every manifest, so it must fail rather than be trusted.
    """
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    _task_03_tree(art, va.DEFAULT_PART, REPORT_CLEAN_A_DIRTY_D)
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va.check_03(c)
    (status, detail), = _statuses(c, "bound to the manifest under audit")
    assert status == va.FAIL
    assert va.INPUTS_MARKER in detail
    assert _statuses(c, "check A-published: ")[0][0] == va.FAIL


def test_an_unbound_report_is_refused_even_when_its_count_is_dirty(
        art, monkeypatch):
    """Unbound outranks the count.

    A "7 VIOLATIONS" read out of a report about another manifest is not a
    smaller finding than zero -- it is not a finding at all, and reporting it as
    one would send a reader to look for seven bounds that this manifest may not
    have.
    """
    report = REPORT_CLEAN_A_DIRTY_D.replace(
        "0 not checkable", "0 not checkable — **7 VIOLATIONS across 2 problems**.")
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    _task_03_tree(art, va.DEFAULT_PART, report)
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va.check_03(c)
    (status, detail), = _statuses(c, "check A-published: ")
    assert status == va.FAIL
    assert detail.startswith("REFUSED")


def test_a_manifest_rebuilt_in_place_makes_the_gate_refuse(art, monkeypatch):
    """Same path, different bytes: the report is stale evidence, and says which.

    This is the case a path comparison would miss, and it is the likely one --
    a manifest is regenerated far more often than it is renamed.
    """
    man = _bound_tree(art, REPORT_CLEAN_A_DIRTY_D)
    man.write_text(man.read_text().replace('"t_sol_ms": 0.05',
                                           '"t_sol_ms": 0.06'))
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va.check_03(c)
    (status, detail), = _statuses(c, "bound to the manifest under audit")
    assert status == va.FAIL
    assert "same filename, different bytes" in detail


def test_a_report_generated_with_no_manifest_makes_the_gate_refuse(
        art, monkeypatch):
    """`--manifest` omitted at generation: the count is about no manifest."""
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    _task_03_tree(art, va.DEFAULT_PART, _bind(REPORT_CLEAN_A_DIRTY_D, None))
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va.check_03(c)
    (status, detail), = _statuses(c, "bound to the manifest under audit")
    assert status == va.FAIL
    assert "generated with no --manifest" in detail


def test_a_changed_t_sol_tier_fails_the_gate(art, tmp_path):
    """The floor A-published compares against comes from the traffic tier.

    A stale tier is a false green the manifest binding does NOT cover: rebuild
    the tier without rebuilding the manifest and the manifest's digest is
    unchanged while the floor underneath the count has moved. This was a WARN
    for one reason -- `artifacts/03-MI355X/t_sol.json` was the known-broken
    artifact scheduled for re-derivation, and turning the part's gate red the
    moment that repair landed would have reported a fix as a regression. That
    re-derivation landed, the tier and the manifest were rebuilt on top of it
    and the report regenerated last, so nothing on the tree trips this -- which
    is the only honest moment to harden a gate.
    """
    tier = tmp_path / "t_sol_traffic.json"
    tier.write_text('{"problems": {}}')
    inputs = {"t_sol_traffic": scc.file_binding(tier),
              "t_b": {"path": "tb", "present": True, "n_files": 3}}
    c = va.Checks()
    assert va._report_inputs_are_current(c, inputs) is True
    assert _statuses(c, "inputs are the ones on disk")[0][0] == va.PASS

    tier.write_text('{"problems": {"L1__001_x": {}}}')
    c = va.Checks()
    assert va._report_inputs_are_current(c, inputs) is False
    (status, detail), = _statuses(c, "inputs are the ones on disk")
    assert status == va.FAIL
    assert "t_sol_traffic" in detail and "has changed since" in detail


def test_a_report_predating_the_input_record_is_not_judged_on_freshness():
    """No record at all is the BINDING check's case, not this one.

    Both frozen MI350X reports are in it, and they must stay there: emitting a
    freshness failure for them would turn one known defect into two on the part
    whose gate matrix is this session's regression signal.
    """
    c = va.Checks()
    assert va._report_inputs_are_current(c, None) is True
    assert _statuses(c, "inputs are the ones on disk") == []


def test_a_stale_floor_refuses_the_a_published_count(art, monkeypatch, tmp_path):
    """A clean count taken against bytes that are gone is not a clean count.

    The count was already coupled to the manifest binding; this couples it to the
    floor as well. The two refusals are worded apart so a reader knows which
    artifact to regenerate.
    """
    man = _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
                    {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    tier = tmp_path / "t_sol_traffic.json"
    tier.write_text('{"problems": {}}')
    extra = {"t_sol_traffic": scc.file_binding(tier)}
    tier.write_text('{"problems": {"L1__001_x": {}}}')   # moved after the report
    _task_03_tree(art, va.DEFAULT_PART,
                  _bind(REPORT_CLEAN_A_DIRTY_D, man, extra=extra))
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va.check_03(c)
    (status, detail), = _statuses(c, "no published bound is below")
    assert status == va.FAIL
    assert "REFUSED" in detail and "not the one on disk" in detail
    # ...and the binding itself is fine: the refusal is about the floor only.
    assert _statuses(c, "bound to the manifest under audit")[0][0] == va.PASS


def test_the_generators_own_record_binds(art, tmp_path):
    """The two halves of the contract, against each other.

    `sol_cross_checks.py` writes the record and `verify_artifacts.py` reads it.
    Either file tested alone would pass while the pair disagreed about the
    marker's spelling or the digest's meaning, so this asserts on the exact
    comment `sol_cross_checks.main` emits.
    """
    man = _manifest(art, va.DEFAULT_PART, "manifest-v9.json",
                    {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    args = argparse.Namespace(manifest=str(man), t_sol=str(man), arch=str(man),
                              t_sol_traffic=None,
                              t_b=str(tmp_path), data="data/nope")
    line = (f"<!-- {scc.INPUTS_MARKER} "
            f"{json.dumps(scc.input_bindings(args), default=str)} -->")

    inputs = va._report_inputs(f"# report\n\n{line}\n\nbody\n")
    assert inputs is not None
    ok, why = va._report_binds_manifest(inputs, man)
    assert ok, why
    # ... and the T_b tree is recorded too, so a future gate can bind it.
    assert inputs["t_b"]["present"] and inputs["t_b"]["digest_kind"]
    assert inputs["t_sol_traffic"]["present"] is False


def test_report_inputs_survives_a_report_that_has_no_record():
    assert va._report_inputs("# report\n\nno record here\n") is None
    assert va._report_inputs(
        f"<!-- {va.INPUTS_MARKER} not json -->") is None


# -- check D ---------------------------------------------------------------

def _manifest(art, part: str, name: str, workloads: dict) -> Path:
    d = art / ("09" if part == va.DEFAULT_PART else f"09-{part}")
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps({
        "_provenance": {"utc": "2026-08-15T00:00:00+00:00", "git_sha": "abc"},
        "problems": {"L1__001_x": {"workloads": workloads}},
    }))
    return p


def _scored(art, part: str, records: list[dict], run: str = "run-1") -> Path:
    """The layout `score_solutions.py` writes: 10/scores/<run>/<harness>/<problem>."""
    d = art / "10" / "scores" / run / "codex"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "L1__001_x.json"
    p.write_text(json.dumps({
        "_provenance": {"utc": "2026-08-15T00:00:00+00:00", "git_sha": "abc",
                        "part": part,
                        "torch": {"devices": [f"AMD Instinct {part}"]}},
        "problem": "L1__001_x",
        "records": records,
    }))
    return p


def test_a_manifest_without_the_published_column_still_bounds_everything(
        art, monkeypatch):
    """The blinding this fix must never reintroduce.

    MI350X's `manifest-v1.json` and `manifest-v1.2.json` carry
    `t_sol_ms_published` on 0 of 3717 scoreable workloads. Reading that field
    alone empties the bounds map, and a check that bounds nothing passes over
    nothing: today's real "144 of 7840 measured workloads are faster than
    T_SOL" would be reported as a clean sheet.
    """
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05},
               "u1": {"scoreable": True, "t_sol_ms": 0.05}})
    _scored(art, va.DEFAULT_PART, [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.02},   # beats it
        {"workload_uuid": "u1", "status": "PASSED", "t_k_ms": 0.08},
    ])
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "no measurement beats its T_SOL")
    assert status == va.FAIL
    assert detail.startswith("1 of 2 measured workloads")


def test_bound_falls_back_to_the_published_column_then_to_t_sol_ms():
    """`_bound_for` in isolation: no bracket means no re-clocking, not no bound."""
    assert va._bound_for({"t_sol_ms": 0.05}, None) == (0.05, False)
    assert va._bound_for(
        {"t_sol_ms": 0.05, "t_sol_ms_published": 0.066}, None) == (0.066, False)
    # A bracket the sampler failed on is not evidence about the clock.
    assert va._bound_for({"t_sol_ms": 0.05},
                         {"clock_bracket_sampler_error": "smi"}) == (0.05, False)


def test_bound_is_re_derived_at_the_measurements_own_clock():
    """A compute-bound record, evaluated at the bracket the measurement recorded.

    900 compute cycles at 1800 MHz is 0.0005 ms; the same 900 cycles at the
    2250-2400 MHz this measurement actually saw is 0.0004 ms. That 1.25x is
    D63, and it is the whole of the 7 phantom violations: the bound was stated
    at a reference clock nothing ran at.
    """
    w = {"t_sol_ms": 0.0005, "compute_cycles": 900, "memory_bytes": 0,
         "dram_byte_per_sec": 8.0e12}
    bound, own = va._bound_for(
        w, {"clock_before_mhz": 2400, "clock_after_mhz": 2250})
    assert own is True
    assert bound == pytest.approx(900 / (2250 * 1e3))     # the MINIMUM-clock end


def test_check_d_clears_a_clock_phantom_and_keeps_a_real_violation(
        art, monkeypatch):
    """The MI355X result, in miniature: 2 reported against `t_sol_ms`, 1 real.

    `u0` is 1.28x SLOWER than a bound evaluated at its own bracket and only
    looks fast against the 1.8 GHz column. `u1` beats the bound at every clock
    in its bracket. A gate that reports both is over-reporting; one that reports
    neither has been silenced.
    """
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "t_sol_ms": 0.0005, "t_sol_ms_published": 0.0004,
               "compute_cycles": 900, "memory_bytes": 0,
               "dram_byte_per_sec": 8.0e12},
        "u1": {"scoreable": True, "t_sol_ms": 0.0005, "t_sol_ms_published": 0.0004,
               "compute_cycles": 900, "memory_bytes": 0,
               "dram_byte_per_sec": 8.0e12},
    })
    bracket = {"clock_before_mhz": 2400, "clock_after_mhz": 2250}
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.00045,
         "clock_bracket": bracket},
        {"workload_uuid": "u1", "status": "PASSED", "t_k_ms": 0.00030,
         "clock_bracket": bracket},
    ])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "no measurement beats its T_SOL")
    assert status == va.FAIL
    assert detail.startswith("1 of 2 measured workloads")
    assert "0.75x" in detail                     # 0.00030 / (900/2250e3)


def test_check_d_says_how_many_bounds_it_re_clocked(art, monkeypatch):
    """A count against the reference-clock column and one against the
    measurement's own clock are different numbers on an unlocked part, so the
    passing line has to say which one it is."""
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "t_sol_ms": 0.0005, "compute_cycles": 900,
               "memory_bytes": 0, "dram_byte_per_sec": 8.0e12}})
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.9,
         "clock_bracket": {"clock_before_mhz": 2400, "clock_after_mhz": 2250}}])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "no measurement beats its T_SOL")
    assert status == va.PASS
    assert "1 of them against a bound re-derived at the measurement's own" in detail


def test_check_d_reports_the_legacy_column_it_no_longer_reads(art, monkeypatch):
    """Check D moved off `t_sol_ms`; nothing else did. It must still be counted.

    `u0` beats the stored 1.8 GHz column and not the bound re-derived at its own
    bracket -- the D63 shape. The gate is right to pass it and wrong to say
    nothing, because `leaderboard/ingest.py`, `app.py`, `bound_headroom.py` and
    `score_distribution.py` all still read that column raw.
    """
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "t_sol_ms": 0.0005, "compute_cycles": 900,
               "memory_bytes": 0, "dram_byte_per_sec": 8.0e12}})
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.00045,
         "clock_bracket": {"clock_before_mhz": 2400, "clock_after_mhz": 2250}}])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    assert _statuses(c, "no measurement beats its T_SOL")[0][0] == va.PASS
    (status, detail), = _statuses(c, "legacy `t_sol_ms` column")
    assert status == va.WARN
    assert detail.startswith("1 of 1 measurements beat")
    assert "against 0 beating" in detail
    assert "1 of 1 records carry no f_ref_mhz" in detail
    assert "bound_headroom.published_bound_ms" in detail


def test_the_legacy_column_report_is_not_a_second_alarm_on_one_defect(
        art, monkeypatch):
    """On a locked part the column IS the bound, so it must not read as news."""
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    _scored(art, va.DEFAULT_PART,
            [{"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.02}])
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "legacy `t_sol_ms` column")
    assert status == va.PASS
    assert detail.startswith("identical to check D's own count (1 of 1)")


# -- check D-terms: the published column vs the terms it claims to state ----

def test_a_poisoned_published_ms_column_is_caught(art, monkeypatch):
    """The hole check D left when it started re-deriving from terms.

    The auditor multiplied `t_sol_ms`, `t_sol_ms_published` and
    `t_sol_ms_at_clock_*` by 100 on a scratch manifest and check D stayed at
    "1 of 2078"; multiplying the TERMS by 100 moved it to "3 of 2078". So the
    columns themselves are unvalidated, and the uncovered direction is a column
    too LARGE for its own terms -- A-published covers too small.
    """
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "compute_cycles": 900, "memory_bytes": 0,
               "dram_byte_per_sec": 8.0e12, "t_sol_published_at_mhz": 2250,
               "t_sol_ms_published": 0.0004 * 100, "t_sol_ms": 0.0005,
               "f_ref_mhz": 1800}})
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.9,
         "clock_bracket": {"clock_before_mhz": 2400, "clock_after_mhz": 2250}}])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    assert _statuses(c, "no measurement beats its T_SOL")[0][0] == va.PASS
    (status, detail), = _statuses(c, "check D-terms")
    assert status == va.FAIL
    assert "1 of 1 t_sol_ms_published" in detail
    assert "0.04 ms where its terms give 0.0004 ms at 2250 MHz" in detail


def test_the_legacy_column_is_checked_against_its_own_stamped_clock(
        art, monkeypatch):
    """`f_ref_mhz` is what makes `t_sol_ms` legible; where it is there, use it."""
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "compute_cycles": 900, "memory_bytes": 0,
               "dram_byte_per_sec": 8.0e12, "t_sol_published_at_mhz": 2250,
               "t_sol_ms_published": 0.0004, "t_sol_ms": 0.0005 * 100,
               "f_ref_mhz": 1800}})
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.9,
         "clock_bracket": {"clock_before_mhz": 2400, "clock_after_mhz": 2250}}])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "check D-terms")
    assert status == va.FAIL
    assert "1 of 1 t_sol_ms disagree" in detail


def test_one_cycle_of_rounding_in_the_legacy_column_is_not_a_failure(
        art, monkeypatch):
    """`t_sol_ms` is `int(cycles) / f_ref`; the terms are not quantised.

    `FlashInfer-Bench__016/1cf13773` moves 631,056 B at 7.99992e12 B/s = 189.32
    cycles at 2400 MHz, stored as 189, so the column states 7.875e-05 ms where
    the terms give 7.9166667e-05 -- exactly one cycle, and 0.53% relative only
    because the bound is 189 cycles long. Five of manifest-v4's 3717 records are
    in that state and every one is exactly one cycle out. Judged at float noise,
    rounding reads as corruption.

    This test uses those real numbers so it fails if the allowance is removed
    OR widened past a cycle.
    """
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "compute_cycles": 18.0625,
               "memory_bytes": 631056, "dram_byte_per_sec": 7999920000000.0,
               "t_sol_published_at_mhz": 2400,
               "t_sol_ms_published": 190 / 2.4e6,
               "t_sol_ms": 189 / 2.4e6, "t_sol_cycles": 189,
               "f_ref_mhz": 2400}})
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.9,
         "clock_bracket": {"clock_before_mhz": 2400, "clock_after_mhz": 2400}}])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "check D-terms")
    assert status == va.PASS, detail
    assert "one cycle at their own stated clock" in detail


def test_two_cycles_of_error_in_the_legacy_column_still_fails(
        art, monkeypatch):
    """The allowance is one cycle, not "small".

    A column written by something else is out by orders of magnitude, but the
    boundary is what makes this a detector rather than a tolerance, so it is
    pinned from both sides: the same record two cycles out must fail.
    """
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "compute_cycles": 18.0625,
               "memory_bytes": 631056, "dram_byte_per_sec": 7999920000000.0,
               "t_sol_published_at_mhz": 2400,
               "t_sol_ms_published": 190 / 2.4e6,
               # 188 cycles at 2400 MHz: two cycles below the 190 the terms
               # give, one more than rounding can explain.
               "t_sol_ms": 188 / 2.4e6, "t_sol_cycles": 188,
               "f_ref_mhz": 2400}})
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.9,
         "clock_bracket": {"clock_before_mhz": 2400, "clock_after_mhz": 2400}}])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "check D-terms")
    assert status == va.FAIL
    assert "1 of 1 t_sol_ms disagree" in detail


def test_the_published_column_gets_no_cycle_slack(art, monkeypatch):
    """Only the quantised column is quantised.

    `t_sol_ms_published` is evaluated from the terms directly and reproduces on
    3717 of 3717 MI355X records at 1e-9. Extending the cycle allowance to it
    would blunt the one check that watches it, so a one-cycle error there is
    still a failure.
    """
    _manifest(art, "MI355X", "manifest-v3.json", {
        "u0": {"scoreable": True, "compute_cycles": 18.0625,
               "memory_bytes": 631056, "dram_byte_per_sec": 7999920000000.0,
               "t_sol_published_at_mhz": 2400,
               "t_sol_ms_published": 189 / 2.4e6,     # one cycle low
               "t_sol_ms": 190 / 2.4e6, "f_ref_mhz": 2400}})
    _scored(art, "MI355X", [
        {"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.9,
         "clock_bracket": {"clock_before_mhz": 2400, "clock_after_mhz": 2400}}])
    _use(monkeypatch, part="MI355X", manifest="manifest-v3.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "check D-terms")
    assert status == va.FAIL
    assert "1 of 1 t_sol_ms_published" in detail


def test_check_d_terms_skips_mi350x_loudly_rather_than_passing(
        art, monkeypatch):
    """MI350X's frozen manifests carry the terms on 0 of 3717 workloads.

    A skip reported as PASS is the failure mode this whole session was spent
    removing, so it reports WARN and says what made it inapplicable.
    """
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    _scored(art, va.DEFAULT_PART,
            [{"workload_uuid": "u0", "status": "PASSED", "t_k_ms": 0.9}])
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "check D-terms")
    assert status == va.WARN
    assert detail.startswith("not evaluable: 0 of 1 scoreable workloads")
    assert "Skipped, not passed" in detail


# -- the generator: section D, re-derived at the anchor's own bracket -------

def _mini_report(tmp_path, winner: dict, *, t_sol_ms: float = 0.0005) -> str:
    """Run `sol_cross_checks.main()` over a two-file tree and return the report.

    Section D is the only section this exercises: `--data` points at an empty
    directory, so A/B skip every workload for want of a definition, and C's
    hand-derived problems are absent.
    """
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "tb").mkdir(exist_ok=True)
    (tmp_path / "arch.yaml").write_text(
        "freq_GHz: 2.4\nDRAM_byte_per_cycle: 3333.3\n"
        "MAC_per_cycle_bf16_tc: 1024\n")
    (tmp_path / "t_sol.json").write_text(json.dumps({
        "_provenance": {"utc": "2026-08-15T00:00:00+00:00", "git_sha": "abc"},
        "problems": {"L1__001_x": {"precision": "bf16", "workloads": {
            "u0": {"t_sol_cycles": 900, "t_sol_ms": t_sol_ms,
                   "compute_cycles": 900, "memory_bytes": 0,
                   "dram_byte_per_sec": 8.0e12}}}}}))
    (tmp_path / "tb" / "L1__001_x.json").write_text(json.dumps({
        "problem": "L1__001_x", "winner_by_workload": {"u0": winner}}))
    out = tmp_path / "xc.md"
    sys.argv = ["sol_cross_checks.py",
                "--t-sol", str(tmp_path / "t_sol.json"),
                "--arch", str(tmp_path / "arch.yaml"),
                "--data", str(tmp_path / "data"),
                "--t-b", str(tmp_path / "tb"),
                "--out", str(out)]
    scc.main()
    return out.read_text()


def test_section_d_re_derives_at_the_anchors_own_bracket(tmp_path):
    """The 120 published violations that were clock arithmetic.

    900 cycles at the stored 1.8 GHz reference is 0.0005 ms and loses to a
    0.00042 ms anchor; the same 900 cycles at the 2250 MHz minimum of the
    bracket that anchor was measured in is 0.0004 ms and does not. Re-derived
    over the real tree, section D's count goes 120 -> 0 on 2694 of 2694
    workloads, with 0 falling back -- and the artifact went on publishing "120
    VIOLATIONS, each one a config error" after the session had fixed them.
    """
    text = _mini_report(tmp_path, {"t_b_ms": 0.00042, "variant": "v1_eager",
                                   "clock_before_mhz": 2400,
                                   "clock_after_mhz": 2250})
    d = va._section(text, "## D — T_SOL <= best measured time")
    assert "1/1 workloads satisfy" in d
    assert "VIOLATIONS" not in d
    assert "re-derived at each anchor's own clock bracket" in d


def test_section_d_still_reports_a_real_violation(tmp_path):
    """Re-clocking is not silencing: an anchor faster than the bound at its own
    minimum clock is still a violation, and still named."""
    text = _mini_report(tmp_path, {"t_b_ms": 0.0003, "variant": "v1_eager",
                                   "clock_before_mhz": 2400,
                                   "clock_after_mhz": 2250})
    d = va._section(text, "## D — T_SOL <= best measured time")
    assert "**1 VIOLATIONS**" in d
    assert "0.0004" in d                       # the re-derived bound, not 0.0005


def test_section_d_refuses_an_anchor_with_no_bracket(tmp_path):
    """No bracket means NOT CHECKABLE, never a fallback to the stored column.

    The fallback is the D63 read this correction exists to retire, and it would
    reappear exactly on the records least able to survive it.
    """
    text = _mini_report(tmp_path, {"t_b_ms": 0.00042, "variant": "v1_eager",
                                   "clock_bracket_sampler_error": "smi"})
    d = va._section(text, "## D — T_SOL <= best measured time")
    assert "0/0 workloads satisfy" in d
    assert "1 anchors carry no usable bracket" in d
    assert "VIOLATIONS" not in d


def test_the_report_records_what_it_was_generated_from(tmp_path):
    """`main()` emits the binding record, and `verify_artifacts` can read it."""
    text = _mini_report(tmp_path, {"t_b_ms": 0.00042, "variant": "v1_eager",
                                   "clock_before_mhz": 2400,
                                   "clock_after_mhz": 2250})
    inputs = va._report_inputs(text)
    assert inputs is not None
    assert inputs["manifest"]["present"] is False        # none was passed
    assert inputs["t_sol"]["sha256"] and inputs["t_b"]["n_files"] == 1
    # ... and with no manifest the A-published section says so rather than
    # implying a claim.
    ok, why = va._report_binds_manifest(inputs, tmp_path / "t_sol.json")
    assert not ok and "generated with no --manifest" in why


def test_check_d_reads_the_older_scored_json_layout_too(art, monkeypatch):
    """`artifacts/10/<run>/scored.json` is where MI350X's 144 come from."""
    _manifest(art, va.DEFAULT_PART, "manifest-v1.json",
              {"u0": {"scoreable": True, "t_sol_ms": 0.05}})
    d = art / "10" / "old-run"
    d.mkdir(parents=True)
    (d / "scored.json").write_text(json.dumps({
        "results": [{"problem": "L1__001_x", "workload_uuid": "u0",
                     "status": "PASSED", "latency_ms": 0.01}]}))
    _use(monkeypatch, manifest="manifest-v1.json")

    c = va.Checks()
    va._check_d(c)
    (status, detail), = _statuses(c, "no measurement beats its T_SOL")
    assert status == va.FAIL
    assert detail.startswith("1 of 1 measured workloads")
