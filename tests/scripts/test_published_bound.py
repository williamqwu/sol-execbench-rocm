# SPDX-License-Identifier: Apache-2.0
"""Which T_SOL column a consumer reads, and why it is not a matter of taste.

A manifest carries two millisecond columns and they are different numbers.
`t_sol_ms` is a cycle count over whatever reference clock the tier that wrote it
used -- 1.8 GHz for one MI355X tier and 2.4 GHz for the other, which is D63.
`t_sol_ms_published` is the bound the manifest publishes and the one every score
is computed against, re-derived at the minimum of the T_b measurement's own
clock bracket.

Over `artifacts/09-MI355X/manifest-v4.json` the two differ on 3685 of 3717
scoreable workloads, from 0.7481x to 1.3370x, in both directions. Read the wrong
one and 147 workloads land in a different `bound_quality` band and 214 in a
different headroom band -- on the board, in the one column whose whole purpose
is to say how much a score means.

**And the tempting one-line fix is wrong in the other direction.** Both frozen
MI350X manifests carry `t_sol_ms_published` on 0 of their 3717 scoreable
workloads, and `f_ref_mhz` on none of them either. A hard switch, or a raise
from `t_sol_at.bound_ms`, would blank every headroom, every quality word and
every band on the MI350X board at once. So the fallback is not a convenience and
these tests treat losing it as the more serious regression of the two.

CPU-only. The last two tests read the real manifests, because the claim they
make is about those files and cannot be made against a fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "leaderboard"))

from bound_headroom import (  # noqa: E402
    LEGACY_AT_STATED_CLOCK,
    LEGACY_UNSTAMPED,
    NO_BOUND,
    PUBLISHED,
    published_bound_ms,
)

MI355X_V4 = ROOT / "artifacts" / "09-MI355X" / "manifest-v4.json"
MI350X = [ROOT / "artifacts" / "09" / "manifest-v1.json",
          ROOT / "artifacts" / "09" / "manifest-v1.2.json"]


def test_the_published_bound_wins_when_it_exists():
    """Both columns present and 1.333x apart -- the D63 ratio exactly."""
    w = {"t_sol_ms": 16.31118222222222, "t_sol_ms_published": 12.39346897425074,
         "f_ref_mhz": None}
    assert published_bound_ms(w) == (12.39346897425074, PUBLISHED)


def test_a_stamped_legacy_record_is_read_through_the_choke_point():
    """No published bound, but the record says which clock its own column is at,
    so `t_sol_at.bound_ms` accepts it and the read is unambiguous."""
    assert published_bound_ms({"t_sol_ms": 2.0, "f_ref_mhz": 1800.0}) == (
        2.0, LEGACY_AT_STATED_CLOCK)


def test_an_unstamped_legacy_record_degrades_rather_than_refusing():
    """The MI350X case, and the one that must never become a raise: `bound_ms`
    refuses a column with no clock, and honouring that refusal here would empty
    the MI350X board rather than protect it."""
    assert published_bound_ms({"t_sol_ms": 2.0}) == (2.0, LEGACY_UNSTAMPED)


def test_the_basis_comes_back_so_the_fallback_can_be_counted():
    """A board that cannot say how many of its rows are on an unstated clock is
    indistinguishable from a board with none."""
    assert {published_bound_ms(w)[1] for w in (
        {"t_sol_ms_published": 1.0},
        {"t_sol_ms": 1.0, "f_ref_mhz": 2400.0},
        {"t_sol_ms": 1.0},
        {})} == {PUBLISHED, LEGACY_AT_STATED_CLOCK, LEGACY_UNSTAMPED, NO_BOUND}


@pytest.mark.parametrize("w", [
    {},
    {"t_sol_ms": None},
    {"t_sol_ms": 0.0},
    {"t_sol_ms_published": 0.0, "t_sol_ms": 0.0},
    {"f_ref_mhz": 2400.0},                       # stamped, but not a bound
])
def test_nothing_usable_is_no_bound_and_not_a_zero(w):
    """`T_b / 0` is not a headroom, and a zero bound is a shape this repo has
    produced before by rounding (`t_sol_at.t_sol_cycles_at`)."""
    assert published_bound_ms(w) == (None, NO_BOUND)


def test_a_published_bound_of_zero_falls_through_rather_than_being_returned():
    """Zero is not a bound whichever column it is in, but a usable legacy column
    beside it still is."""
    assert published_bound_ms(
        {"t_sol_ms_published": 0.0, "t_sol_ms": 3.0}) == (3.0, LEGACY_UNSTAMPED)


def test_every_consumer_shares_one_definition():
    """Not four copies that agree today. Two consumers quietly disagreeing about
    which column a millisecond lives in is exactly how D63 happened, so this is
    an identity check and not an equality check."""
    import ingest
    import score_distribution

    assert ingest.published_bound_ms is published_bound_ms
    assert score_distribution.published_bound_ms is published_bound_ms


# --------------------------------------------------------------------------
# against the real manifests
# --------------------------------------------------------------------------

def _workloads(path: Path):
    doc = json.loads(path.read_text())
    for prob in doc["problems"].values():
        for w in (prob.get("workloads") or {}).values():
            if w.get("scoreable"):
                yield w


@pytest.mark.parametrize("path", MI350X, ids=lambda p: p.name)
def test_the_frozen_mi350x_manifests_still_yield_a_bound_on_every_workload(path):
    """The regression that matters most. These two files are FROZEN: they carry
    `t_sol_ms_published` on 0 of 3717 and will never gain it, so any change that
    stops falling back takes the whole MI350X board with it and shows up as
    blank columns rather than as an error."""
    if not path.exists():
        pytest.skip(f"{path.name} not in this tree")
    n = 0
    for w in _workloads(path):
        value, basis = published_bound_ms(w)
        assert basis == LEGACY_UNSTAMPED
        assert value == w["t_sol_ms"]
        n += 1
    assert n == 3717, f"{path.name} has {n} scoreable workloads, expected 3717"


def test_mi355x_v4_is_read_at_its_published_bound_and_the_two_columns_differ():
    """The other half: on the part that cannot lock a clock, reading the legacy
    column is reading a bound at a frequency the card was never at."""
    if not MI355X_V4.exists():
        pytest.skip("manifest-v4 not in this tree")
    n = differing = 0
    ratios = []
    for w in _workloads(MI355X_V4):
        value, basis = published_bound_ms(w)
        assert basis == PUBLISHED
        assert value == w["t_sol_ms_published"]
        n += 1
        if w["t_sol_ms"] and value != w["t_sol_ms"]:
            differing += 1
            ratios.append(value / w["t_sol_ms"])
    assert n == 3717
    # Measured 2026-08-15. Pinned as a floor, not as an exact count: a later
    # manifest may move it, but "the two columns are basically the same" is the
    # belief this test exists to keep from creeping back.
    assert differing >= 3000, differing
    # The band moved once already, and it moved because a defect was repaired,
    # not because the belief changed. When this was first pinned,
    # `artifacts/03-MI355X/t_sol.json` stated one tier at 1.8 GHz and the other
    # at 2.4 GHz, so published/legacy spanned 0.748 .. 1.337 -- mostly clock
    # error. That file was re-derived at a single 2400 MHz reference and
    # manifest-v4 rebuilt on it, leaving only the real term (each anchor's own
    # clock bracket against that reference): 0.946 .. 1.517 over 3685 differing
    # workloads, measured 2026-08-15.
    #
    # So it is asserted from BOTH sides now. The columns must still visibly
    # differ, or `published_bound_ms` has stopped mattering; and the spread must
    # not re-widen past what one clock bracket can explain, or a second reference
    # clock has crept back into the tier.
    lo, hi = min(ratios), max(ratios)
    assert lo < 0.99 and hi > 1.10, (lo, hi)
    assert lo > 0.85 and hi < 2.0, (lo, hi)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
