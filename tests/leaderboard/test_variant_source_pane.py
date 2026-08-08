#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A reference variant's page must show the code that variant actually ran.

The four PyTorch baselines have no `run_kernel` row -- nothing was authored --
and the "solution it proposed" pane was gated on exactly that. So every one of
880 baseline problem pages said "No kernel was recorded for this submission on
this problem", while the source sat in `variant_source`, 1,175 rows of it,
regenerated at ingest from each problem's own reference.

"Not recorded" and "not authored" are different claims and only the second was
true. The first sends a reader looking for a file that was never meant to
exist -- the same failure `depth_note` was added for.

Asserted here is the contract, not the markup. Two of the four checks guard the
ways this goes wrong *quietly*: resolving the variant from the slug rather than
from the column (which shows a different transform's real, valid code, and
looks correct), and letting the generic depth fallbacks speak for a run whose
trajectory is absent by construction rather than by omission.

Slugs are read from the fixture, never hardcoded: the fixture's variant slug is
`ref-v1-eager` where production's is `baseline-v1-eager`, which is what makes
the slug-derivation failure visible here instead of only on the real board.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

ALL_VARIANTS = ("v1_eager", "v2_compile", "v3_compile_max_autotune",
                "v4_contiguous", "v5_compile_contiguous")


def _variant_subs(client) -> list[dict]:
    subs = []
    for r in client.get("/api/v1/leaderboard").json():
        if r["kind"] != "reference_variant":
            continue
        d = client.get(f"/api/v1/submissions/{r['slug']}").json()
        subs.append({"slug": r["slug"],
                     "variant": (d.get("submission") or {}).get("variant"),
                     "problems": [p["key"] for p in d.get("problems") or []]})
    assert subs, "no reference variants on the board"
    return subs


def test_every_variant_submission_carries_its_transform_name(client):
    """The column the pane resolves through. Absent, the pane has to guess."""
    for s in _variant_subs(client):
        assert s["variant"], f"{s['slug']} does not name its variant"
        assert s["variant"] in ALL_VARIANTS, s


def test_the_variant_page_shows_the_variant_source(client):
    for s in _variant_subs(client):
        assert s["problems"], s["slug"]
        page = client.get(
            f"/submissions/{s['slug']}/problems/{s['problems'][0]}").text
        assert "No kernel was recorded" not in page, s["slug"]
        assert 'id="k-own"' in page, s["slug"]


def test_the_pane_shows_this_variant_and_not_a_neighbours(client):
    """`L1__001_alpha` has two transforms in `variant_source`. A lookup keyed on
    the problem alone, or on a slug round-trip, returns the wrong one -- and the
    wrong one is real code that renders perfectly."""
    for s in _variant_subs(client):
        page = client.get(
            f"/submissions/{s['slug']}/problems/{s['problems'][0]}").text
        assert f'data-tab="k-own">{s["variant"]}</button>' in page, s
        for other in ALL_VARIANTS:
            if other != s["variant"]:
                assert f'data-tab="k-own">{other}</button>' not in page, (s, other)


def test_the_missing_depth_is_explained_as_structural(client):
    """No trajectory and no cost are facts about what a variant IS, not gaps in
    what was captured. The generic fallbacks assert the opposite."""
    for s in _variant_subs(client):
        page = client.get(
            f"/submissions/{s['slug']}/problems/{s['problems'][0]}").text
        assert "No trajectory was recorded for this run." not in page, s["slug"]
        assert "No per-problem cost was recorded for this run." not in page, s["slug"]
        assert "zero by construction" in page, s["slug"]


def test_a_variant_that_anchors_says_so(client):
    """A transform that wins T_b on a workload scores exactly 0.5 there by
    construction -- the most misreadable number on the page, and the one a
    reader is most likely to mistake for a measured result."""
    seen = False
    for s in _variant_subs(client):
        for key in s["problems"]:
            page = client.get(f"/submissions/{s['slug']}/problems/{key}").text
            if "anchors T" in page:
                assert "0.5" in page, (s["slug"], key)
                seen = True
    assert seen, "no variant page states its own T_b anchoring"
