#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A mean must carry the count it was divided by.

`AVG(score)` skips NULL, and a PASSED result stores NULL when the kernel ran
faster than T_SOL -- the bound is invalid there, so no score is defined. The
consequence is that AVG's denominator varies per row, and three tables on this
site put such rows side by side under one heading.

The real instance is `agent-pilot8` on
`FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`: 38 of 38
workloads PASSED, 25 of them beat a bound that D18 already records as
over-counting paged traffic. Its mean is therefore over 13 workloads and reads
0.9899, printed directly above `agent-glm-run1`'s 0.9430 over all 38 -- in a
sortable column, so the run with a quarter of the evidence sorts to the top.

Nothing here asserts a *value*. Both means are arithmetically correct and both
stay exactly as they were; what is tested is that the page says which set each
one covers, and says it on every row rather than only on the odd one out. A
denominator shown only when it differs cannot be distinguished from a
denominator nobody checked.

The fixture board carries the same shape deliberately: `agent-trial-a` on
`L1__001_alpha` passes a1, a2 and a3, and a3 is the bound-invalid row.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

PROBLEM = "L1__001_alpha"
SHORT = "agent-trial-a"      # 3 passed, 2 scored -- a3's bound was beaten
FULL = "agent-alpha"         # 3 passed, 3 scored


def _denominators(html: str) -> list[str]:
    return [m.strip() for m in re.findall(r'class="sub den[^"]*"[^>]*>(.*?)</span>',
                                          html, re.S)]


def test_the_run_page_prints_a_denominator_on_every_peer_row(client):
    """Every row, not just the short one. The reader has to be able to tell
    "over 38" apart from "nobody said", and that is only possible if the
    ordinary case is also labelled."""
    html = client.get(f"/submissions/{FULL}/problems/{PROBLEM}").text
    dens = _denominators(html)
    # One per peer with a mean; the fixture has four runs with results here.
    assert len(dens) >= 3, dens
    assert all(d.startswith("over ") for d in dens), dens


def test_a_beaten_bound_is_named_where_it_shrank_the_average(client):
    """The short row says how many workloads left the average and why, so the
    caveat travels with the number rather than living in a methodology page."""
    html = client.get(f"/submissions/{SHORT}/problems/{PROBLEM}").text
    short = [d for d in _denominators(html) if "den-short" in html and "of" in d]
    assert any("over 2 of 3" in d for d in short), _denominators(html)
    assert "bound beaten on 1" in html
    # And the explanation attributes it to the bound, not to the kernel.
    assert "faster than T_SOL" in html or "beat T_SOL" in html


def test_a_full_row_is_not_marked_short(client):
    """`den-short` is a warning. Firing it on a row where nothing was excluded
    would train the reader to ignore it."""
    html = client.get(f"/submissions/{FULL}/problems/{PROBLEM}").text
    row = re.search(rf'<tr class="self">.*?</tr>', html, re.S)
    assert row and "den-short" not in row.group(0), row


def test_the_problem_page_and_submission_page_agree(client):
    """The same statistic appears on three pages. All three divide by the same
    count, or one of them is quietly ranking 2 workloads against 3."""
    for url in (f"/problems/{PROBLEM}", f"/submissions/{SHORT}"):
        html = client.get(url).text
        assert "over 2 of 3" in html, (url, _denominators(html))


def test_the_trial_switcher_says_why_a_clean_sweep_scores_low(client):
    """`3/4 passed - mean S 0.47` with no explanation reads as a broken mean.
    The switcher scores a beaten bound as zero (it must -- the alternative is
    to drop the workload and inflate the run), so it has to say so."""
    html = client.get(f"/submissions/{SHORT}/problems/{PROBLEM}").text
    assert "bound" in html and "scored 0" in html
    assert 'class="t-warn"' in html


def test_no_score_changed(client):
    """The whole point: this is a labelling change. The means themselves are
    the ones the API already served."""
    api = client.get(f"/api/v1/problems/{PROBLEM}").json()
    per = {s["slug"]: s["mean_score"] for s in api["submissions"]}
    assert per[SHORT] == pytest.approx(0.94)      # (0.98 + 0.90) / 2, not / 3
    assert per[FULL] == pytest.approx((0.60 + 0.55 + 0.52) / 3)
