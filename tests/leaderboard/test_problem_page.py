#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The problem page answers the reader's questions in the order they ask them.

It used to open with eleven columns of bound derivation — T_SOL in cycles,
which of two derivations won, the bottleneck, the T_b variant, two tolerances —
above a per-workload results table of up to 288 rows, and only then say what
the kernel computes, what it takes and what it returns. Everything on it was
true; almost none of it was what a reader had come for.

What is asserted here is the reading order and the things that would silently
regress: that the sections are in that order and each is in the nav, that the
workload table carries the two bounds a score is defined between, that
NVIDIA's B200 overlay never appears without the paragraph that says what it is
not, that the evidence table is collapsed, and that a submission's name links
to what that submission did HERE.

Not asserted: the filter boxes and the column switches doing anything. They are
JavaScript and there is no runtime on this node. What is asserted is that the
page degrades correctly without them — every row and every column present in
the served HTML.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

KEY = "L1__001_alpha"


@pytest.fixture()
def page(client) -> str:
    r = client.get(f"/problems/{KEY}")
    assert r.status_code == 200
    return r.text


def h2_ids(page: str) -> list[str]:
    return re.findall(r'<h2 id="([^"]+)"', page)


def test_the_page_says_what_the_kernel_is_before_how_it_is_bounded(page):
    assert h2_ids(page) == ["what", "workloads", "reference", "submissions",
                            "results"]


def test_every_section_is_in_the_nav(client, page):
    nav = page.split("</aside>", 1)[0]
    for i in h2_ids(page):
        assert f'href="#{i}"' in nav, i


def test_inputs_outputs_and_axes_are_in_the_first_section(page):
    what = page.split('id="what"', 1)[1].split('id="workloads"', 1)[0]
    for heading in ("Inputs", "Outputs", "Axes"):
        assert f"<h3>{heading}</h3>" in what, heading


def test_the_reference_pane_asks_for_python_highlighting(page):
    assert '<pre class="code" data-lang="python">' in page


def test_the_workload_table_leads_with_the_two_bounds(page):
    """T_b and T_SOL are what a score is defined between; they are the columns
    that are always shown, and in that order."""
    head = page.split('id="wl-table"', 1)[1].split("</thead>", 1)[0]
    cols = [" ".join(re.sub(r"<[^>]+>", "", c).split()) for c in
            re.findall(r"<th[^>]*>(.*?)</th>", head, re.S)]
    assert cols[:5] == ["#", "id", "parameters", "Tb baseline (ms)",
                        "TSOL (ms)"]
    # And the derivation columns are still on the same rows, marked so the
    # switch can hide them -- not moved to a second table.
    assert head.count('class="r c-deriv"') + head.count('class="c-deriv"') >= 6


def test_a_workloads_parameters_actually_render(page):
    """`axes_json` was `{}` for all 3,957 workloads until the ingest read the
    dataset's own workload.jsonl, so this column rendered empty everywhere."""
    body = page.split('id="wl-table"', 1)[1].split("</table>", 1)[0]
    assert re.search(r'<span class="ax">\w+=', body), "no axis chips"


def test_the_derivation_band_has_a_caption_of_its_own(page):
    """The band explains what T_SOL (cyc), bound source, bottleneck, T_b
    variant, atol and rtol each mean — the question the columns raised."""
    note = page.split('class="sub colnote note-deriv"', 1)
    assert len(note) == 2, "derivation columns with no caption"
    note = note[1].split("</p>", 1)[0]
    for term in ("ound source", "ottleneck", "variant", "atol", "rtol",
                 "cyc"):
        assert term in note, term


def test_the_b200_overlay_never_appears_without_its_caveat(page):
    """The columns are another vendor's numbers on another part. If they can be
    shown at all, the paragraph saying they are not a comparison and never seed
    an AMD bound must be on the page with them."""
    assert 'class="r mono c-b200"' in page
    note = page.split('class="sub colnote note-b200"', 1)
    assert len(note) == 2, "B200 columns rendered with no caveat paragraph"
    note = note[1].split("</p>", 1)[0]
    assert "Not a comparison" in note
    assert "never seeds or checks" in note
    # Hidden until the switch turns the columns on, so the caveat cannot read
    # as a warning about the AMD numbers beside it.
    assert note.lstrip().startswith("hidden") or " hidden" in note[:40]


def test_the_evidence_table_is_collapsed_and_counted(page):
    after = page.split('<h2 id="results">Per-workload results</h2>', 1)[1]
    assert after.lstrip().startswith("<details"), after[:80]
    summary = re.search(r"<summary[^>]*>(.*?)</summary>", after, re.S).group(1)
    assert re.search(r"\d+ rows?", summary), summary
    # The heading and its anchor stay OUTSIDE the collapsed block, or the
    # section nav lands on nothing.
    assert '<h2 id="results">' in page.split("<details", 1)[0]


def test_a_submission_name_links_to_what_it_did_here(page):
    """Not to the run's own overview. A reader on a problem page clicking a
    submission is asking what that submission did on this problem."""
    table = page.split('id="submissions"', 1)[1].split('id="results"', 1)[0]
    hrefs = set(re.findall(r'href="([^"]*/submissions/[^"]*)"', table))
    assert hrefs, "no submission links"
    for h in hrefs:
        assert f"/problems/{KEY}" in h, h


def test_a_status_is_shown_in_words_and_keeps_its_enum(client):
    """`INCORRECT_NUMERICAL` and `REWARD_HACK` were printed raw, in a column
    with no key anywhere on the site."""
    body = client.get(f"/problems/{KEY}").text
    hits = re.findall(r'<span\s+class="bad" title="([A-Z_]+)">([^<]+)</span>',
                      body)
    assert hits, "no failing status rendered on the fixture problem"
    for enum, shown in hits:
        assert shown != enum
        assert shown == shown.lower()


def test_reward_hacking_is_named_not_abbreviated(client):
    """"FLAGGED"/"REWARD_HACK" is an accusation; it should read as one."""
    for url in (f"/problems/{KEY}",
                f"/submissions/agent-alpha/problems/{KEY}"):
        body = client.get(url).text
        assert ">flagged<" not in body, url
    assert "reward hacks" in client.get(f"/problems/{KEY}").text
