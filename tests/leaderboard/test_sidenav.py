#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Every entry in the section nav must point at a heading that exists.

The nav is authored in `app.py` (TOC_METHODOLOGY, TOC_PROBLEM) and the ids it
points at are authored in the templates. Two files, no compiler between them --
so renaming a heading, or deleting a section, leaves a link that scrolls
nowhere. It fails silently in the worst way available: the link still renders,
still looks live, and clicking it does nothing at all. Nobody files that.

The reverse direction is checked too, but only as a warning-shaped assertion on
`h2`: a new section that nobody added to the nav is a smaller defect than a
nav entry that lies, but it is the same drift and it is why the nav stops being
trustworthy after a few months.

What is NOT asserted here: that the nav is *visible*, that the sticky column
positions correctly, or that the scroll spy marks the right entry. All three
are CSS and JS with no runtime on this node. What is asserted is the part that
has to be right in the served HTML for any of that to have something to act on.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

IDS = re.compile(r'\sid="([^"]+)"')
TOC_LINKS = re.compile(r'<aside class="sidenav">(.*?)</aside>', re.S)
HREFS = re.compile(r'href="#([^"]+)"')
H2 = re.compile(r'<h2(?:\s[^>]*)?>', re.S)
H2_ID = re.compile(r'<h2\s+id="([^"]+)"')


def _page(client, url: str) -> str:
    r = client.get(url)
    assert r.status_code == 200, (url, r.status_code)
    return r.text


def _nav_targets(page: str) -> list[str]:
    m = TOC_LINKS.search(page)
    assert m, "no .sidenav rendered"
    return HREFS.findall(m.group(1))


def _body_ids(page: str) -> set[str]:
    # Everything after the nav, so a nav entry cannot satisfy itself.
    body = page.split("</aside>", 1)[1]
    return set(IDS.findall(body))


@pytest.mark.parametrize("url", ["/methodology", "/problems/L1__001_alpha"])
def test_every_nav_entry_has_a_heading(client, url):
    page = _page(client, url)
    targets = _nav_targets(page)
    assert targets, f"{url}: nav rendered with no entries"
    missing = [t for t in targets if t not in _body_ids(page)]
    assert not missing, f"{url}: nav points at ids that do not exist: {missing}"


@pytest.mark.parametrize("url", ["/methodology", "/problems/L1__001_alpha"])
def test_the_nav_is_in_reading_order(client, url):
    """The spy marks "the last heading seen above the fold", which is only the
    section you are reading if the nav lists them in document order."""
    page = _page(client, url)
    body = page.split("</aside>", 1)[1]
    pos = {i: body.index(f'id="{i}"') for i in _nav_targets(page)}
    order = list(pos)
    assert order == sorted(order, key=pos.get), \
        f"{url}: nav order {order} is not document order"


@pytest.mark.parametrize("url", ["/methodology", "/problems/L1__001_alpha"])
def test_no_section_is_missing_from_the_nav(client, url):
    """An h2 with no nav entry is a section the reader cannot jump to."""
    body = _page(client, url).split("</aside>", 1)[1]
    listed = set(_nav_targets(_page(client, url)))
    orphans = [h for h in H2.findall(body)
               if not (m := H2_ID.match(h)) or m.group(1) not in listed]
    assert not orphans, f"{url}: h2 not reachable from the nav: {orphans}"


def test_a_page_without_a_toc_keeps_one_column(client):
    """The grid is opt-in. `/problems` and `/` pass no `toc` and must render
    neither the aside nor the class that turns on the two-column layout."""
    for url in ("/", "/problems"):
        page = _page(client, url)
        assert '<aside class="sidenav">' not in page, url
        assert 'class="wrap main"' in page, url
