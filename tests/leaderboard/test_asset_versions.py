#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static assets must be cache-busted by content, and code panes must not be
shrunk out of alignment by an element selector.

Two defects, one page, and they compound: the second was invisible for as long
as the first stopped fixes reaching a browser.

**The cache key.** `?v=N` was a literal typed into the template. `style.css?v=12`
was bumped twelve times, because a CSS change that does not land is obvious
within seconds. `highlight.js?v=1` was never bumped once in the file's entire
history -- so every JS change ever made to the gutter was invisible to any
browser that had loaded the board before, and there was no symptom here,
because curl and a fresh profile always get fresh bytes. The person who sees a
stale asset is never the person who changed it. Derived from the file's hash
now, which cannot be forgotten.

**The gutter.** `code,.mono{font-size:.92em}` at the top of the stylesheet is an
unscoped element selector -- the third to land on this file, after D32's
`nav{}`. `.src` is a `<code>` and the gutter is a `<span>`, so it hit exactly
one of them: 10.58px against 11.5px, and with `line-height:1.6` unitless that
is 16.93px per line of code against 18.4px per number. The columns drift 8.7%.
Source line 56 renders opposite the number 51, and a 79-line kernel trails
about six numbered but empty rows. It reads as trailing blank lines. It is
actually every line number on the page being wrong, and getting wronger the
further down you read, in a viewer whose whole job is showing kernels.

Both checks are on the served bytes. There is no browser here, so what is
asserted is that the stylesheet pins the metrics rather than that the rendering
came out aligned -- the same limit `test_coverage_bar` works within.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

ASSET = re.compile(r'/static/([a-z0-9_.-]+)\?v=([0-9a-f]+)')
LITERAL = re.compile(r'/static/[a-z0-9_.-]+\?v=\d+(?:["\'\s>])')


def _pages(client) -> dict[str, str]:
    key = client.get("/api/v1/leaderboard").json()[0]["slug"]
    sub = client.get(f"/api/v1/submissions/{key}").json()
    out = {"/": client.get("/").text,
           "/methodology": client.get("/methodology").text}
    probs = sub.get("problems") or []
    if probs:
        out["run"] = client.get(
            f"/submissions/{key}/problems/{probs[0]['key']}").text
    return out


def test_every_static_asset_is_content_hashed(client):
    seen = 0
    for name, page in _pages(client).items():
        for fn, ver in ASSET.findall(page):
            seen += 1
            assert len(ver) >= 8, (name, fn, ver)
        assert not LITERAL.search(page), (
            f"{name} still has a hand-written ?v=N cache key; it is the one "
            "that goes stale, because nothing forces it to change")
    assert seen >= 3, f"expected style.css, highlight.js and run.js; saw {seen}"


def test_the_hash_changes_when_the_file_does(client, tmp_path):
    """Otherwise it is a constant with extra steps."""
    import app as appmod
    before = appmod.asset("/static/highlight.js")
    f = appmod.HERE / "static" / "highlight.js"
    original = f.read_bytes()
    try:
        f.write_bytes(original + b"\n/* touched by the test */\n")
        after = appmod.asset("/static/highlight.js")
    finally:
        f.write_bytes(original)
    assert before != after
    assert appmod.asset("/static/highlight.js") == before, "not restored"


def test_a_missing_asset_does_not_take_the_page_down(client):
    """A 404 the reader can see beats a 500 they cannot."""
    import app as appmod
    assert appmod.asset("/static/does-not-exist.js") == "/static/does-not-exist.js"


def test_the_gutter_and_the_code_share_the_pre_s_metrics(client):
    """The bare `code{font-size:.92em}` is still in the file -- it is correct
    for inline code. What must hold is that it cannot reach `.src`."""
    css = client.get("/static/style.css").text
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    m = re.search(r'pre\.code\.has-ln \.gutter,\s*pre\.code\.has-ln \.src\{([^}]*)\}',
                  css)
    assert m, "the two columns no longer pin their own metrics"
    body = m.group(1)
    assert "font-size:inherit" in body
    assert "line-height:inherit" in body


def test_the_stylesheet_still_scopes_its_element_selectors(client):
    """`nav{}` was D32, `code{}` reaching `.src` was this one. A bare selector
    on any element the panes use is the same bug waiting for its third turn."""
    css = client.get("/static/style.css").text
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    for el in ("pre", "span", "nav"):
        assert not re.search(rf'(?m)^{el}\s*\{{', css), f"unscoped `{el} {{`"
