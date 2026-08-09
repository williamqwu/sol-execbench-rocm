#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The board has to survive a 390px viewport.

Not "look nice on a phone" -- survive. The failure mode this suite exists for
is content that is present in the HTML, correct, and unreachable: a table wider
than the screen inside a box that clips instead of scrolling shows four of its
seven columns and gives no indication the other three exist. Ten of the
fourteen tables on this site were in that state, including the leaderboard,
whose `.board` card carries `overflow:hidden` for its rounded corners.

Three classes of defect are pinned here, each of which shipped:

  * **A table with no scrollport.** Wrapping is per-table and therefore
    forgettable, so the check is over every template, not over a list.
  * **Header height as a literal.** `59px` was hardcoded in three rules -- the
    section nav's sticky offset, its max-height, and every anchor's
    scroll-margin. The header is 58px only while it fits on one line; on a
    phone it wraps, and then in-page anchors land with the heading they point
    at hidden behind it. On the one viewport where an anchor is the main way to
    navigate a long page.
  * **A scoped selector with an unscoped copy.** D32 was a bare `nav{}` reaching
    the section nav. The top-level rules were scoped; the two inside the 900px
    media query were not, so the bug came back below 900px on a page that
    looked fixed. That is worse than never having scoped it.

There is no browser here, so nothing below asserts a rendered pixel. What is
asserted is the structure the rendering depends on -- the same limit
`test_coverage_bar` and `test_asset_versions` work within.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

TEMPLATES = Path(__file__).resolve().parents[2] / "leaderboard" / "templates"
PHONE = 640  # the breakpoint; also the widest phone this is claimed to hold


def _css(client) -> str:
    """Served stylesheet with comments stripped -- prose about a selector is
    not the selector, and every rule in this file is heavily commented."""
    return re.sub(r'/\*.*?\*/', '', client.get("/static/style.css").text, flags=re.S)


def _templates() -> list[Path]:
    out = sorted(TEMPLATES.glob("*.html"))
    assert out, f"no templates under {TEMPLATES}"
    return out


def test_every_table_is_inside_a_scrollport():
    """The one that made the leaderboard unusable on a phone. Checked over the
    templates rather than the rendered pages because a table inside a `{% if %}`
    that no fixture triggers is still a table someone will see."""
    offenders = []
    for f in _templates():
        src = f.read_text()
        for m in re.finditer(r'<table\b', src):
            before = src[:m.start()].rstrip()
            if not before.endswith('<div class="scroll">'):
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{f.name}:{line}")
    assert not offenders, (
        "these tables have no horizontal scrollport, so on a narrow viewport "
        "their right-hand columns are clipped and unreachable: "
        + ", ".join(offenders))


def test_the_scrollport_scrolls_horizontally(client):
    css = _css(client)
    assert re.search(r'\.scroll\{[^}]*overflow-x:auto', css), \
        "`.scroll` no longer scrolls; every table wrapper is now just a div"


def test_the_header_height_is_a_variable_not_a_literal(client):
    """`--headh` exists so the header may wrap without stranding anchors."""
    css = _css(client)
    assert re.search(r'--headh:\s*\d+px', css), "--headh is gone"
    # The three consumers must read the variable.
    for pat in (r'\.sidenav\{[^}]*top:calc\(var\(--headh\)',
                r'\.sidenav\{[^}]*max-height:calc\([^}]*var\(--headh\)',
                r'scroll-margin-top:calc\(var\(--headh\)'):
        assert re.search(pat, css), f"a header offset stopped reading --headh: {pat}"


def test_the_phone_breakpoint_redefines_the_header_height(client):
    """A wrapped header that still claims to be 59px tall is the anchor bug."""
    css = _css(client)
    m = re.search(r'@media \(max-width:(\d+)px\)\{((?:[^{}]|\{[^{}]*\})*)\}', css)
    blocks = {int(w): b for w, b in
              re.findall(r'@media \(max-width:(\d+)px\)\{((?:[^{}]|\{[^{}]*\})*)\}', css)}
    assert PHONE in blocks, f"no {PHONE}px breakpoint; phones get the desktop layout"
    phone = blocks[PHONE]
    assert "--headh" in phone, (
        "the phone breakpoint wraps the header but does not restate --headh, "
        "so every in-page anchor lands behind it")
    assert "flex-wrap:wrap" in phone, "the header cannot wrap, so it will overflow"


def test_no_unscoped_element_selector_reaches_the_landmarks(client):
    """D32 was a bare `nav{}` reaching the section nav. D34's sibling was a bare
    `code{}` reaching the code pane's `.src`. Both are the same shape: an
    element selector written when the site had one such element.

    Selectors are parsed rather than matched by lookbehind. The first attempt
    at this test used `(?<![\\w.\\-#>~+ ])nav\\{` and the space in that character
    class silently excluded every indented rule -- which is to say every rule
    inside a media query, which is exactly where the bug had survived. It
    passed against the real defect.
    """
    css = _css(client)
    # A BARE element selector: the whole selector is one element name, so it
    # matches every such element on the site. `pre.code` and `header nav` are
    # qualified and are the fix, not the bug.
    #
    # `code` is deliberately absent from this list. It is global on purpose --
    # inline <code> in prose is styled site-wide -- and the code pane is immune
    # because `.src` restates its own font-size and line-height. That immunity
    # is asserted in test_asset_versions, which is where it belongs; banning
    # the selector here would be treating the symptom.
    # Only elements that appear MORE THAN ONCE per page in DIFFERENT roles.
    # `header`, `footer` and `table` are bare on purpose: there is one of each
    # landmark per page, and `table{border-collapse}` is meant to reach every
    # table. A bare selector is only a hazard when a second element of the same
    # name arrives wanting different rules -- which is precisely what happened
    # when the section nav was added beside the site nav.
    LANDMARKS = {"nav", "pre", "aside", "main"}
    bad = []
    for m in re.finditer(r'([^{}@]+)\{[^{}]*\}', css):
        for sel in m.group(1).split(","):
            sel = sel.strip()
            if sel in LANDMARKS:
                bad.append(sel)
    assert not bad, (
        "bare element selectors on landmarks the page now uses more than once: "
        + ", ".join(sorted(set(bad)))
        + " -- this is the D32 shape; qualify them with an ancestor or a class")


def test_touch_targets_are_declared_on_the_phone_breakpoint(client):
    """Not measured -- declared. 44px is the floor; the desktop padding gives
    about 28px, which is a link you miss twice."""
    css = _css(client)
    phone = dict(re.findall(r'@media \(max-width:(\d+)px\)\{((?:[^{}]|\{[^{}]*\})*)\}',
                            css))[str(PHONE)]
    assert "min-height:44px" in phone, "no 44px hit area anywhere on the phone layout"
    assert re.search(r'header nav a\{[^}]*min-height:44px', phone), \
        "the site nav is the one thing a phone reader must be able to hit"


def test_the_search_box_does_not_trigger_ios_zoom(client):
    """Under 16px, iOS zooms the viewport in on focus and does not zoom back
    out. One rule between a reader and a page they have to pinch out of."""
    css = _css(client)
    phone = dict(re.findall(r'@media \(max-width:(\d+)px\)\{((?:[^{}]|\{[^{}]*\})*)\}',
                            css))[str(PHONE)]
    m = re.search(r'\.filters input\[type=search\]\{([^}]*)\}', phone)
    assert m, "the search box keeps its desktop font-size on a phone"
    size = re.search(r'font-size:(\d+(?:\.\d+)?)px', m.group(1))
    assert size and float(size.group(1)) >= 16, (
        f"search font-size is {size and size.group(1)}px; iOS zooms below 16")


def test_the_viewport_is_declared(client):
    """Without it a phone renders at 980px and scales down, and every fix above
    is dead code."""
    page = client.get("/").text
    assert re.search(r'<meta name="viewport"[^>]*width=device-width', page)


def test_nothing_is_hidden_from_phones_that_carries_a_number(client):
    """`display:none` on a phone is how a board quietly becomes a different
    board. The part-switch count is the one sanctioned drop and it is a
    duplicate of what the page it links to states in full."""
    css = _css(client)
    phone = dict(re.findall(r'@media \(max-width:(\d+)px\)\{((?:[^{}]|\{[^{}]*\})*)\}',
                            css))[str(PHONE)]
    hidden = re.findall(r'([^{};]+)\{[^}]*display:none', phone)
    assert not hidden, (
        "the phone layout hides " + ", ".join(s.strip() for s in hidden) +
        " -- a phone reader and a desktop reader must see the same board")
