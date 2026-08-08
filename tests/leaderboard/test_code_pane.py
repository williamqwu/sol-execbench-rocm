#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What the code pane shows must be the source, character for character.

The pane is the only place a reader sees the kernel that produced a number, and
the copy button hands `pre.textContent` to the clipboard -- so a template that
mangles the source does not merely look wrong, it exports something that was
never run. Three ways that happens, all silent:

  * an HTML parser drops a single LF immediately after `<pre>`, so a source
    beginning with a blank line loses it from the pane, from `textContent`, and
    therefore from the clipboard;
  * a Jinja filter (`trim`, `indent`, autoescaping applied twice) rewrites the
    body without raising;
  * `&`, `<` and `>` survive one escape and not the round trip back.

None of it is visible by reading the page -- escaped-once and escaped-twice
render identically for most sources. The check has to be mechanical: unescape
the served pane and compare it to the bytes in the database.

Highlighting itself is client-side (`static/highlight.js`) and this node has no
JavaScript runtime, so what is asserted here is the input that file is handed
and the wiring that hands it over. The tokenizer's own guarantees -- every
emitted chunk passes through `esc()`, sticky regexes that cannot match ahead of
the offset, a total fallback branch, and `data-hl` so a block is never scanned
twice -- are structural in that file and are not re-derived here.
"""

from __future__ import annotations

import html
import re
import sqlite3

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

PANE = re.compile(
    r'<div class="tabpane on" id="k-submitted">'
    r'<pre class="code sm" data-lang="([^"]*)">\n?(.*?)</pre>', re.S)

# A source built to break every one of the three failure modes at once: a
# leading blank line, the three characters HTML escapes, a docstring, a tab,
# and trailing whitespace that `trim` would eat.
NASTY = (
    "\n"
    "# <script>alert(1)</script> & \"quoted\" 'single'\n"
    'def f(a, b):\n'
    '    """a < b & b > a"""\n'
    "\tif a < b and b > a:\n"
    "        return a & b\n"
    "    return None   \n"
    "\n"
)


def _pane(client, slug: str, key: str) -> tuple[str, str]:
    html_ = client.get(f"/submissions/{slug}/problems/{key}").text
    m = PANE.search(html_)
    assert m, f"no submitted-kernel pane on /submissions/{slug}/problems/{key}"
    return m.group(1), html.unescape(m.group(2))


def test_a_hostile_source_survives_the_round_trip(board, client):
    """Unescaping the pane gives back exactly the bytes stored."""
    # `board.conn()` opens read-only; this test has to doctor the file, and
    # the app re-opens on mtime so the write is picked up.
    with sqlite3.connect(board.path()) as c:
        c.execute("UPDATE run_kernel SET source=? WHERE problem_key=?",
                  (NASTY, "L1__001_alpha"))
    _, got = _pane(client, "agent-trial-a", "L1__001_alpha")
    assert got == NASTY, repr(got)


def test_the_script_that_reads_the_pane_is_loaded(client):
    """A pane with no highlighter and no copy button still reads fine, which is
    why its absence is easy to ship. The tag is asserted, not the effect.

    This used to require `?v=<integer>` and so pinned the defect: the literal
    was never incremented in the file's whole history, and every gutter fix
    stayed invisible to any browser that had already cached it. The cache key
    is the file's content hash now, and a bare integer is the thing to reject.
    """
    page = client.get("/submissions/agent-trial-a/problems/L1__001_alpha").text
    m = re.search(r'<script src="/static/highlight\.js\?v=([0-9a-f]+)" defer>', page)
    assert m, "the highlighter is not loaded, or is not cache-busted at all"
    assert not m.group(1).isdigit(), (
        "hand-maintained integer cache key is back; it is the one that goes "
        "stale, because nothing forces it to change")


def test_every_pane_declares_a_language(real_client, real_conn):
    """`data-lang` is what selects the rule set. An empty or absent value
    degrades to plain text -- correct, but silently, on every kernel."""
    rows = real_conn.execute(
        """SELECT s.slug, k.problem_key FROM run_kernel k
           JOIN submission s ON s.id = k.submission_id
           WHERE k.source IS NOT NULL""").fetchall()
    if not rows:
        pytest.skip("no kernels on the built board")
    bad = [(r["slug"], r["problem_key"])
           for r in rows
           if _pane(real_client, r["slug"], r["problem_key"])[0]
           not in ("python", "hip", "cpp", "c", "cuda")]
    assert not bad, bad


def test_the_built_board_serves_its_kernels_verbatim(real_client, real_conn):
    """The same round trip against real sources rather than a constructed one.

    This is the test that would have caught a filter added to the template
    later: the fixture source is three tidy lines and would survive almost
    anything, while the shipped kernels are hundreds of lines of Triton with
    decorators, f-strings and comparison operators in them.
    """
    rows = real_conn.execute(
        """SELECT s.slug, k.problem_key, k.source FROM run_kernel k
           JOIN submission s ON s.id = k.submission_id
           WHERE k.source IS NOT NULL""").fetchall()
    if not rows:
        pytest.skip("no kernels on the built board")
    bad = []
    for r in rows:
        _, got = _pane(real_client, r["slug"], r["problem_key"])
        if got != r["source"]:
            bad.append((r["slug"], r["problem_key"], len(got), len(r["source"])))
    assert not bad, bad
