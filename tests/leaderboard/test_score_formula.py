#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The one equation the whole board is derived from, set as an equation.

It was `<code>S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))</code>` — a
division inside a division, both written as slashes, four levels of nested
parentheses, in 12px monospace. Now it is two stacked fractions in HTML and
CSS, from one partial included by both pages that show it, so the front page
and the methodology cannot come to disagree about the score.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

PAGES = ("/", "/methodology")


@pytest.mark.parametrize("url", PAGES)
def test_the_formula_is_set_as_a_fraction(client, url):
    body = client.get(url).text
    eq = body.split('<div class="eq"', 1)
    assert len(eq) == 2, f"{url} renders no equation block"
    eq = eq[1].split("</div>", 1)[0]
    # Two fractions: the outer 1/(1+x) and the inner ratio inside its
    # denominator. One would mean the nesting was flattened back to a slash.
    assert eq.count('class="frac"') == 2, eq[:300]
    assert eq.count('class="fr-n"') == 2 and eq.count('class="fr-d"') == 2


@pytest.mark.parametrize("url", PAGES)
def test_the_equation_reads_aloud(client, url):
    """A screen reader walking the spans hears the numerator and the
    denominator run together, which is a different equation."""
    eq = client.get(url).text.split('<div class="eq"', 1)[1][:400]
    assert 'role="img"' in eq and "aria-label=" in eq


@pytest.mark.parametrize("url", PAGES)
def test_both_anchors_are_named_beside_it(client, url):
    """An equation with no key is a puzzle. T_b and T_SOL are what make 0.5
    and 1.0 mean anything, and they are in the block, not three paragraphs
    below it."""
    key = client.get(url).text.split('class="eq-key"', 1)[1].split("</div>", 1)[0]
    assert "0.5" in key and "1.0" in key
    assert "T<sub>b</sub>" in key and "T<sub>SOL</sub>" in key


def test_the_flat_form_is_gone_from_the_front_page(client):
    assert "1 / (1 + (T" not in client.get("/").text


def test_the_manifests_own_string_still_appears_on_the_methodology(client):
    """The rendering is a rendering; the artifact is the authority on what was
    computed, and it stays printed verbatim underneath."""
    body = client.get("/methodology").text
    assert "S = (T_b - T_k) / (T_b - T_SOL) / 2 + 0.5" in body \
        or "S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))" in body
