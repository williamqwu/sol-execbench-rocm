# SPDX-License-Identifier: Apache-2.0
"""One number system for the whole board, and the gate that keeps it sortable.

Two things are tested here and they fail in very different ways.

**The helpers** (`app.dur`, `app.ratio`, `app.sci`, ...) are checked against the
extreme values that are actually in `leaderboard/db/solbench-MI350X.db`; where a
case is a synthetic probe of a ladder boundary instead, it says `synthetic`, so
that no reader has to open the database to find out which kind a number is. The
interesting real cases are all at the ends: the smallest T_SOL on the board is
7.692307692307693e-07 ms, which is exactly ONE GPU cycle at F_LOCK 1300 MHz and
rendered as `0.00000` under the old `%.5f`; the largest T_b is 17060.6171875 ms;
`workload.bound_headroom` reaches 1499133.450644357; and 76 workloads across 5
problems record a tolerance of exactly 0.0, which truthiness printed as "not
recorded".

A ladder also has two ends that today's data does not reach, and an untested
end is where a formatter re-grows the defect it was written to remove. Both are
pinned below.

**The sort lint** is the one that matters more, because its failure is silent.
`base.html`'s sort handler falls back to

    parseFloat(text.replace(/[,%$×x*]/g, ""))

when a cell has no `data-sort`. Feed it "1.23k×" and it returns 1.23; feed it
"377 µs" and it returns 377. Not NaN — a plausible WRONG number, which sorts
into a plausible wrong position, in a column that looks perfectly fine. So
every helper that can emit a non-numeric character tags its wrapper
`q-needs-sort`, and this file asserts that no such cell in a sortable column
ships without a `data-sort` that round-trips through `float()`.

The convention is not the guarantee. This is.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "leaderboard"))


@pytest.fixture(scope="module")
def a():
    pytest.importorskip("fastapi", reason="leaderboard venv only")
    import app
    return app


# --------------------------------------------------------------------------
# the helpers, against the real extremes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("v,expected", [
    # One cycle at F_LOCK 1300 MHz -- the smallest t_sol_ms on the board, and
    # the D39 signal that `%.5f` rendered as `0.00000`.
    # Every value below was read out of `leaderboard/db/solbench-MI350X.db`
    # this session (`mode=ro`) and its home column is named. A case that is
    # NOT in the database is labelled `synthetic`.
    (7.692307692307693e-07, "0.769 ns"),     # min t_sol_ms
    (3.076923076923077e-06, "3.08 ns"),      # t_sol_ms, 8 rows
    (0.0006, "600 ns"),                      # b200_sol_ms, 36 rows
    (0.007500499952584505, "7.50 µs"),       # min latency_ms
    (0.37706, "377 µs"),                     # t_sol_ms, FlashInfer-Bench__016
    (1.0, "1.00 ms"),                        # synthetic: the ms rung's floor
    (276.288133240805, "276 ms"),            # max t_sol_ms
    (17060.6171875, "17.1 s"),               # max t_b_ms
    (17194.080078125, "17.2 s"),             # latency_ms
    (17314.3701171875, "17.3 s"),            # max latency_ms
    (6e-08, "0.060 ns"),                     # min b200_sol_ms
])
def test_a_duration_never_needs_scientific_notation(a, v, expected):
    """The ns/µs/ms/s ladder covers the entire measured span.

    6e-08 ms to 17314.37 ms is twelve decades, and every mantissa on it lands
    in [0.06, 999]. That is the whole reason `dur` can drop `fmt_ms`'s style
    switch at 1e-4 ms, which put two notations in one column.
    """
    assert a.dur_text(v) == expected
    assert "e-" not in a.dur_text(v) and "e+" not in a.dur_text(v)


def test_a_duration_column_carries_its_unit_in_its_own_slot(a):
    """The unit stripe is what makes a mixed-magnitude column readable."""
    html = str(a.dur(0.37706))
    assert '<span class="qv">377</span>' in html
    assert '<span class="qu">µs</span>' in html
    assert "q-needs-sort" in html, "a laddered cell must demand a data-sort"


def test_the_micro_sign_is_u_00b5(a):
    """U+00B5 MICRO SIGN, never U+03BC GREEK SMALL LETTER MU.

    They render nearly identically and would silently split any grep or
    assertion that matched on one of them.
    """
    assert "µs" in a.dur_text(0.37706)
    assert "μ" not in a.dur_text(0.37706)


def test_the_ladder_promotes_rather_than_printing_a_four_digit_mantissa(a):
    """999.7 ms is `1.00 s`, not `1000 ms`.

    Choosing the rung before rounding is how a ladder ends up printing a
    mantissa outside the band its unit promises.
    """
    assert a.dur_text(999.7) == "1.00 s"
    assert a.dur_text(0.0009997) == "1.00 µs"


def test_the_bottom_of_the_ladder_never_renders_a_real_value_as_zero(a):
    """`0.000 ns` is the SAME defect as `0.00000`, twelve decades lower.

    Nothing on the board is below 6e-08 ms, so this is latent -- which is the
    reason to pin it: the promotion in `_laddered` is guarded by `i > 0` and
    the demotion has no rung to demote to, so both ends have to be handled
    explicitly or they quietly are not handled at all.
    """
    assert a.dur_text(1e-10) == "1.00e-04 ns"
    assert a.dur_text(1e-12) == "1.00e-06 ns"
    for v in (1e-10, 1e-12, 1e-15):
        rendered = a.dur_text(v)
        assert not rendered.startswith("0.000 "), rendered
        assert "0" != rendered


def test_the_top_of_the_ladder_never_renders_a_four_digit_mantissa(a):
    """`1000M×` is three significant figures printed as four.

    The largest ratio on the board is `workload.bound_headroom` =
    1499133.450644357 -> `1.50M×`, so nothing reaches this either. `M×` is the
    top rung and the `i > 0` promotion cannot fire on it.
    """
    assert a.ratio_text(1e9) == "1.00e+03M×"
    assert a.ratio_text(1.5e6) == "1.50M×"
    for v in (1e9, 1e12, 999.5e6):
        mantissa = a.ratio_text(v).rstrip("M×")
        assert not re.fullmatch(r"\d{4}", mantissa), a.ratio_text(v)


@pytest.mark.parametrize("v,expected", [
    (1.1097593673588968, "1.11×"),  # the tightest bound_headroom on the board
    (2.03, "2.03×"),
    (12.4, "12.4×"),
    (769.0, "769×"),
    (1234.0, "1.23k×"),
    (115004.95628074363, "115k×"),  # L2__006, the loosest median headroom
    (805180.1, "805k×"),
    (1499133.450644357, "1.50M×"),  # the largest bound_headroom
    (0.01520103232079208, "0.015×"),  # the slowest geomean_speedup on record
    (422.7051885272674, "423×"),      # the fastest
])
def test_a_ratio_is_three_significant_figures_with_a_k_or_m_suffix(a, v, expected):
    assert a.ratio_text(v) == expected


@pytest.mark.parametrize("v,expected", [
    (0.0, "0.0000"),           # THE case: a real zero is not a missing value
    (0.04572772828684364, "0.0457"),   # the smallest result.score
    (0.5, "0.5000"),           # T_b, by construction. The landmark.
    (0.9992750809932985, "0.9993"),    # the largest result.score
    (1.0, "1.0000"),
])
def test_a_score_keeps_four_decimals_and_a_zero_is_a_zero(a, v, expected):
    assert a.score_text(v) == expected
    assert str(a.score(v)) == expected, "an in-contract score needs no marking"


@pytest.mark.parametrize("v", [32.354054, -0.781048, 1.0000001])
def test_a_score_outside_its_contract_is_marked_not_hidden(a, v):
    """27 of 7,073 `trajectory_eval.mean_score` rows are outside [0,1].

    No cause is asserted here and none has been investigated. The point is
    only that `run.html` renders those in the same `.big` mono style as a board
    score of 0.4907, which invites a comparison between two different
    quantities -- so the cell says which one it is.
    """
    html = str(a.score(v))
    assert "q-oor" in html
    assert "not a board score" in html


@pytest.mark.parametrize("v,mantissa,exp", [
    (2.6609234646237377e-11, "2.66", -11),   # the smallest nonzero tol_atol
    (2.66e-11, "2.66", -11),
    (778240.0, "7.78", 5),                   # the largest tol_atol
    (0.6666666666666666, "6.67", -1),        # the largest tol_rtol
])
def test_a_tolerance_is_scientific_with_an_html_sup(a, v, mantissa, exp):
    """`<sup>`, never Unicode superscript: `⁻¹¹` has a font-fallback risk in
    the JetBrains Mono / Menlo / Consolas stack and `<sup>` has none."""
    html = str(a.sci(v))
    assert f'{mantissa}×10<sup>{exp}</sup>' in html
    assert "⁻" not in html and "¹" not in html


def test_a_tolerance_of_exactly_zero_says_so_rather_than_going_missing(a):
    """76 workloads across 5 problems (L1__028, L1__058, L2__006, L2__049,
    Quant__011) record atol = rtol = 0.0 -- an EXACT match is required.

    `{{ '%.2e'|format(w.tol_atol) if w.tol_atol else '&mdash;' }}` printed
    every one of them as "not recorded". A measured fact displayed as a missing
    one is the bug this whole module exists to stop.
    """
    html = str(a.sci(0.0))
    assert ">0<" in html
    assert "exact match is required" in html
    assert html != str(a.MISSING)


@pytest.mark.parametrize("fn_name", [
    "dur", "score", "delta", "ratio", "sci", "cycles", "n", "usd", "mins",
    "pct", "bytes_h",
])
def test_every_helper_returns_the_one_missing_marker_for_none(a, fn_name):
    """One spelling of the em-dash, in one place, with one meaning."""
    assert str(getattr(a, fn_name)(None)) == str(a.MISSING)
    assert getattr(a, fn_name + "_text")(None) == a.MISSING_TEXT


@pytest.mark.parametrize("fn_name,zero", [
    ("dur", 0.0), ("score", 0.0), ("delta", 0.0), ("cycles", 0), ("n", 0),
    ("usd", 0.0), ("pct", 0.0),
])
def test_no_helper_treats_a_zero_as_a_missing_value(a, fn_name, zero):
    """The truthiness guard, in the abstract. `if v` is forbidden."""
    assert str(getattr(a, fn_name)(zero)) != str(a.MISSING)


def test_a_negative_or_zero_duration_is_marked_as_the_defect_it_is(a):
    """Nothing takes negative time and nothing takes zero time.

    Neither appears in the board today. Both would be a defect in the
    measurement path, and the cell has to make one visible rather than
    rendering `-1.00 ms` in the same grey as every other row.
    """
    assert "q-bad" in str(a.dur(-1.0))
    assert a.dur_text(-1.0) == "-1.00 ms"
    zero = str(a.dur(0.0))
    assert "q-zero" in zero and "not a measurement" in zero


def test_cycles_stay_exact(a):
    """Audit columns get exact values; comparison columns get 3 s.f.

    T_SOL in cycles is the derived architectural quantity the millisecond
    column is computed from, it is F_LOCK-invariant, and it is in the `c-deriv`
    group -- so `398,131,200`, not `398M`.
    """
    assert a.cycles_text(398131200) == "398,131,200"
    assert a.cycles_text(1) == "1"


def test_a_wall_clock_span_never_reads_as_no_time(a):
    assert a.mins_text(40) == "<1 min"
    assert a.mins_text(0) == "<1 min"
    assert a.mins_text(5400) == "1 h 30 min"
    assert a.mins_text(3600) == "1 h 00 min"
    assert a.mins_text(1800) == "30 min"


def test_a_nonzero_share_never_rounds_to_zero_percent(a):
    assert a.pct_text(0.02) == "<0.1%"
    assert a.pct_text(22.3) == "22.3%"
    assert a.pct_text(0.0) == "0.0%"


def test_usd_and_bytes(a):
    assert a.usd_text(8.014785) == "$8.01"      # run_effort.cost_usd, low end
    assert a.usd_text(8.2273595) == "$8.23"     # high end
    assert a.usd_text(12345.6) == "$12,346"
    assert a.bytes_h_text(78_000_000) == "78.0 MB"


def test_sortv_is_the_raw_float_and_blank_for_absent(a):
    """`data-sort` is where the digits the 3-s.f. cell dropped still live."""
    assert float(a.sortv(7.692307692e-07)) == 7.692307692e-07
    assert float(a.sortv(17060.6171875)) == 17060.6171875
    assert a.sortv(None) == ""


def test_base_html_reads_the_presence_of_data_sort_not_its_truthiness():
    """`sortv(None)` is "", and "" has to MEAN "sort me last".

    It did not: the read was `getAttribute("data-sort") || c.innerText`, and an
    empty attribute is falsy in JS, so the cell fell through to its own text.
    That gave the right answer only where the text happened to be the em-dash
    the next line maps to null; run.html's trajectory "at" column says "not
    recorded", which `parseFloat` turns into NaN and the sorter then compares
    as a lowercase string against numbers -- neither first nor last.
    """
    base = (Path(__file__).resolve().parents[2]
            / "leaderboard" / "templates" / "base.html").read_text()
    assert 'var ds = c.getAttribute("data-sort");' in base
    assert 'var t = (ds === null ? (c.innerText || "") : ds).trim();' in base
    assert 'c.getAttribute("data-sort") || c.innerText' not in base, (
        "the falsy read is back: an empty data-sort no longer means null")


# --------------------------------------------------------------------------
# the lint: every laddered cell in a sortable column carries a data-sort
# --------------------------------------------------------------------------

class _Cells(HTMLParser):
    """Collect (table_classes, column_index, header_classes, td_attrs, td_html).

    A parser rather than a regex because the question is structural: which
    COLUMN a cell is in, and whether that column's `th` opted out with
    `.nosort`. `base.html` sorts by `th.cellIndex`, so the column index is the
    cell's position in its own row -- which is what this reproduces.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []      # stack
        self.cells: list[tuple] = []
        self._buf: list[str] = []
        self._td = None

    # -- structure
    def handle_starttag(self, tag, attrs):
        at = dict(attrs)
        if tag == "table":
            self.tables.append({"cls": at.get("class", ""), "th": [], "in_head": False})
        elif tag == "thead" and self.tables:
            self.tables[-1]["in_head"] = True
        elif tag == "tr":
            self._col = 0
        elif tag == "th" and self.tables:
            self.tables[-1]["th"].append(at.get("class", ""))
        elif tag == "td" and self.tables:
            self._td = (self.tables[-1], self._col, at)
            self._buf = []
        if tag in ("td", "th"):
            self._col = getattr(self, "_col", 0) + 1
        # A self-closing/void tag inside a td still belongs to the td body.
        if self._td is not None and tag not in ("td",):
            self._buf.append(self.get_starttag_text() or "")

    def handle_startendtag(self, tag, attrs):
        if self._td is not None:
            self._buf.append(self.get_starttag_text() or "")

    def handle_data(self, data):
        if self._td is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self._td is not None:
            table, col, at = self._td
            ths = table["th"]
            self.cells.append((table["cls"], col,
                               ths[col] if col < len(ths) else "",
                               at, "".join(self._buf)))
            self._td = None
        elif tag == "thead" and self.tables:
            self.tables[-1]["in_head"] = False
        elif tag == "table" and self.tables:
            self.tables.pop()
        elif self._td is not None:
            self._buf.append(f"</{tag}>")


PAGES = [
    "/",
    "/problems",
    "/problems/L1__001_alpha",
    "/problems/Quant__003_gamma",
    "/methodology",
    "/submissions/agent-trial-a",
    "/submissions/agent-trial-a/problems/L1__001_alpha",
    "/submissions/agent-timeout/problems/L2__002_beta",
]


def _cells(client, url):
    r = client.get(url)
    assert r.status_code == 200, (url, r.status_code)
    p = _Cells()
    p.feed(r.text)
    return p.cells


@pytest.mark.parametrize("url", PAGES)
def test_no_laddered_cell_in_a_sortable_column_is_missing_its_data_sort(client, url):
    """The gate.

    `parseFloat("1.23k×".replace(/[,%$×x*]/g,""))` is 1.23 and
    `parseFloat("377 µs")` is 377. Both are wrong and neither is NaN, so a
    column sorted on the rendered text looks exactly like one sorted on the
    value. Without this test the format change breaks a column sort within two
    commits and nothing says so.
    """
    bad = []
    for tcls, col, thcls, at, body in _cells(client, url):
        if "sortable" not in tcls:
            continue
        if "nosort" in thcls:
            continue                      # that column is never sorted
        if "q-needs-sort" not in body:
            continue
        if at.get("data-sort") is None:
            bad.append((url, col, body[:120]))
    assert not bad, f"laddered cell with no data-sort: {bad}"


@pytest.mark.parametrize("url", PAGES)
def test_every_data_sort_round_trips_through_float_or_is_deliberately_blank(
        client, url):
    """`base.html` treats "" as null and sorts it last in either direction.

    Anything else in a numeric column has to parse, or the column silently
    falls back to a lexical sort where "10" precedes "9".
    """
    for tcls, col, thcls, at, body in _cells(client, url):
        v = at.get("data-sort")
        if v is None or v == "":
            continue
        if re.fullmatch(r"[-+0-9.eE]+", v):
            float(v)                      # raises -> the test fails, correctly
        # Non-numeric data-sort is legal and used on purpose: status enums,
        # submission names, the zero-padded dataset-index key.


def test_the_lint_holds_on_the_real_board_too(real_client, real_conn):
    """The fixture has twelve tidy workloads; the real board has 3,717.

    Only the real one exercises the cases the ladder exists for -- a T_SOL
    column spanning `377 µs` to `3.08 ns`, a headroom column reaching `1.50M×`
    -- and only it can catch a column that renders fine on 0.1 ms and not on
    7.7e-07. The fixture test above is the invariant; this is the coverage.
    """
    key = real_conn.execute(
        "SELECT problem_key FROM workload WHERE t_sol_ms IS NOT NULL "
        "GROUP BY problem_key ORDER BY MAX(t_b_ms) / MIN(t_sol_ms) DESC "
        "LIMIT 1").fetchone()[0]
    row = real_conn.execute(
        "SELECT s.slug, r.problem_key FROM result r "
        "JOIN submission s ON s.id = r.submission_id LIMIT 1").fetchone()
    urls = ["/", "/problems", f"/problems/{key}", "/methodology",
            f"/submissions/{row[0]}",
            f"/submissions/{row[0]}/problems/{row[1]}"]
    checked, bad = 0, []
    for url in urls:
        for tcls, col, thcls, at, body in _cells(real_client, url):
            if "sortable" not in tcls or "nosort" in thcls:
                continue
            if "q-needs-sort" not in body:
                continue
            checked += 1
            if at.get("data-sort") is None:
                bad.append((url, col, body[:120]))
            else:
                float(at["data-sort"])
    assert not bad, f"laddered cell with no data-sort: {bad}"
    assert checked > 100, f"only {checked} laddered cells seen; the probe missed"


def test_the_strip_regex_in_base_html_was_not_widened():
    """Widening it would make MORE strings parse, and more of those wrong.

    `parseFloat("1.23k×")` after stripping `×` is 1.23. Adding `k`, `M`, `µ` or
    `s` to the class makes "1.23k×" parse to 1.23 just the same and "377 µs" to
    377 -- the failure this design routes around with `data-sort` instead.
    """
    base = (Path(__file__).resolve().parents[2]
            / "leaderboard" / "templates" / "base.html").read_text()
    assert r'parseFloat(t.replace(/[,%$×x*]/g,""))' in base, (
        "the sorter's fallback changed; re-read this design before widening it")


_TAG = re.compile(r"<[^>]*>")


def test_the_em_dash_has_exactly_one_spelling_in_value_position(client):
    """A cell whose whole visible text is the dash MUST come from `app.EM_DASH`.

    The previous version of this test asserted only that no literal
    `&amp;mdash;` reached the page, which is a much weaker claim than its own
    name and did not notice that two value positions still spelled the dash
    themselves: `problems.html`'s deferred submission-count cell and
    `run.html`'s trajectory pass-count cell. Both went through the parser as a
    plain `<span class="sub">` and rendered a dash with no `q-na`, no title,
    and no way for a reader to tell "absent" from "styled small".

    Rendered pages, not source, because the question is what a reader sees.
    """
    seen, stray = 0, []
    for url in PAGES:
        r = client.get(url)
        assert r.status_code == 200, (url, r.status_code)
        assert "&amp;mdash;" not in r.text, (
            f"{url}: a literal '&mdash;' reached the page unescaped-by-|safe")
        p = _Cells()
        p.feed(r.text)
        for tcls, col, thcls, at, body in p.cells:
            if _TAG.sub("", body).strip() != "—":
                continue          # prose, or a real value; not this invariant
            seen += 1
            if 'class="q-na"' not in body:
                stray.append((url, col, body[:160]))
    assert not stray, f"a dash-only cell not built from app.EM_DASH: {stray}"
    assert seen, "no dash-only cell on any fixture page; the probe missed"


def test_no_template_spells_the_em_dash_in_value_position():
    """The source-side half, so the offender is named at its line.

    An element whose entire content is a dash is a value position by
    definition. Both forms are banned: `>&mdash;<` and the bare character.
    Prose dashes inside a sentence are untouched -- they are not values.
    """
    tpl = Path(__file__).resolve().parents[2] / "leaderboard" / "templates"
    bad = []
    for f in sorted(tpl.glob("*.html")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(r">\s*(?:&mdash;|—)\s*<", line):
                bad.append(f"{f.name}:{i}: {line.strip()[:120]}")
    assert not bad, (
        "a template spells the em-dash itself; use `missing` or `na(reason)`: "
        + "; ".join(bad))


def test_a_zero_tolerance_renders_as_zero_on_a_real_problem_page(client, board):
    """End to end, through the template, not just the helper.

    The fixture's workloads carry atol = rtol = 1e-3; this drives one to
    exactly 0.0 the way five real problems have it, and asserts the page says
    "exactly zero" rather than showing the missing-value dash.
    """
    board.write("UPDATE workload SET tol_atol = 0.0, tol_rtol = 0.0 "
                "WHERE uuid = 'a1'")
    page = client.get("/problems/L1__001_alpha").text
    assert "exactly zero — an exact match is required" in page
    assert 'data-sort="0.0"' in page


def test_the_numeric_columns_are_set_in_tabular_figures():
    """style.css had zero `font-variant-numeric` rules, so digits in the
    proportional face did not line up at all."""
    css = (Path(__file__).resolve().parents[2]
           / "leaderboard" / "static" / "style.css").read_text()
    assert "font-variant-numeric:tabular-nums" in css.replace(" ", "")
    for sel in ("td.r", "td.mono", ".big", ".card .num", ".q"):
        assert sel in css
