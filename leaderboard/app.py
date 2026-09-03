#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SOL-ExecBench-ROCm leaderboard — local web app.

    leaderboard/run.sh              # http://127.0.0.1:8088

Pages render server-side from SQLite; the same data is available as JSON under
`/api/` for anything that wants to consume it.

Two ranking modes, both shown, because a leaderboard that offers only one is
easy to game:

* **Benchmark score** — the sum of per-workload scores over *every* scoreable
  workload in the benchmark, with anything not passed contributing zero. This
  is the headline. A submission that solves eight problems brilliantly cannot
  outrank one that solves two hundred.
* **Mean score (attempted)** — the average over the workloads a submission
  *attempted*, with an attempt that failed counting as zero. Always displayed
  next to its coverage. Useful for reading a partial run; meaningless without
  the coverage figure, so the UI never shows one without the other.

  This deliberately does **not** average over passes only. Averaging over
  passes rewards giving up: `torch.compile` reads 0.4907 that way, above eager
  PyTorch's 0.4548, purely because the 585 workloads it raised on vanish from
  the denominator instead of scoring zero. Both denominators are defensible in
  isolation, but only one of them cannot be improved by attempting less.

One database per part, and the part is not a filter over one dataset — it
*selects* the dataset. A score measured on MI350X and one measured on MI355X
differ in power cap, sustained clock and therefore F_LOCK, T_SOL and T_b, so
putting them in one table would be the most damaging kind of wrong: every
number plausible, nothing detectably broken. Keeping them in separate files
means no query can mix them by accident, only a deliberate join across two
connections could.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import inputs
import submit

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context
from markupsafe import Markup, escape

from models import (Health, LeaderboardRow, PartInfo, ProblemDetail,
                    ProblemSummary, ProvisionalRow, RunDetail, Stats,
                    SubmissionDetail)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The part registry, from the port itself rather than a second list here: a
# name that is not in `PARTS` is not a part this benchmark knows how to
# measure, and two copies of that fact would drift. `src` is not installed in
# the leaderboard venv -- it deliberately holds only fastapi, uvicorn and
# jinja2, so that serving the board cannot perturb the measurement image -- so
# the path goes on `sys.path` instead of becoming a packaging dependency.
# Appended, not prepended: nothing in `src` should ever shadow the stdlib or
# the venv, and this is the web app, not the measurement environment.
sys.path.append(str(ROOT / "src"))
from solexbench_rocm.parts import PARTS   # noqa: E402

DB_DIR = HERE / "db"
LEGACY_DB = HERE / "solbench.db"
DEFAULT_PART = "MI350X"
PART_COOKIE = "part"

app = FastAPI(title="SOL-ExecBench-ROCm leaderboard", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))
# `meta` is a flat string->string table, so JSON-valued rows come back as text.
templates.env.filters["from_json"] = lambda s: json.loads(s) if s else []

# The result enum, in words. The pages printed the raw value -- a reader met
# `REWARD_HACK` and `INCORRECT_NUMERICAL` in a status column with no key
# anywhere on the site, and the first of those is an accusation that deserves
# to be legible. The enum stays: it is what `/api/v1` returns and what the
# artifacts hold, so the templates keep it in the cell's `title` and sort by it.
# Anything unmapped degrades to the enum lowercased rather than to a blank.
STATUS_LABELS = {
    "PASSED": "passed",
    "REWARD_HACK": "reward hacking",
    "INCORRECT_NUMERICAL": "wrong output",
    "RUNTIME_ERROR": "runtime error",
    "COMPILE_ERROR": "compile error",
    "TIMEOUT": "timed out",
    "ERROR": "error",
}
templates.env.filters["status_label"] = (
    lambda s: STATUS_LABELS.get(s, (s or "").replace("_", " ").lower()))


# --------------------------------------------------------------------------
# numbers — one helper per KIND, never per site
# --------------------------------------------------------------------------
#
# Every number on this site used to be formatted where it was printed. That
# produced five renderings of one ratio (`%.1f×`, `%.3f×`, `%.2f×`, `%.3f×`,
# `{:,.0f}×`) across four pages, and — worse — a FIXED NUMBER OF DECIMALS on
# quantities that span decades. `'%.5f'` is 10 significant figures at
# `17060.61719 ms` and 1 at `0.00010 ms`, in the same column. Constant
# *relative* precision is what a timing has; constant absolute precision is
# what a format string gives, and that mismatch is the defect.
#
# So the invariant every helper below enforces is: **constant significant
# figures, and an explicit magnitude marker on every cell**. Scientific
# notation is one way to meet it; a unit ladder is the other, and it is the
# readable one wherever the quantity has a unit. Which one applies is decided
# by the KIND of number, never by the page:
#
#   * unit-scaled where there is a unit ladder  -> `dur`, `bytes_h`, `mins`
#   * a k/M suffix where it is a bare multiple  -> `ratio`
#   * true scientific where there is neither
#     and the span is enormous                  -> `sci`  (tolerances, 16 decades)
#   * plain fixed decimals where the quantity
#     is bounded and has a landmark             -> `score`, `delta`, `pct`,
#                                                  `usd`, `n`, `cycles`
#
# `5.00e-01` would destroy the one landmark the score column exists to show
# (T_b is S = 0.5 exactly), and `1.11e+00` would destroy "this problem is
# already at the roofline". Hence the split.
#
# Two rules that are load-bearing rather than stylistic:
#
# 1. **Every helper tests `v is None`, never truthiness.** `if w.tol_atol` is
#    why 76 real workloads across 5 problems whose tolerance is exactly 0.0 —
#    an exact match is required — rendered as "not recorded". A measured fact
#    displayed as a missing one is the failure mode this whole module exists
#    to stop.
# 2. **Any helper that can emit a non-numeric character tags its wrapper
#    `q-needs-sort`**, and the cell must then carry `data-sort`. base.html's
#    sorter falls back to `parseFloat(text.replace(/[,%$×x*]/g,""))`, which
#    turns "1.23k×" into 1.23 and "377 µs" into 377 — a plausible WRONG
#    number, not a NaN, so a mis-sorted column looks perfectly fine.
#    `tests/leaderboard/test_number_format.py` is the gate; the convention is
#    not the guarantee.

#: THE em-dash. Every one in value position anywhere on the site comes from
#: this character, through `MISSING` or through `na()` — no template spells it
#: again, as an entity or as a literal. Three spellings used to compete
#: (`'&mdash;'|safe`, `else '—'`, and a bare inline entity), and the check in
#: front of them was usually truthiness.
#:
#: The dash has exactly one meaning: *there is no value here*. Why there is
#: none is the title's job, not the glyph's. A measured zero is NOT this — it
#: renders as `0`, dimmed, with a title saying so.
EM_DASH = "—"
MISSING = Markup(f'<span class="q-na" title="not recorded">{EM_DASH}</span>')
MISSING_TEXT = EM_DASH


def na(reason: str) -> Markup:
    """The dash, for a value that is absent for a stated reason other than
    "not recorded" — a deferred problem cannot have a submission count, and
    saying "not recorded" there would be wrong."""
    return Markup(f'<span class="q-na" title="{escape(reason)}">{EM_DASH}</span>')

# The micro sign is U+00B5 (MICRO SIGN), never U+03BC (GREEK SMALL LETTER MU).
# They render nearly identically and would silently split any grep or test
# that matches on one of them.
MICRO = "µ"

_DUR_LADDER = ((1e3, "s"), (1.0, "ms"), (1e-3, MICRO + "s"), (1e-6, "ns"))
_RATIO_LADDER = ((1e6, "M×"), (1e3, "k×"), (1.0, "×"))
_BYTE_LADDER = ((1e9, "GB"), (1e6, "MB"), (1e3, "kB"), (1.0, "B"))


def _sig3(m: float) -> str:
    """Three significant figures, never scientific.

    The cut points are the rounded ones, so 99.96 comes out `100` rather than
    `100.0`: choosing the spec from the unrounded magnitude is how a "3 s.f."
    formatter quietly prints four.
    """
    a = abs(m)
    if a >= 99.95:
        return f"{m:.0f}"
    if a >= 9.995:
        return f"{m:.1f}"
    if a >= 0.9995:
        return f"{m:.2f}"
    return f"{m:.3f}"


def _sci3(m: float) -> str:
    """Three significant figures, scientific. Only the ends of a ladder use it.

    The exponent is chosen from the ROUNDED mantissa, for the same reason
    `_sig3`'s cut points are: 9.996e2 taken naively prints `10.00e+02`, four
    digits and the wrong decade.
    """
    e = math.floor(math.log10(abs(m)))
    if abs(m) / 10 ** e >= 9.995:
        e += 1
    return f"{m / 10 ** e:.2f}e{e:+03d}"


def _laddered(v: float, ladder) -> tuple[str, str]:
    """(mantissa, unit) on *ladder*, promoting a rung when rounding hits 1000.

    Without the promotion, 999.7 ms renders `1000 ms` — a mantissa outside the
    band the ladder promises, and the one case where the unit stripe would lie
    about which decade the cell is in.

    A ladder has two ends, and at each one there is no rung to move to. Both
    are OUTSIDE the measured span of this board — no value in
    `leaderboard/db/solbench-MI350X.db` reaches either — but a formatter that
    re-creates the defect it was written to remove, for inputs it merely does
    not happen to see, has not removed it:

    * below the last rung, `_sig3` rounds to `0.000`, which is the "a real
      bound displayed as zero" failure moved down twelve decades rather than
      eliminated;
    * above the first rung, the `i > 0` promotion cannot fire, and `_sig3`
      prints a four-digit mantissa (`1000M×`).

    At both ends the honest render is scientific: the unit still says which
    rung, and the exponent carries what the rung cannot.
    """
    a = abs(v)
    last = len(ladder) - 1
    for i, (scale, unit) in enumerate(ladder):
        if a >= scale or i == last:
            m = v / scale
            if abs(m) >= 999.5:
                if i == 0:
                    return _sci3(m), unit
                scale, unit = ladder[i - 1]
                m = v / scale
            elif i == last and 0 < abs(m) < 0.0005:
                return _sci3(m), unit
            return _sig3(m), unit
    raise AssertionError("unreachable: the last rung always matches")


def _q(value: str, unit: str = "", cls: str = "", title: str = "",
       needs_sort: bool = False) -> Markup:
    """The quantity wrapper: a fixed-width value slot and a fixed-width unit.

    `.qu` gets its own slot so the units form a vertical stripe down the
    column, which is what lets a reader see at a glance that this cell is ns
    and the one above it is µs. That is the whole reason the design works on
    the 131-of-235 problem pages whose T_SOL column spans more than one UNIT of
    the ladder. A unit is three decades, so that is the stronger statement of
    the two; counted by running `_laddered` over every `workload.t_sol_ms`.
    """
    classes = " ".join(c for c in ("q", cls, "q-needs-sort" if needs_sort else "") if c)
    attrs = f' title="{escape(title)}"' if title else ""
    unit_html = f'<span class="qu">{escape(unit)}</span>' if unit else ""
    # `escape` on the value, not just on the unit: `mins` renders "<1", and an
    # unescaped "<1" opens a tag.
    return Markup(f'<span class="{classes}"{attrs}>'
                  f'<span class="qv">{escape(value)}</span>{unit_html}</span>')


# ---- durations ------------------------------------------------------------

def dur_text(v) -> str:
    """A millisecond figure as text, on the ns/µs/ms/s ladder.

    The ladder covers the entire measured span on this board — 6e-08 ms (a
    B200 SOL figure, 0.06 ns) to 17314.3701171875 ms (17.3 s) — with every
    mantissa in [0.06, 999], so this kind NEVER needs scientific notation.
    The old `fmt_ms` switched style at 1e-4 ms, which put two notations in one
    T_SOL column wherever a problem's workloads straddled that threshold.

    The smallest T_SOL on the board, 7.692307692307693e-07 ms, is exactly one
    cycle at F_LOCK 1300 MHz. `0.769 ns` is the right thing for a reader to
    see: a one-cycle bound is the D39 signal, and `0.00000` hid it.

    Both spans measured this session against `leaderboard/db/solbench-MI350X.db`
    (opened `mode=ro`), over every positive `workload.t_sol_ms`, `t_b_ms`,
    `b200_baseline_ms`, `b200_sol_ms` and `result.latency_ms`: 28,112 distinct
    values, mantissae 0.06 … 999. (31,742 is the same count taken per column
    and summed, which double-counts a value two columns share.)
    """
    if v is None:
        return MISSING_TEXT
    if v == 0:
        return "0"
    m, unit = _laddered(float(v), _DUR_LADDER)
    return f"{m} {unit}"


def dur(v) -> Markup:
    if v is None:
        return MISSING
    if v == 0:
        # Not a small time: a duration recorded as exactly zero is a defect,
        # and it should be visible as one rather than rendered as `0.00 ns`.
        return _q("0", "", "q-zero",
                  "recorded as exactly zero — not a measurement")
    m, unit = _laddered(float(v), _DUR_LADDER)
    # A negative duration is likewise a defect. Marked, never hidden.
    return _q(m, unit, "q-bad" if v < 0 else "", needs_sort=True)


# ---- the score, S ---------------------------------------------------------

def score_text(v) -> str:
    """S, contract [0,1]. Fixed decimals on purpose.

    A bounded quantity with a landmark: T_b is S = 0.5 exactly, and 3,758
    cells on this board sit there by construction — `SELECT COUNT(*) FROM
    result WHERE score = 0.5` against `leaderboard/db/solbench-MI350X.db`
    (`mode=ro`), unchanged when joined to `submission.board_visible = 1`.
    `5.00e-01` would be the same number and a worse page.
    """
    if v is None:
        return MISSING_TEXT
    return f"{float(v):.4f}"


def score(v) -> Markup:
    if v is None:
        return MISSING
    s = f"{float(v):.4f}"
    if not 0.0 <= float(v) <= 1.0:
        # 27 of 7,073 `trajectory_eval.mean_score` rows are outside the
        # contract, the largest 32.354054. No cause has been investigated and
        # none is asserted here. The point is only that a run page renders
        # those in the same `.big` mono style as a board score of 0.4907,
        # which invites a comparison between two different quantities.
        return Markup(f'<span class="q-oor" title="outside the [0,1] contract'
                      f' — this is the agent harness’s own figure, not a'
                      f' board score">{s}</span>')
    return Markup(s)


def delta_text(v) -> str:
    if v is None:
        return MISSING_TEXT
    return f"{float(v):+.4f}"


def delta(v) -> Markup:
    return MISSING if v is None else Markup(delta_text(v))


# ---- bare multiples -------------------------------------------------------

def ratio_text(v) -> str:
    """A ×-multiple: headroom, speedup, geomean. 3 s.f. with a k/M suffix.

    The five extremes actually on the board, each read this session from
    `leaderboard/db/solbench-MI350X.db` (`mode=ro`): `workload.bound_headroom`
    runs 1.1097593673588968 → `1.11×` (as close to the roofline as anything
    gets) to 1499133.450644357 → `1.50M×`; `problem.median_headroom` tops out
    at 115004.95628074363 → `115k×` (L2__006); `trajectory_eval.geomean_speedup`
    runs 0.01520103232079208 → `0.015×` to 422.7051885272674 → `423×`.
    """
    if v is None:
        return MISSING_TEXT
    m, unit = _laddered(float(v), _RATIO_LADDER)
    return f"{m}{unit}"


def ratio(v) -> Markup:
    if v is None:
        return MISSING
    m, unit = _laddered(float(v), _RATIO_LADDER)
    return _q(m, unit, needs_sort=True)


# ---- pure magnitudes: the tolerances --------------------------------------

def sci_text(v) -> str:
    if v is None:
        return MISSING_TEXT
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(float(v))))
    return f"{float(v) / 10 ** e:.2f}e{e:+03d}"


def sci(v) -> Markup:
    """atol and rtol: no unit, no ladder, sixteen decades. Scientific is right.

    `<sup>`, not Unicode superscript characters: `⁻¹¹` has a real font-fallback
    risk in the JetBrains Mono / Menlo / Consolas stack and `<sup>` has none.

    Both columns get this style because atol and rtol sit adjacent and are read
    as a pair. Neither is narrow: over the board's nonzero tolerances, rtol runs
    1.192e-07 … 6.667e-01, 6.75 decades.
    """
    if v is None:
        return MISSING
    if v == 0:
        # 76 workloads across 5 problems. `if w.tol_atol` rendered every one
        # of them as "not recorded"; the truth is that an EXACT match is
        # required, which is a stricter statement than any number here.
        return _q("0", "", "q-zero",
                  "exactly zero — an exact match is required")
    e = math.floor(math.log10(abs(float(v))))
    m = f"{float(v) / 10 ** e:.2f}"
    return Markup(f'<span class="q q-needs-sort"><span class="qv">{m}'
                  f'×10<sup>{e}</sup></span></span>')


# ---- exact integers -------------------------------------------------------

def _as_int(v) -> int:
    """Counts arrive as ints from COUNT(*) and as strings from `meta`."""
    return v if isinstance(v, int) else int(float(v))


def cycles_text(v) -> str:
    """T_SOL in GPU cycles: the derived architectural quantity, exact.

    Audit columns get exact values; comparison columns get 3 s.f. This one is
    in the `c-deriv` group, it is what the millisecond column is computed
    from, and it is F_LOCK-invariant — so `398,131,200`, not `398M`.
    """
    return MISSING_TEXT if v is None else f"{_as_int(v):,}"


def cycles(v) -> Markup:
    return MISSING if v is None else Markup(cycles_text(v))


def n_text(v) -> str:
    return MISSING_TEXT if v is None else f"{_as_int(v):,}"


def n(v) -> Markup:
    return MISSING if v is None else Markup(n_text(v))


# ---- money, time, share, size ---------------------------------------------

def usd_text(v) -> str:
    if v is None:
        return MISSING_TEXT
    return f"${float(v):.2f}" if abs(float(v)) < 1000 else f"${float(v):,.0f}"


def usd(v) -> Markup:
    return MISSING if v is None else Markup(usd_text(v))


def mins_text(seconds) -> str:
    """A wall-clock span, given in SECONDS.

    One implementation of what run.html did inline and eight other sites
    approximated with `%.0f` of `/60`. `<1 min` rather than `0 min`: a session
    that ran for forty seconds did not run for no time.
    """
    if seconds is None:
        return MISSING_TEXT
    s = float(seconds)
    if s < 60:
        return "<1 min"
    m = int(round(s / 60))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def mins(seconds) -> Markup:
    if seconds is None:
        return MISSING
    txt = mins_text(seconds)
    if txt.startswith("<"):
        return _q("<1", "min", needs_sort=True)
    head, _, unit = txt.rpartition(" ")
    return _q(head, unit, needs_sort=True)


def pct_text(x) -> str:
    """A share already expressed 0–100.

    `<0.1%` rather than `0.0%`: a band with one workload in it is not an empty
    band, and rounding it to zero says it is.
    """
    if x is None:
        return MISSING_TEXT
    v = float(x)
    if 0 < v < 0.05:
        return "<0.1%"
    return f"{v:.1f}%"


def pct(x) -> Markup:
    return MISSING if x is None else Markup(escape(pct_text(x)))


def bytes_h_text(b) -> str:
    if b is None:
        return MISSING_TEXT
    if b == 0:
        return "0 B"
    m, unit = _laddered(float(b), _BYTE_LADDER)
    return f"{m} {unit}"


def bytes_h(b) -> Markup:
    if b is None:
        return MISSING
    if b == 0:
        return _q("0", "B", "q-zero")
    m, unit = _laddered(float(b), _BYTE_LADDER)
    return _q(m, unit, needs_sort=True)


def sortv(v) -> str:
    """The raw float for `data-sort`, or "" for a value that is not there.

    base.html treats a PRESENT-but-empty `data-sort` as null and sorts it last
    in either direction, which is what a missing measurement deserves. It did
    not until this was checked: the read was `getAttribute(...) || innerText`,
    and an empty attribute is falsy in JS, so the cell fell through to its own
    text. That happened to give the same answer wherever the text was the
    em-dash, and the wrong answer in run.html's trajectory "at" column, whose
    text is "not recorded". base.html now tests for the attribute's presence.
    The value is NOT
    rounded: `data-sort` is where the digits the 3-s.f. cell dropped still
    live, alongside `/api/v1`, which is untouched and remains the authority.
    """
    if v is None:
        return ""
    return repr(float(v))


for _name, _fn in (
        ("dur", dur), ("dur_text", dur_text),
        ("score", score), ("score_text", score_text),
        ("delta", delta), ("delta_text", delta_text),
        ("ratio", ratio), ("ratio_text", ratio_text),
        ("sci", sci), ("sci_text", sci_text),
        ("cycles", cycles), ("cycles_text", cycles_text),
        ("n", n), ("n_text", n_text),
        ("usd", usd), ("usd_text", usd_text),
        ("mins", mins), ("mins_text", mins_text),
        ("pct", pct), ("pct_text", pct_text),
        ("bytes_h", bytes_h), ("bytes_h_text", bytes_h_text),
        ("sortv", sortv)):
    templates.env.filters[_name] = _fn

# For the handful of non-numeric value positions that still need the dash --
# `{{ meta.part or missing }}`. It used to be written `'&mdash;'` without
# `|safe`, which autoescape renders as the literal text `&mdash;`.
templates.env.globals["missing"] = MISSING
templates.env.globals["na"] = na


def asset(path: str) -> str:
    """`/static/x.js` -> `/static/x.js?v=<hash of x.js>`.

    The cache-buster used to be a literal typed into the template, and it
    failed exactly the way a hand-maintained cache key fails: `style.css?v=12`
    got bumped twelve times because CSS changes are visible immediately, while
    `highlight.js?v=1` was never bumped once in the file's whole history. Every
    JS change since it was written -- including a fix to the line-number gutter
    -- was invisible to any browser that had ever loaded the board, and there
    was no symptom on this machine, because a fresh curl always gets fresh
    bytes. The person who sees the stale asset is never the person who changed
    it.

    Hashed at request time rather than at import: the board is served straight
    from a working tree during development, and an editor save has to be one
    reload away. Four small files, and the OS page cache holds all of them.
    """
    f = HERE / path.lstrip("/").removeprefix("static/")
    f = HERE / "static" / f.name
    try:
        return f"{path}?v={hashlib.sha256(f.read_bytes()).hexdigest()[:10]}"
    except OSError:
        # A missing asset is the template's problem to show, not ours to hide.
        return path


templates.env.globals["asset"] = asset

# Whether the header links the OpenAPI browser. Off by default: the API itself
# is untouched and every `/api/v1` route still serves, but nothing consumes it
# locally and `/api/docs` is Swagger UI -- its own bundle, the whole schema
# expanded -- which on the small public host is the heaviest page on the site
# and is reachable from every other page by every crawler. Documented in
# leaderboard/README.md; set SOLBENCH_API_NAV=1 to link it again.
templates.env.globals["api_nav"] = os.environ.get("SOLBENCH_API_NAV") == "1"


# --------------------------------------------------------------------------
# parts — which dataset a request is about
# --------------------------------------------------------------------------

class NoDataForPart(Exception):
    """A known part with no database. Not an error: an honest empty state.

    Raised rather than returned so that a page handler does not have to guard
    every query. It is answered by `no_data_for_part()` below with the empty
    state for pages and a 503 for the API -- never a 404, which would say the
    part does not exist, and never a 500, which would say something broke.
    """

    def __init__(self, part: str):
        super().__init__(part)
        self.part = part


def known_parts() -> list[str]:
    """The parts this port targets, in display order.

    Filtered on the ISA, not on a hardcoded name list: SOL-ExecBench-ROCm is a
    CDNA4 port, so gfx950 is exactly the set. MI300X is in `PARTS` because
    other tooling in the repo knows about it, but it is CDNA3 (gfx942, fnuz
    FP8, a different MAC/cycle table) and nothing here has ever been measured
    on it -- offering it in the switch would advertise a dataset that cannot
    exist without a separate port.
    """
    return sorted(name for name, p in PARTS.items() if p.gfx == "gfx950")


_meta_cache: dict[tuple[str, int], str | None] = {}
_count_cache: dict[tuple[str, int], int] = {}
_provisional_count_cache: dict[tuple[str, int], int] = {}
_mismatch_cache: dict[tuple[str, int], list[dict]] = {}


def _cached(cache: dict, path: Path, compute):
    """Memoise a cheap per-database fact, keyed by path AND mtime.

    `ingest.py` swaps a new file in with `os.replace`, so the mtime changes on
    every rebuild and a stale entry cannot survive one. Keyed by mtime rather
    than invalidated on a timer for that reason: the answer is only ever as old
    as the file it came from.

    The connection carries the same `row_factory` as `db()`. Without it a
    *compute* built on `rows()` works on every empty result and raises on the
    first non-empty one -- which for `part_mismatches` means it would pass
    every test on a clean board and fail only on the board it exists to warn
    about.
    """
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in cache:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                cache[key] = compute(conn)
            finally:
                conn.close()
        except sqlite3.Error:
            return None
    return cache[key]


def db_part(path: Path) -> str | None:
    """The part a database says it holds, from its own `meta`."""
    return _cached(_meta_cache, path, lambda c: (
        (c.execute("SELECT value FROM meta WHERE key='part'").fetchone() or [None])[0]))


def db_n_results(path: Path) -> int | None:
    """Measured workload results in a database — the switch's "count" figure."""
    return _cached(_count_cache, path, lambda c: (
        c.execute("SELECT COUNT(*) FROM result").fetchone()[0]))


def db_n_provisional(path: Path) -> int:
    """Unranked jobs in a database, including zero and old schemas."""
    def count(conn):
        if not table_exists(conn, "provisional_job"):
            return 0
        return conn.execute("SELECT COUNT(*) FROM provisional_job").fetchone()[0]
    return _cached(_provisional_count_cache, path, count)


def part_databases() -> dict[str, Path]:
    """Every part that has data, and the file holding it.

    Three sources, in this order:

    * `SOLBENCH_DB` pins one database and collapses the world to it. The tests
      and `worker.py` depend on that, and it must keep meaning "serve exactly
      this file" -- so a pin does not make the *other* parts resolve to it,
      which would be the mixing this whole split exists to prevent.
    * `db/solbench-<PART>.db`, the layout `ingest.py` writes today.
    * `leaderboard/solbench.db`, the single-file layout that predates the
      split, under whatever part its own `meta` names -- never under an
      assumed one -- and only for a part the per-part layout has not produced,
      so a fresh build always wins over a leftover.
    """
    pin = os.environ.get("SOLBENCH_DB")
    if pin:
        p = Path(pin)
        if not p.is_file():
            return {}
        # A pinned file with no `meta.part` is served under the default. That
        # is an operator's explicit choice of file, not an inference about
        # where its numbers came from; every database `ingest.py` writes names
        # its part.
        return {db_part(p) or DEFAULT_PART: p}

    out: dict[str, Path] = {}
    for name in known_parts():
        f = DB_DIR / f"solbench-{name}.db"
        if f.is_file():
            out[name] = f
    if LEGACY_DB.is_file():
        legacy = db_part(LEGACY_DB)
        if legacy and legacy not in out:
            out[legacy] = LEGACY_DB
    return out


def resolve_part(request: Request | None = None, explicit: str | None = None) -> str:
    """Which part's dataset this request is about (DESIGN-v2 §6).

    Query > cookie > `SOLBENCH_PART` > the only part with a database > MI350X.

    An unknown name in the query is a 400: silently serving MI350X to someone
    who asked for something else is how a reader ends up comparing two parts
    without knowing it. An unknown name in the *cookie* is ignored instead --
    a cookie outlives a rename, and one stale value should not brick every
    page for that browser. An unknown `SOLBENCH_PART` is server misconfigura-
    tion and fails loudly, because nobody is going to notice a warning.
    """
    known = known_parts()
    if explicit is not None:
        if explicit not in known:
            raise HTTPException(
                400, f"unknown part {explicit!r}: this build knows {known}")
        return explicit
    if request is not None:
        cookie = request.cookies.get(PART_COOKIE)
        if cookie in known:
            return cookie
    env = os.environ.get("SOLBENCH_PART")
    if env:
        if env not in known:
            raise HTTPException(
                500, f"SOLBENCH_PART={env!r} is not a part this build knows: {known}")
        return env
    dbs = part_databases()
    if len(dbs) == 1:
        return next(iter(dbs))
    return DEFAULT_PART


def part_infos(request: Request | None, active: str | None = None) -> list[dict]:
    """The switch's own data: every targeted part, whether it has anything.

    `n_results` is None where there is no database, never 0. "Not measured" and
    "measured nothing" are different statements and the switch has to be able
    to say which one it means.
    """
    dbs = part_databases()
    out = []
    for name in known_parts():
        path = dbs.get(name)
        out.append({
            "name": name,
            "available": path is not None,
            "n_results": db_n_results(path) if path else None,
            "n_provisional": db_n_provisional(path) if path else None,
            "active": name == active,
            "url": switch_url(request, name),
        })
    return out


def switch_url(request: Request | None, part: str) -> str:
    """This same page, on another part: path and query preserved, `part` set."""
    if request is None:
        return _with_query("/", part, {})
    return _with_query(request.url.path, part,
                       dict(parse_qsl(request.url.query, keep_blank_values=True)))


def _with_query(path: str, part: str | None, params: dict) -> str:
    split = urlsplit(path)
    q = dict(parse_qsl(split.query, keep_blank_values=True))
    q.update({k: v for k, v in params.items() if v is not None})
    if part:
        q["part"] = part
    return urlunsplit(("", "", split.path, urlencode(q), split.fragment))


@pass_context
def part_url(ctx, path: str, **params) -> str:
    """Jinja global: an internal link that carries the active part forward.

    Context-aware rather than bound per request, so a template never has to
    pass the part explicitly and cannot forget to. A link that drops `?part=`
    lands the reader back on the default dataset with no indication that the
    part changed under them -- which is the same failure as mixing the two
    datasets, just one click later.
    """
    return _with_query(path, params.pop("part", None) or ctx.get("part"), params)


templates.env.globals["part_url"] = part_url


@app.middleware("http")
async def sticky_part(request: Request, call_next):
    """`?part=` sets the cookie, so the choice survives the next click.

    One place rather than in each handler: "whenever the query parameter is
    used" includes the API and the empty state, and a handler that forgot it
    would give a switch that silently springs back. A session cookie, not a
    dated one -- the URL is the durable, shareable form of the choice.
    """
    response = await call_next(request)
    want = request.query_params.get("part")
    if want and want in known_parts():
        response.set_cookie(PART_COOKIE, want, path="/", samesite="lax")
    return response


def db(part: str | None = None) -> sqlite3.Connection:
    part = part or resolve_part()
    path = part_databases().get(part)
    if path is None:
        pin = os.environ.get("SOLBENCH_DB")
        if pin and not Path(pin).is_file():
            raise HTTPException(
                503, f"database not built: run leaderboard/ingest.py ({pin})")
        raise NoDataForPart(part)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def meta(part: str | None = None) -> dict:
    part = part or resolve_part()
    with db(part) as conn:
        m = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM meta")}
    m["freshness"] = freshness(m, part_databases().get(part))
    return m


def empty_meta(part: str) -> dict:
    """`meta` for a part with no database: the part, and nothing invented.

    Every other key is absent rather than blank, so the footer's F_LOCK, ROCm
    and torch line renders empty instead of showing the *other* part's values
    under this part's heading.
    """
    return {"part": part, "freshness": None}


def todo_runbook(part: str) -> str | None:
    """Repo-relative path of the runbook for `part`, or None if none is written."""
    rel = f"docs/TODO-{part}.md"
    return rel if (ROOT / rel).is_file() else None


@app.exception_handler(NoDataForPart)
def no_data_for_part(request: Request, exc: NoDataForPart) -> Response:
    if request.url.path.startswith("/api"):
        return JSONResponse(
            {"detail": f"nothing has been measured on {exc.part}; there is no "
                       f"database for it. The port needs no work for this "
                       f"part; the measurements do."}, status_code=503)
    infos = part_infos(request, exc.part)
    return templates.TemplateResponse(
        request, "part_missing.html",
        {"meta": empty_meta(exc.part), "part": exc.part, "parts": infos,
         "other_parts": [i for i in infos if i["name"] != exc.part],
         # Stated, not omitted: this page extends the same base as every other
         # one, and the empty list is the honest answer -- a part with no
         # database holds no submissions to be measured on the wrong part.
         "part_mismatch": [],
         # Checked on disk, not interpolated and hoped for. The old form was
         # f"TODO-{part}.md", which names a runbook for every part the config
         # admits -- including ones nobody ever wrote. Sending a reader to a
         # file that does not exist is worse than saying there is no runbook,
         # because it reads as "someone planned this".
         "todo_path": todo_runbook(exc.part), "nav": None})


def freshness(m: dict, path: Path | None = None) -> dict:
    """Is the database still a faithful view of the artifacts it was built from?

    It is a *derived* store, so it goes stale the moment those artifacts move,
    and the failure is silent: every page still renders and every number still
    looks plausible.

    This compares the INPUTS, not the repo. An earlier version compared git
    HEAD, which fired on every commit -- including one that only touched a
    stylesheet -- and would still have called the board fresh when the
    untracked `glm-run1` agent run appeared. Both failures came from asking
    git a question about data git does not track. See `inputs.py`.
    """
    out: dict = {"stale": None, "reasons": []}
    out["db_built_utc"] = m.get("db_built_utc")
    out["built_from_git_sha"] = m.get("repo_git_sha")   # provenance, not a check
    try:
        recorded = json.loads(m.get("input_signature") or "{}")
        extra = [Path(p) for p in json.loads(m.get("input_extra_roots") or "[]")]
        manifest = Path(m.get("input_manifest_path") or inputs.MANIFEST)
        provisional_raw = m.get("input_provisional_path")
        provisional = Path(provisional_raw) if provisional_raw else None
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        if provisional is not None and not provisional.is_absolute():
            provisional = ROOT / provisional
        current = inputs.signature(
            extra,
            manifest_path=manifest,
            provisional_path=provisional,
        )
        out["inputs"] = {"recorded": recorded, "current": current}
        if not recorded:
            out["reasons"] = ["the database recorded no input signature"]
            out["error"] = "freshness is unknown for a database built without a signature"
        elif recorded.get("n_files", 0) and current["n_files"] == 0:
            out["reasons"] = [
                "the build inputs are not present in this serving environment"
            ]
            out["error"] = (
                "freshness is unknown: the database can be served, but its "
                "manifest and result artifacts cannot be re-read here"
            )
        else:
            out["reasons"] = inputs.compare(recorded, current)
            out["stale"] = bool(out["reasons"])
        # The command must carry the roots this build actually used. A bare
        # `ingest.py` re-reads only artifacts/10, so following the banner
        # literally would silently drop every run kept outside the repo --
        # turning a freshness warning into a way to lose a submission.
        #
        # It must also name the file being SERVED. A bare `ingest.py` now
        # writes `db/solbench-<PART>.db`; run against a board served from the
        # legacy single-file path or from `SOLBENCH_DB`, it would rebuild a
        # different database and leave the stale banner up with nothing to
        # show for it. `--db` is emitted only where the two differ, so the
        # ordinary case keeps the short command.
        cmd = f"python leaderboard/ingest.py --manifest {manifest}"
        if path is not None and path != DB_DIR / f"solbench-{m.get('part')}.db":
            cmd += f" --db {path}"
        out["rebuild_command"] = cmd + (
            " --agent-runs " + " ".join(str(p) for p in extra) if extra else "")
    except Exception as exc:                             # never 500 a page over this
        out["reasons"] = []
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def rows(conn, sql: str, args=()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args)]


def table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------

def conn_part(conn) -> str | None:
    """The part this connection's database says it holds, from its own `meta`."""
    row = conn.execute("SELECT value FROM meta WHERE key='part'").fetchone()
    return row[0] if row else None


def part_mismatches(conn) -> list[dict]:
    """Submissions whose own part disagrees with the database they are stored in.

    This should always be empty: `ingest.py` refuses to write a run measured on
    one part into another part's database. It is checked again here because
    ranking is where that mistake would do its damage — an MI355X T_k scored
    against MI350X bounds is a plausible number with nothing detectably wrong,
    which §6 of DESIGN-v2 calls the most damaging kind of error this board can
    make. `leaderboard_rows()` drops such a row; this is what makes the drop
    visible, because a row that silently vanishes from a ranking is its own
    failure mode.
    """
    want = conn_part(conn)
    if want is None:
        return []
    return rows(conn, """SELECT slug, name, part AS submission_part
                           FROM submission
                          WHERE part IS NOT NULL AND part <> ?
                          ORDER BY id""", (want,))


def part_mismatches_for(part: str) -> list[dict]:
    """`part_mismatches()` for a part, for callers holding no connection.

    Cached on (path, mtime) like the other per-database facts, because every
    page render asks and the answer only changes when `ingest.py` swaps a new
    file in. A part with no database has no submissions and so no mismatch --
    `[]` rather than a raise, since the empty state is a page too.
    """
    path = part_databases().get(part)
    if path is None:
        return []
    return _cached(_mismatch_cache, path, part_mismatches) or []


def known_categories(conn) -> list[str]:
    """The categories this database actually holds, in listing order."""
    return [r["category"] for r in conn.execute(
        "SELECT DISTINCT category FROM problem ORDER BY category")]


def check_category(conn, category: str | None) -> None:
    """An unknown `?category=` is a 400, exactly like an unknown `?part=`.

    It used to be a filter that matched nothing, and the board answered with a
    full table of 0.0000 scores and empty coverage bars -- every submission
    rendered as having achieved nothing, on a benchmark subset that does not
    exist. A page of plausible zeros nobody measured is the failure mode this
    whole repo is organised against; refusing the request is the only reading
    that is true.
    """
    if category is None:
        return
    known = known_categories(conn)
    if category not in known:
        raise HTTPException(
            400, f"unknown category {category!r}: this board has {known}")


def scoreable_totals(conn, category: str | None = None) -> dict:
    """The denominators the board divides by, in the scope on screen.

    One implementation, read both by `leaderboard_rows()` and by the page that
    LABELS those rows -- because the two disagreed. The rows honour
    `?category=`; the labels above them quoted the manifest's whole-benchmark
    figures, so `/?category=L1` printed "divide by all 3,717 scoreable
    workloads" and "coverage -- all 220 problems" over a table divided by
    1,480 workloads across 94. A mislabelled denominator is not a cosmetic
    defect: it is the reader being told which question the number answers,
    wrongly.
    """
    return {
        "workloads": conn.execute(
            f"""SELECT COUNT(*) FROM workload w WHERE w.scoreable=1
                {'AND w.problem_key LIKE ?' if category else ''}""",
            (f"{category}__%",) if category else ()).fetchone()[0],
        "problems": conn.execute(
            f"""SELECT COUNT(*) FROM problem p WHERE p.n_scoreable > 0
                {'AND p.category = ?' if category else ''}""",
            (category,) if category else ()).fetchone()[0],
    }


def leaderboard_rows(conn, category: str | None = None) -> list[dict]:
    """One row per submission. Missing workloads score zero, by construction."""
    where_cat = "AND w.problem_key LIKE ?" if category else ""
    cat_arg = (f"{category}__%",) if category else ()

    # Defence in depth against a part mix, paired with `part_mismatches()`
    # above, which is what tells the reader a row was dropped. A NULL `part` is
    # not a disagreement — it is a run whose artifacts never named one — and
    # treating it as one would silently empty the board of every submission
    # ingested before the column existed.
    want_part = conn_part(conn)
    part_guard = "AND (part IS NULL OR part = ?)" if want_part else ""
    part_arg = (want_part,) if want_part else ()

    totals = scoreable_totals(conn, category)
    total = totals["workloads"]
    total_problems = totals["problems"]

    out = []
    # `board_visible = 0` is read HERE and nowhere in the ingest of results, so
    # hiding a run cannot move a visible run's number. pilot8 is the case: it
    # really ran and its evidence is reachable, but every one of its sessions
    # was stopped by the budget cap, so its mean is survivorship over whatever
    # happened to finish and ranking it would invite a comparison nobody made.
    for s in rows(conn, f"SELECT * FROM submission WHERE board_visible=1 "
                        f"{part_guard} ORDER BY id", part_arg):
        agg = conn.execute(
            f"""SELECT COUNT(*) AS n_passed,
                       COALESCE(SUM(r.score),0) AS score_sum,
                       COALESCE(AVG(r.score),0) AS score_mean
                  FROM result r
                  JOIN workload w ON w.problem_key=r.problem_key
                                 AND w.uuid=r.workload_uuid
                 WHERE r.submission_id=? AND r.status='PASSED'
                   AND r.score IS NOT NULL AND w.scoreable=1 {where_cat}""",
            (s["id"], *cat_arg)).fetchone()
        # Flagged is counted on its OWN, over every result, and that is the
        # whole point. It used to live in the aggregate above, whose WHERE is
        # `status='PASSED' AND score IS NOT NULL` -- and a flagged workload has
        # status REWARD_HACK and a NULL score, so each of those clauses on its
        # own excluded it. The counter could not return anything but zero.
        #
        # The board read "0 flagged" from the day it was built, /methodology
        # said so in prose, and on 2026-08-10 the harness caught 48 real ones
        # and the column still read 0. A negative result that is guaranteed by
        # construction is not evidence of anything.
        n_flagged = conn.execute(
            f"""SELECT COALESCE(SUM(r.flagged),0)
                  FROM result r
                  JOIN workload w ON w.problem_key=r.problem_key
                                 AND w.uuid=r.workload_uuid
                 WHERE r.submission_id=? {where_cat}""",
            (s["id"], *cat_arg)).fetchone()[0]
        attempted = conn.execute(
            f"""SELECT COUNT(*) FROM result r
                  JOIN workload w ON w.problem_key=r.problem_key
                                 AND w.uuid=r.workload_uuid
                 WHERE r.submission_id=? AND w.scoreable=1 {where_cat}""",
            (s["id"], *cat_arg)).fetchone()[0]
        # Every scoreable problem, in one of four states, adding up to the whole
        # benchmark. This replaces the old `coverage` + `problems` pair, which
        # asked overlapping questions in incompatible units -- one counted
        # workloads out of 3,717, the other counted problems the submission had
        # swept clean, and neither said how many problems it had tried and
        # failed. A reader could not get "how much of the benchmark is this"
        # out of them at all.
        #
        # LEFT JOIN from `problem`, so a problem with no results is still a row
        # and lands in `untouched`. Grouping the other way round can only ever
        # count problems that produced something.
        buckets = conn.execute(
            # `nflag > 0` is tested FIRST, so a problem carrying a reward hack
            # lands in the flagged bucket whatever else it did.
            #
            # On today's board this is indistinguishable from carving flagged
            # out of `failed`: all three flagged problems have all 16 of their
            # workloads flagged and none passing, so both rules give 2 and 1.
            # Priority is chosen for the case that is not on the board yet -- a
            # problem with some passing workloads and one refused kernel. That
            # entry has not cleanly solved the problem, and the carve-out rule
            # would draw it green.
            #
            # (An earlier revision of this comment justified priority by
            # claiming the carve-out renders empty on the real board. That was
            # read off `leaderboard/solbench.db`, a stale database that does not
            # contain either agent run. It is wrong and the numbers above are
            # the check.)
            f"""SELECT SUM(CASE WHEN seen = 0 THEN 1 ELSE 0 END) AS untouched,
                       SUM(CASE WHEN seen > 0 AND nflag > 0
                                THEN 1 ELSE 0 END) AS flagged,
                       SUM(CASE WHEN seen > 0 AND ok = 0 AND nflag = 0
                                THEN 1 ELSE 0 END) AS failed,
                       SUM(CASE WHEN ok > 0 AND ok < n_scoreable AND nflag = 0
                                THEN 1 ELSE 0 END) AS partial,
                       SUM(CASE WHEN ok > 0 AND ok = n_scoreable AND nflag = 0
                                THEN 1 ELSE 0 END) AS clean,
                       COUNT(*) AS n
                  FROM (
                    SELECT p.key, p.n_scoreable,
                           COUNT(r.workload_uuid) AS seen,
                           SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS ok,
                           COALESCE(SUM(r.flagged), 0) AS nflag
                      FROM problem p
                      JOIN workload w ON w.problem_key=p.key AND w.scoreable=1
                      LEFT JOIN result r ON r.problem_key=w.problem_key
                                        AND r.workload_uuid=w.uuid
                                        AND r.submission_id=?
                     WHERE p.n_scoreable > 0 {'AND p.category = ?' if category else ''}
                     GROUP BY p.key)""",
            (s["id"], *((category,) if category else ()))).fetchone()
        clean = buckets["clean"] or 0
        touched = (buckets["n"] or 0) - (buckets["untouched"] or 0)

        out.append({
            **s,
            "workloads_total": total,
            "workloads_passed": agg["n_passed"],
            "workloads_attempted": attempted,
            # Everything the benchmark score counts as zero, split by WHY it is
            # zero. Untested and failed are both zeroes in the headline, but
            # they mean different things about the submission, and a reader who
            # cannot tell them apart cannot tell a partial run from a bad one.
            "workloads_untested": total - attempted,
            "workloads_failed": attempted - agg["n_passed"],
            "partial": attempted < total,
            "problems_total": total_problems,
            "problems_complete": clean,
            "problems_attempted": touched,
            # The four states of the coverage bar, over problems, summing to
            # `problems_total`. Percentages are not precomputed: the bar wants
            # them of the whole benchmark and the tooltip wants the counts, and
            # a rounded percentage that no longer sums to 100 is the classic way
            # for a stacked bar to end with a 1px gap nobody can explain.
            "problems_clean": clean,
            "problems_partial": buckets["partial"] or 0,
            "problems_failed": buckets["failed"] or 0,
            "problems_flagged": buckets["flagged"] or 0,
            "problems_untouched": buckets["untouched"] or 0,
            # ONE score, two scopes, and the board switches between them rather
            # than printing both side by side -- see the `scope` note below.
            #
            #   full      : divided by every scoreable workload. Never attempted
            #               is a zero exactly like failed.
            #   attempted : divided by what this submission was actually run on.
            #               A failed attempt is still a zero; only workloads it
            #               never saw leave the denominator.
            #
            # The API keys keep their original names, because they already meant
            # exactly these two things and renaming them would break every
            # consumer to gain nothing.
            "benchmark_score": (agg["score_sum"] / total) if total else 0.0,
            "mean_score_attempted": (agg["score_sum"] / attempted) if attempted else 0.0,
            "mean_score_passed": agg["score_mean"],
            "coverage": (agg["n_passed"] / total) if total else 0.0,
            "coverage_attempted": (agg["n_passed"] / attempted) if attempted else 0.0,
            "n_flagged": n_flagged,
        })
    # Two orderings, both computed here, because the ranking is not a view of
    # the score -- it IS the score's scope, and a rank recomputed in the browser
    # from four decimal places of printed text would disagree with this one on
    # ties. `rank` stays the full-benchmark rank so nothing that reads the API
    # today changes meaning.
    for key, field in (("rank", "benchmark_score"),
                       ("rank_attempted", "mean_score_attempted")):
        for i, r in enumerate(
                sorted(out, key=lambda r: (-r[field], -r["workloads_attempted"])), 1):
            r[key] = i
    out.sort(key=lambda r: r["rank_attempted"])
    return out


def provisional_rows(conn, category: str | None = None) -> list[dict]:
    """Visible evidence rows that are structurally absent from the ranking."""
    if not table_exists(conn, "provisional_job"):
        return []
    category_where = "AND p.problem_key LIKE ?" if category else ""
    args = (f"{category}__%",) if category else ()
    got = rows(
        conn,
        f"""SELECT s.slug, s.name, s.model,
                   COUNT(*) AS jobs,
                   COUNT(DISTINCT p.problem_key) AS problems,
                   SUM(p.selected) AS kernels,
                   MAX(COALESCE(p.finished_utc,p.created_utc)) AS latest_utc
              FROM submission s
              JOIN provisional_job p ON p.submission_id=s.id
             WHERE s.kind='provisional' {category_where}
             GROUP BY s.id
             ORDER BY LOWER(s.model)""",
        args,
    )
    for row in got:
        row["evidence_tier"] = "provisional"
        row["ranked"] = False
        row["url"] = f"/submissions/{row['slug']}"
    return got


def provisional_jobs(conn, submission_id: int, slug: str) -> list[dict]:
    got = rows(
        conn,
        """SELECT job_id,task_id,task_name,problem_key,model,created_utc,
                  finished_utc,submission_n,validation_note,kernel_sha256,
                  kernel_bytes,evidence,selected
             FROM provisional_job
            WHERE submission_id=?
            ORDER BY problem_key,created_utc,job_id""",
        (submission_id,),
    )
    for row in got:
        row["url"] = (
            f"/submissions/{slug}/problems/{row['problem_key']}"
            if row["selected"]
            else f"/submissions/{slug}"
        )
    return got


def problem_rows(conn, category: str | None = None) -> list[dict]:
    # Headline numbers, so `board_visible = 0` is excluded from both. These are
    # the figures a reader ranks problems by; an off-board run may be read but
    # it may not be the number at the top of a page. pilot8 is the case that
    # forced this: every one of its sessions was stopped by its $8 cap, so it
    # is a cost measurement rather than a score measurement, and unfiltered it
    # took L1__053's best from 0.5 to 0.9788 and FlashInfer-Bench__019's from
    # 0.99685 to 0.99977 -- the latter against a bound already known wrong
    # (D18). It still appears in the per-problem submissions table below,
    # flagged: declining to rank a run and deleting its evidence are different
    # decisions and only the first one was made.
    sql = """SELECT p.*,
                    (SELECT MAX(r.score) FROM result r
                       JOIN submission s ON s.id = r.submission_id
                      WHERE r.problem_key=p.key AND r.status='PASSED'
                        AND s.board_visible=1) AS best_score,
                    (SELECT COUNT(DISTINCT r.submission_id) FROM result r
                       JOIN submission s ON s.id = r.submission_id
                      WHERE r.problem_key=p.key AND r.status='PASSED'
                        AND s.board_visible=1) AS n_submissions
               FROM problem p"""
    args: tuple = ()
    if category:
        sql += " WHERE p.category = ?"
        args = (category,)
    sql += " ORDER BY p.category, p.name"
    return rows(conn, sql, args)


def problem_detail(conn, key: str) -> dict:
    # "Best score recorded" is the page's headline, so it reads the ranked
    # runs only -- same rule as `problem_rows()` above, and for the same
    # reason. The evidence paths below (per-workload results, the per-
    # submission table) still carry the off-board runs, flagged.
    p = conn.execute("""
        SELECT p.*,
               (SELECT MAX(r.score) FROM result r
                  JOIN submission s ON s.id = r.submission_id
                 WHERE r.problem_key=p.key AND r.status='PASSED'
                   AND s.board_visible=1) AS best_score
          FROM problem p WHERE p.key=?""", (key,)).fetchone()
    if p is None:
        raise HTTPException(404, f"no such problem: {key}")
    p = dict(p)
    for f in ("axes_json", "inputs_json", "outputs_json"):
        p[f.replace("_json", "")] = json.loads(p.pop(f) or "{}")

    # Dataset order, not manifest order. The manifest sorts workloads by uuid,
    # which is an ordering nobody else shares; `dataset_index` is the position
    # in the dataset's own workload.jsonl and therefore the position upstream
    # lists the same workload at. Rows with no index (no dataset checked out)
    # keep the old ordering behind the ones that have one.
    wls = rows(conn, """SELECT * FROM workload WHERE problem_key=?
                         ORDER BY (dataset_index IS NULL), dataset_index, rowid""",
               (key,))
    for w in wls:
        w["axes"] = json.loads(w.pop("axes_json") or "{}")
        # Read, not recomputed. `workload.bound_headroom` is `T_b / T_SOL`
        # against the PUBLISHED bound, written at ingest by the same call that
        # assigns `bound_quality`. Dividing the two columns here produced a
        # SECOND answer to the same question, and the two disagreed wherever the
        # manifest's legacy `t_sol_ms` and its published bound do -- 3685 of
        # 3717 workloads on MI355X manifest-v4 -- so one row could print a
        # headroom and a quality word derived from different bounds (D63; see
        # `published_bound_ms` in ingest.py).
        #
        # NULL rather than a division also carries the old guard forward: a
        # deferred problem has a T_SOL (architectural, needs no GPU) and no T_b,
        # and `bound_quality` returns (None, None) there -- which is what stops
        # a None/float from 500'ing all 15 deferred problem pages.
        w["headroom"] = w["bound_headroom"]
        # Off-board submissions are included here, and carry the flag that says
        # so. `board_visible=0` takes a run out of the RANKING; its per-workload
        # timings are still measurements that happened, and dropping them from
        # the problem page would be deleting the evidence rather than declining
        # to rank it -- the distinction the flag exists to preserve. The row
        # says which, so no reader mistakes one for a ranked result.
        w["results"] = rows(conn, """
            SELECT r.*, s.slug, s.name AS submission_name, s.kind,
                   s.board_visible, s.exclusion_reason
              FROM result r JOIN submission s ON s.id=r.submission_id
             WHERE r.problem_key=? AND r.workload_uuid=?
             ORDER BY (r.score IS NULL), r.score DESC""", (key, w["uuid"]))

    # No FILTER and no NULLS LAST: the system SQLite here is 3.26, which
    # predates both (3.30). They parse as syntax errors, not as ignored hints.
    #
    # `scored` is not decoration. AVG() skips NULL, and a PASSED result scores
    # NULL when the kernel beat T_SOL -- the bound is invalid, so there is no
    # defensible score. That makes AVG's denominator vary per row: on
    # FlashInfer-Bench__019, pilot8's mean is over 13 workloads (D18 voided the
    # other 25) while every other run's is over all 38. Printing both means
    # side by side under one "mean S" heading ranks 13 against 38. Nothing
    # here changes a number; it says what each one was divided by.
    per_sub = rows(conn, """
        SELECT s.slug, s.name, s.kind, s.model, s.board_visible, s.exclusion_reason,
               COUNT(*) AS attempted,
               SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS passed,
               SUM(CASE WHEN r.status='PASSED' AND r.score IS NOT NULL
                        THEN 1 ELSE 0 END) AS scored,
               AVG(CASE WHEN r.status='PASSED' THEN r.score END) AS mean_score,
               MIN(CASE WHEN r.status='PASSED' THEN r.latency_ms END) AS best_latency,
               SUM(r.flagged) AS flagged
          FROM result r JOIN submission s ON s.id=r.submission_id
         WHERE r.problem_key=?
         GROUP BY s.id
         ORDER BY (mean_score IS NULL), mean_score DESC""", (key,))

    # Which axes actually MOVE across this problem's workloads. The workload
    # table prints every axis, the way upstream does -- `head_dim=128` is part
    # of the shape whether or not it varies -- but the results table below it
    # is one row per workload PER SUBMISSION, and seven identical chips on 288
    # rows is noise that hides the one or two that differ. Same data, and the
    # set is derived rather than declared so it cannot go stale.
    varying = sorted({k for w in wls for k, v in w["axes"].items()
                      if any(o["axes"].get(k) != v for o in wls)})
    return {"problem": p, "workloads": wls, "submissions": per_sub,
            "varying_axes": varying}


def trials(conn, s: dict, key: str | None = None) -> list[dict]:
    """Every trial of this submission's setup — the same setup, run again.

    A trial is a whole run, so the group is a set of `submission` rows and this
    is a query over them, not an extra dimension on `result`. Ungrouped
    submissions get an empty list: a group of one is not a trial, and the UI
    shows the switcher only when there is something to switch to.

    With a *key*, each trial also carries its outcome on that one problem. The
    trials are not merged and no aggregate spans them: two runs under different
    budgets are two measurements, and averaging them would invent a third.
    """
    if not s.get("group_slug"):
        return []
    # `touched` is not `attempted > 0`. A kernel whose re-time timed out
    # produces no result rows at all (D23, glm-run1 on FlashInfer-Bench__014),
    # and rendering that as "not in this trial" would turn a failed measurement
    # into a problem the trial never opened.
    #
    # The denominator is ATTEMPTED SCOREABLE workloads and the numerator sums
    # scores with a missing one contributing zero -- exactly what the run
    # page's own summary computes for `mean_attempted`, deliberately, because
    # the switcher and the card sit six lines apart on the same page and must
    # not print two different numbers under one label. `AVG(score)` was the bug:
    # SQL skips NULLs, and a bound-invalid row stores NULL, so pilot8 on
    # FlashInfer-Bench__019 read 0.9899 in the switcher against 0.3387 in the
    # card. The NULL there is not a missing measurement to be averaged around;
    # it is a workload whose bound is wrong, which earns nothing.
    out = rows(conn, """
        SELECT t.slug, t.name, t.trial_label, t.trial_n, t.constraint_json,
               t.board_visible, t.exclusion_reason,
               SUM(CASE WHEN w.uuid IS NOT NULL THEN 1 ELSE 0 END) AS attempted,
               SUM(CASE WHEN w.uuid IS NOT NULL AND r.status='PASSED'
                        THEN 1 ELSE 0 END) AS passed,
               SUM(CASE WHEN w.uuid IS NOT NULL AND r.status='PASSED'
                        THEN COALESCE(r.score, 0) ELSE 0 END) AS score_sum,
               -- Counted so the switcher can say why "38/38 passed" sits next
               -- to a mean of 0.34: the zeroes are beaten bounds, not failures.
               SUM(CASE WHEN w.uuid IS NOT NULL AND r.status='PASSED'
                        AND r.score IS NULL THEN 1 ELSE 0 END) AS unbounded,
               EXISTS(SELECT 1 FROM run_kernel k
                       WHERE k.submission_id=t.id AND k.problem_key=?) AS has_kernel
          FROM submission t
          LEFT JOIN result r ON r.submission_id=t.id AND r.problem_key=?
          LEFT JOIN workload w ON w.problem_key=r.problem_key
                              AND w.uuid=r.workload_uuid AND w.scoreable=1
         WHERE t.group_slug=?
         GROUP BY t.id
         ORDER BY (t.trial_n IS NULL), t.trial_n, t.id""",
                (key or "", key or "", s["group_slug"]))
    for t in out:
        score_sum = t.pop("score_sum") or 0.0
        t["mean_score"] = (score_sum / t["attempted"]) if t["attempted"] else None
        t["constraint"] = json.loads(t.pop("constraint_json") or "{}")
        t["is_current"] = t["slug"] == s["slug"]
        t["url"] = (f"/submissions/{t['slug']}/problems/{key}" if key
                    else f"/submissions/{t['slug']}")
        if key is None:
            # No problem in context, so there is no per-problem outcome to
            # report. None, not zero: this trial did not score nothing here.
            t["touched"] = t["attempted"] = t["passed"] = t["mean_score"] = None
        else:
            t["touched"] = bool(t["attempted"] or t["has_kernel"])
        t.pop("has_kernel", None)
    return out


def submission_detail(conn, slug: str) -> dict:
    s = conn.execute("SELECT * FROM submission WHERE slug=?", (slug,)).fetchone()
    if s is None:
        raise HTTPException(404, f"no such submission: {slug}")
    s = dict(s)
    s["provenance"] = json.loads(s.pop("provenance_json") or "{}")
    # The constraint this trial ran under, as its own artifact recorded it.
    # `board_visible` and `exclusion_reason` come through in the `SELECT *`:
    # a run that is off the ranking says so on its own page, or the reader has
    # no way to tell why it is not on the board.
    s["constraint"] = json.loads(s.pop("constraint_json") or "{}")
    per_problem = rows(conn, """
        SELECT p.key, p.category, p.name, p.n_scoreable,
               SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS passed,
               COUNT(*) AS attempted,
               -- See `scored` in problem_detail(): AVG skips the NULL score a
               -- beaten bound produces, so the denominator is per-row.
               SUM(CASE WHEN r.status='PASSED' AND r.score IS NOT NULL
                        THEN 1 ELSE 0 END) AS scored,
               AVG(CASE WHEN r.status='PASSED' THEN r.score END) AS mean_score,
               SUM(r.flagged) AS flagged,
               e.cost_usd, e.wall_seconds, e.turns, e.harness_evals
          FROM result r
          JOIN problem p ON p.key=r.problem_key
          LEFT JOIN run_effort e ON e.submission_id=r.submission_id
                                AND e.problem_key=r.problem_key
         WHERE r.submission_id=?
         GROUP BY p.key
         ORDER BY (mean_score IS NULL), mean_score DESC""", (s["id"],))

    # Kernels that produced no results at all. Without this they are absent
    # from the submission page entirely -- the agent wrote a kernel, the
    # re-time failed, and the run looks like it never touched the problem.
    unmeasured = rows(conn, """
        SELECT k.problem_key AS key, p.category, p.name, p.n_scoreable,
               k.n_lines, k.retime_error
          FROM run_kernel k JOIN problem p ON p.key = k.problem_key
         WHERE k.submission_id = ?
           AND NOT EXISTS (SELECT 1 FROM result r
                            WHERE r.submission_id = k.submission_id
                              AND r.problem_key = k.problem_key)
         ORDER BY k.problem_key""", (s["id"],))

    provisional = (
        provisional_jobs(conn, s["id"], s["slug"])
        if s["kind"] == "provisional"
        else []
    )
    return {"submission": s, "problems": per_problem, "unmeasured": unmeasured,
            "trials": trials(conn, s), "provisional_jobs": provisional}


def run_detail(conn, slug: str, key: str) -> dict:
    """One submission on one problem — the cell where the board's two axes meet.

    Rendered per request from the same tables as everything else. It is
    deliberately not pre-generated: there are 6 submissions x 235 problems =
    1410 of these today and the product grows with every run added, so a static
    build would be 1410 files that go stale the moment `ingest.py` runs again,
    which is exactly the failure mode the freshness check exists to catch.

    Rows come from `workload` LEFT JOINed to `result`, not from `result` alone.
    That is the whole point of the page: a workload this submission never
    attempted still has a T_SOL and a T_b, and it still contributes a zero to
    the benchmark score. Driving the table off `result` would silently omit
    precisely the rows that explain the score.
    """
    s = conn.execute("SELECT * FROM submission WHERE slug=?", (slug,)).fetchone()
    if s is None:
        raise HTTPException(404, f"no such submission: {slug}")
    p = conn.execute("SELECT * FROM problem WHERE key=?", (key,)).fetchone()
    if p is None:
        raise HTTPException(404, f"no such problem: {key}")
    s, p = dict(s), dict(p)
    s["provenance"] = json.loads(s.pop("provenance_json") or "{}")
    s["constraint"] = json.loads(s.pop("constraint_json") or "{}")
    for f in ("axes_json", "inputs_json", "outputs_json"):
        p.pop(f, None)

    wls = rows(conn, """
        SELECT w.uuid, w.axes_json, w.scoreable, w.t_sol_ms, w.t_sol_source,
               w.sol_bottleneck, w.t_b_ms, w.t_b_variant, w.bound_headroom,
               r.status, r.latency_ms, r.score, r.flagged, r.note
          FROM workload w
          LEFT JOIN result r ON r.problem_key = w.problem_key
                            AND r.workload_uuid = w.uuid
                            AND r.submission_id = ?
         WHERE w.problem_key = ?
         ORDER BY w.rowid""", (s["id"], key))

    kernel = conn.execute(
        """SELECT source, n_lines, sha256, retime_ok, retime_error
             FROM run_kernel WHERE submission_id=? AND problem_key=?""",
        (s["id"], key)).fetchone()
    # D23. A kernel exists for this problem and its authoritative re-time did
    # not complete, so none of its workloads have result rows. "Tried, could
    # not be measured" is not "did not try", and `retime_ok` exists precisely
    # to keep the two apart -- without this the grid draws thirty cells reading
    # "not attempted" directly above a banner saying a kernel was submitted and
    # its re-time hit TimeoutExpired (glm-run1 on FlashInfer-Bench__014).
    # `retime_ok == 0` and not `not retime_ok`: NULL means the ingest recorded
    # nothing about the re-time, which is a third state and not this one.
    retime_failed = kernel is not None and kernel["retime_ok"] == 0

    n_att = n_pass = 0
    score_sum = 0.0
    for w in wls:
        w["axes"] = json.loads(w.pop("axes_json") or "{}")
        w["attempted"] = w["status"] is not None
        w["passed"] = w["status"] == "PASSED"
        # Speedup against the anchor, which is the number a kernel author
        # actually feels. S is the scored quantity but it is compressed; 1.8x
        # faster than optimized PyTorch reads as an outcome, S=0.64 does not.
        # Only on a pass, for the same reason the score is: T_k on a failed
        # workload is how fast the wrong answer was produced, and "0.99x vs
        # optimized PyTorch" printed next to FAILED reads as near-parity.
        w["speedup_vs_tb"] = ((w["t_b_ms"] / w["latency_ms"])
                              if w["passed"] and w["t_b_ms"] and w["latency_ms"]
                              else None)
        # The ingest's figure, against the published bound -- same reason as on
        # the problem page: one question, one answer, and no page recomputing a
        # ratio the database already holds against a different T_SOL column.
        w["headroom"] = w["bound_headroom"]
        # A score present on a row whose bound was beaten is impossible by
        # construction: ingest stores NULL there. Surface the reason instead of
        # an empty cell, or the page reads as a missing measurement.
        w["bound_invalid"] = (w["passed"] and w["score"] is None
                              and bool(w["latency_ms"]))
        w["unmeasured"] = retime_failed and not w["attempted"]
        if w["attempted"] and w["scoreable"]:
            n_att += 1
            if w["passed"]:
                n_pass += 1
                score_sum += w["score"] or 0.0

    n_scoreable = p["n_scoreable"] or 0
    summary = {
        "n_scoreable": n_scoreable,
        "attempted": n_att,
        "passed": n_pass,
        "failed": n_att - n_pass,
        "untested": max(0, n_scoreable - n_att),
        # Same three denominators as the board, for the same reason.
        "mean_attempted": (score_sum / n_att) if n_att else None,
        "mean_passed": (score_sum / n_pass) if n_pass else None,
        "problem_score": (score_sum / n_scoreable) if n_scoreable else None,
        # Passed rows only. Taking the max over every score let a submission
        # that passed nothing on this problem still advertise a "best workload
        # S", drawn from a row whose correctness check had failed.
        "best": max((w["score"] for w in wls
                     if w["passed"] and w["score"] is not None), default=None),
        "n_flagged": sum(1 for w in wls if w["flagged"]),
    }

    peers = rows(conn, """
        SELECT s.slug, s.name, s.kind, s.board_visible, s.exclusion_reason,
               SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS passed,
               -- See `scored` in problem_detail(). This is the column a reader
               -- most naturally reads as a ranking of peers, so it is the one
               -- where an unstated denominator does the most damage.
               SUM(CASE WHEN r.status='PASSED' AND r.score IS NOT NULL
                        THEN 1 ELSE 0 END) AS scored,
               AVG(CASE WHEN r.status='PASSED' THEN r.score END) AS mean_score
          FROM result r JOIN submission s ON s.id = r.submission_id
         WHERE r.problem_key = ?
         GROUP BY s.id
         ORDER BY (mean_score IS NULL), mean_score DESC""", (key,))

    # Which T_b formulations actually anchored this problem, and on how many
    # workloads. Not "the baseline", plural: T_b is chosen per workload, so a
    # single problem is routinely anchored by two or three different variants
    # and a page that names only one is showing the wrong side of the diff for
    # the rest.
    won = {r["t_b_variant"]: r["n"] for r in conn.execute(
        """SELECT t_b_variant, COUNT(*) AS n FROM workload
            WHERE problem_key=? AND t_b_variant IS NOT NULL
            GROUP BY t_b_variant""", (key,))}
    variants = []
    for vname, n in sorted(won.items(), key=lambda kv: -kv[1]):
        row = conn.execute(
            "SELECT source, n_lines FROM variant_source WHERE problem_key=? AND variant=?",
            (key, vname)).fetchone()
        if row:
            variants.append({"variant": vname, "source": row["source"],
                             "n_lines": row["n_lines"], "won_workloads": n})

    # What THIS submission ran, when it is a reference variant. Distinct from
    # `variants` above, which is "whichever transforms won T_b here" and is the
    # same list for every viewer of this problem. A variant row has no `kernel`
    # -- nothing was authored -- but the code it executed is not therefore
    # unknown, and the pane used to say it was.
    own_variant = None
    if s.get("variant"):
        row = conn.execute(
            "SELECT source, n_lines FROM variant_source "
            "WHERE problem_key=? AND variant=?", (key, s["variant"])).fetchone()
        if row:
            own_variant = {"variant": s["variant"], "source": row["source"],
                           "n_lines": row["n_lines"],
                           # Whether this transform is also the anchor here. If
                           # it is, the score is 0.5 by construction, and saying
                           # so on the page it is visible from is cheaper than
                           # having a reader rediscover it.
                           "won_workloads": won.get(s["variant"], 0)}

    traj = rows(conn, """
        SELECT n, utc, minutes_in, ok, all_passed, passed, workloads,
               geomean_speedup, mean_score, kernel_sha, kernel_lines
          FROM trajectory_eval
         WHERE submission_id=? AND problem_key=? ORDER BY n""", (s["id"], key))
    _mark_regressions(traj)
    _label_times(traj)

    effort = conn.execute(
        "SELECT * FROM run_effort WHERE submission_id=? AND problem_key=?",
        (s["id"], key)).fetchone()
    tr = conn.execute(
        "SELECT bytes, n_lines, n_turns, tools_json FROM transcript "
        "WHERE submission_id=? AND problem_key=?", (s["id"], key)).fetchone()

    return {
        "submission": s, "problem": p, "workloads": wls,
        "summary": summary, "peers": peers,
        "trials": trials(conn, s, key),
        "window": run_window(conn, s["id"], key),
        # The part this RUN was measured on, from its own re-time provenance --
        # NULL where its artifacts recorded none. Not the database's part: that
        # is a fact about the bounds, and substituting it here would give every
        # run a part whether or not anything says so.
        "part": s.get("part"),
        "kernel": dict(kernel) if kernel else None,
        "own_variant": own_variant,
        "variants": variants,
        "reference": p.get("reference"),
        "trajectory": traj,
        "effort": {k: v for k, v in dict(effort).items()
                   if k not in ("submission_id", "problem_key")} if effort else None,
        "transcript": ({"bytes": tr["bytes"], "n_lines": tr["n_lines"],
                        "n_turns": tr["n_turns"],
                        "tools": json.loads(tr["tools_json"] or "{}"),
                        "url": f"/api/v1/submissions/{slug}/problems/{key}/transcript"}
                       if tr else None),
        "depth_note": s.get("depth_note"),
    }


def run_window(conn, sub_id: int, key: str) -> dict | None:
    """When this run worked on this problem, and by whose clock.

    `source` travels with the numbers and is never dropped: `first_last_eval`
    is a window *inside* the session (the agent was working before its first
    eval and after its last), `retime_only` is when the kernel was scored on
    GPU 0, not when it was worked on. Rendering either as "the session" is the
    kind of wrong that looks right.

    No row means no timestamp evidence, and that is the answer -- there is no
    `unknown` source to fall back to, because absence already says it.
    `elapsed_seconds` is likewise None unless both ends are real; a one-ended
    window has no duration, and computing one from "now" would invent it.
    """
    row = conn.execute(
        """SELECT started_utc, finished_utc, source FROM run_window
            WHERE submission_id=? AND problem_key=?""", (sub_id, key)).fetchone()
    if row is None:
        return None
    w = dict(row)
    a, b = _parse_utc(w["started_utc"]), _parse_utc(w["finished_utc"])
    w["elapsed_seconds"] = (b - a).total_seconds() if a and b else None
    return w


def _parse_utc(s: str | None):
    """ISO-8601 as `provenance.py` writes it. `Z` because JSON in the wild has
    it and `fromisoformat` did not accept it before 3.11."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _label_times(traj: list[dict]) -> None:
    """Pre-render the two nullable clock fields, so absence cannot render as data.

    `minutes_in` and `utc` are both nullable — the harness-eval series is what
    dates an eval, and an eval it did not date has neither. Handed to a
    template raw, the first becomes an invented "+0 min" the moment anyone
    writes `minutes_in or 0`, and the second interpolates as the literal string
    "None". Both read as measurements. So the API emits the rendered form and
    it is None, not a placeholder, when there is nothing to render: a caller
    substituting a dash is making a display choice, a caller printing "+0m" is
    making a claim.
    """
    for e in traj:
        m = e.get("minutes_in")
        e["at_label"] = f"+{m:.0f}m" if m is not None else None
        u = e.get("utc")
        e["utc_label"] = f"{u[11:16]} UTC" if u and len(u) >= 16 else None


def _mark_regressions(traj: list[dict]) -> None:
    """Classify each eval against the best seen so far in the same problem.

    `regression` is reserved for **correctness**: an eval that passes fewer
    workloads than an earlier one broke something that worked, and that needs
    no threshold to assert.

    Score movement deliberately gets no boolean. The first version flagged any
    eval below the running best, which marked ten of L1__030's fifteen evals as
    regressions -- including 0.5652 against a best of 0.5665, a 0.2% difference
    on a plateau. Calling that a regression is a claim about measurement noise,
    and this repo has no figure for the noise of an agent-side eval (STATE.md
    D20 records that the matmul timing spread on this part is bimodal and that
    no defensible constant could be derived). Inventing a threshold to make the
    flag look reasonable would be inventing a measurement. So `delta_vs_best`
    is reported as a number and the reader judges it.

    An eval where the harness itself did not run is neither: `workloads == 0`
    means nothing was measured, so it cannot have regressed. It is called out
    as a harness error instead, which is what it is.
    """
    best_score = None
    best_passed = 0
    for e in traj:
        measured = bool(e.get("ok")) and (e.get("workloads") or 0) > 0
        e["harness_error"] = not measured
        e["regression"] = bool(
            measured and e.get("passed") is not None and e["passed"] < best_passed)
        e["delta_vs_best"] = (
            (e["mean_score"] - best_score)
            if (measured and e.get("mean_score") is not None and best_score is not None)
            else None)
        if not measured:
            continue
        if e.get("mean_score") is not None:
            best_score = (e["mean_score"] if best_score is None
                          else max(best_score, e["mean_score"]))
        if e.get("passed") is not None:
            best_passed = max(best_passed, e["passed"])


# --------------------------------------------------------------------------
# JSON API — /api/v1 is the contract; bare /api/* is kept as a legacy alias
#
# Every v1 route declares a response_model, so `/openapi.json` describes the
# shapes and a client can be generated from it. The unversioned routes below
# predate that and are left in place only because things already call them;
# they return the same objects with no schema attached.
#
# `part` is a query parameter on every data route rather than a header or a
# path segment, so that a URL a reader copies out of the address bar carries
# the dataset it was read from. Omitted, it resolves exactly as the pages do.
# --------------------------------------------------------------------------

V1 = APIRouter(prefix="/api/v1", tags=["v1"])


@V1.get("/parts", response_model=list[PartInfo])
def v1_parts(request: Request, part: str | None = None):
    """Every part this port targets, and whether anything has been measured.

    `?part=` goes through the same resolver as every other route. Ignoring it
    here made this the one endpoint that answered 200 for a part the build does
    not know — where the rest answer 400 — and marked the *resolved* part
    active rather than the requested one, so the switch's own data was the only
    thing on the site that disagreed with the URL it was asked for.
    """
    return part_infos(request, resolve_part(request, part))


@V1.get("/stats", response_model=Stats)
def v1_stats(request: Request, part: str | None = None):
    return api_stats(request, part)


@V1.get("/leaderboard", response_model=list[LeaderboardRow])
def v1_leaderboard(request: Request, category: str | None = None,
                   part: str | None = None):
    """Ordered by `rank` — the full-benchmark scope, as it always was.

    The HTML board now defaults to the attempted scope and `leaderboard_rows()`
    returns that order, so the page is correct with JavaScript off. The API
    keeps its original ordering rather than following the page: a consumer that
    reads position N out of this list must not silently start getting a
    different row because a UI default changed. `rank_attempted` is on every
    row for anyone who wants the other one.
    """
    with db(resolve_part(request, part)) as conn:
        check_category(conn, category)
        return sorted(leaderboard_rows(conn, category), key=lambda r: r["rank"])


@V1.get("/provisional", response_model=list[ProvisionalRow])
def v1_provisional(request: Request, category: str | None = None,
                   part: str | None = None):
    """Unranked local-validation evidence, separate from leaderboard scores."""
    with db(resolve_part(request, part)) as conn:
        check_category(conn, category)
        return provisional_rows(conn, category)


@V1.get("/problems", response_model=list[ProblemSummary])
def v1_problems(request: Request, category: str | None = None,
                part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        check_category(conn, category)
        return problem_rows(conn, category)


@V1.get("/problems/{key}", response_model=ProblemDetail)
def v1_problem(request: Request, key: str, part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        return problem_detail(conn, key)


@V1.get("/submissions/{slug}", response_model=SubmissionDetail)
def v1_submission(request: Request, slug: str, part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        return submission_detail(conn, slug)


@V1.get("/submissions/{slug}/problems/{key}", response_model=RunDetail)
def v1_run(request: Request, slug: str, key: str, part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        return run_detail(conn, slug, key)


@V1.get("/submissions/{slug}/problems/{key}/kernel",
        response_class=PlainTextResponse)
def v1_kernel(request: Request, slug: str, key: str, part: str | None = None):
    """The submitted kernel as source, for diffing without unwrapping JSON."""
    with db(resolve_part(request, part)) as conn:
        row = conn.execute(
            """SELECT k.source FROM run_kernel k
                 JOIN submission s ON s.id = k.submission_id
                WHERE s.slug=? AND k.problem_key=?""", (slug, key)).fetchone()
    if row is None:
        raise HTTPException(404, f"no kernel recorded for {slug} on {key}")
    return PlainTextResponse(row["source"], media_type="text/x-python")


@V1.get("/submissions/{slug}/problems/{key}/transcript")
def v1_transcript(request: Request, slug: str, key: str, part: str | None = None):
    """Stream the agent transcript from disk.

    Transcripts are 2 MB of JSONL each and are deliberately NOT in the
    database. The path is looked up in `transcript`, never taken from the
    request, so a caller cannot name a file the ingest did not index -- the
    slug and key are only ever used as lookup keys, and the resolved path is
    re-checked against the indexed value before the file is opened.
    """
    with db(resolve_part(request, part)) as conn:
        row = conn.execute(
            """SELECT t.path FROM transcript t
                 JOIN submission s ON s.id = t.submission_id
                WHERE s.slug=? AND t.problem_key=?""", (slug, key)).fetchone()
    if row is None:
        raise HTTPException(404, f"no transcript recorded for {slug} on {key}")
    path = Path(row["path"])
    if not path.is_file():
        raise HTTPException(
            410, f"transcript was indexed but is no longer on disk: {path.name}")
    return FileResponse(path, media_type="application/x-ndjson",
                        filename=f"{slug}__{key}.jsonl")


@app.get("/api/stats")
def api_stats(request: Request, part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        m = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM meta")}
        by_cat = rows(conn, """SELECT category, COUNT(*) AS n,
                                      SUM(CASE WHEN deferred=1 THEN 1 ELSE 0 END) AS deferred
                                 FROM problem GROUP BY category ORDER BY category""")
    return {"meta": m, "by_category": by_cat}


@app.get("/api/leaderboard")
def api_leaderboard(request: Request, category: str | None = None,
                    part: str | None = None):
    # Same ordering as /api/v1/leaderboard, for the same reason.
    with db(resolve_part(request, part)) as conn:
        check_category(conn, category)
        return sorted(leaderboard_rows(conn, category), key=lambda r: r["rank"])


@app.get("/api/provisional")
def api_provisional(request: Request, category: str | None = None,
                    part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        check_category(conn, category)
        return provisional_rows(conn, category)


@app.get("/api/problems")
def api_problems(request: Request, category: str | None = None,
                 part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        check_category(conn, category)
        return problem_rows(conn, category)


@app.get("/api/problems/{key}")
def api_problem(request: Request, key: str, part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        return problem_detail(conn, key)


@app.get("/api/submissions/{slug}")
def api_submission(request: Request, slug: str, part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        return submission_detail(conn, slug)


@app.get("/api/submissions/{slug}/problems/{key}")
def api_run(request: Request, slug: str, key: str, part: str | None = None):
    with db(resolve_part(request, part)) as conn:
        return run_detail(conn, slug, key)


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

# The section nav on the two long reference pages. Declared here rather than
# scraped out of the template because the nav has to be in the server's HTML --
# a reader with JS off still gets it, and a test can check every entry against
# the ids actually rendered (tests/leaderboard/test_sidenav.py). Order is
# reading order, which is also the order the headings appear in; the spy in
# base.html relies on that and on nothing else.
TOC_METHODOLOGY = [
    {"id": "score", "label": "The score"},
    {"id": "bounds", "label": "Where the bounds came from"},
    {"id": "measured", "label": "What is measured"},
    {"id": "timing", "label": "How a submission is timed"},
    {"id": "coverage", "label": "Coverage"},
    {"id": "deferred", "label": "Deferred problems"},
    {"id": "not-covered", "label": "What it does not cover"},
    {"id": "bad-bounds", "label": "Known-wrong bounds"},
    {"id": "headroom", "label": "How much room a bound leaves"},
    {"id": "reading", "label": "Reading it honestly"},
]
# The run page is the deepest page on the site and the longest -- kernel source,
# a trajectory chart, a cost breakdown and a transcript, in that order. Until
# 2026-08-10 it was the only long page with no section nav, so the sections
# below the fold were reachable only by scrolling past a full kernel listing,
# and readers were not finding them.
#
# Five of the six sections always render, including when the run recorded
# nothing for them: an empty section that says "no trajectory was recorded" is
# the point, not an omission, and a reader looking for that answer should be
# able to jump to it. Only the transcript is genuinely absent when missing, so
# only it is filtered -- an anchor to a section that is not there is worse than
# no anchor, and an h2 with no nav entry is a section nobody can reach.
TOC_RUN_ALL = [
    {"id": "workloads", "label": "Per-workload"},
    {"id": "solution", "label": "The solution it proposed"},
    {"id": "trajectory", "label": "How it got there"},
    {"id": "cost", "label": "What it cost"},
    {"id": "transcript", "label": "Transcript"},
    {"id": "others", "label": "Everyone else here"},
]
# Reading order, which is now also the order a reader asks the questions in:
# what the kernel does, the code that defines it, the workloads and their two
# bounds, who has run it, and then the evidence. Inputs, outputs and axes used
# to be three sections BELOW the eleven-column bounds table; they are one
# section above it now, which is why they are no longer three entries here.
TOC_PROBLEM = [
    {"id": "what", "label": "What it computes"},
    {"id": "workloads", "label": "Workloads & bounds"},
    {"id": "reference", "label": "Reference implementation"},
    {"id": "submissions", "label": "Submissions"},
    {"id": "results", "label": "Per-workload results"},
]


def page(request: Request, name: str, part: str, **ctx) -> HTMLResponse:
    """Render *name* with the per-request context every template can rely on.

    `part` is threaded through explicitly rather than resolved here, so a
    handler cannot query one part's database and then render another part's
    header over it.

    `part_mismatch` is filled in here, not by each handler, because the banner
    that consumes it is in `base.html` -- so a handler that forgets it does not
    get a page without a banner, it gets a page whose banner CANNOT fire, and
    Jinja resolves the missing name to undefined rather than raising. That was
    the state: only `/` passed it, so a reader landing straight on a problem
    page, a submission page or the methodology saw nothing, while `/healthz`
    reported the same database as not ok. A handler may still pass its own --
    `index()` does, off the connection it already holds -- and `setdefault`
    leaves it alone.
    """
    ctx.setdefault("part_mismatch", part_mismatches_for(part))
    return templates.TemplateResponse(request, name, {
        "meta": meta(part), "part": part, "parts": part_infos(request, part),
        **ctx})


@app.get("/", response_class=HTMLResponse)
def index(request: Request, category: str | None = None, part: str | None = None):
    active = resolve_part(request, part)
    with db(active) as conn:
        check_category(conn, category)
        board = leaderboard_rows(conn, category)
        provisional = provisional_rows(conn, category)
        # What the rows are divided by, in the scope the chips have selected.
        # Passed to the template so every label on the page quotes the same
        # denominator the table used -- see `scoreable_totals()`.
        scope = scoreable_totals(conn, category)
        mismatch = part_mismatches(conn)
        cats = rows(conn, """SELECT category, COUNT(*) AS n,
                                    SUM(CASE WHEN deferred=1 THEN 1 ELSE 0 END) AS deferred
                               FROM problem GROUP BY category ORDER BY category""")
    return page(request, "index.html", active, board=board, categories=cats,
                category=category, scope=scope, provisional=provisional,
                nav="board",
                part_mismatch=mismatch)


@app.get("/problems", response_class=HTMLResponse)
def problems(request: Request, category: str | None = None, q: str | None = None,
             part: str | None = None):
    active = resolve_part(request, part)
    with db(active) as conn:
        check_category(conn, category)
        items = problem_rows(conn, category)
        cats = rows(conn, "SELECT DISTINCT category FROM problem ORDER BY category")
    if q:
        needle = q.lower()
        items = [p for p in items
                 if needle in p["key"].lower()
                 or needle in (p["description"] or "").lower()]
    return page(request, "problems.html", active, problems=items,
                categories=[c["category"] for c in cats], category=category,
                q=q or "", nav="problems")


@app.get("/problems/{key}", response_class=HTMLResponse)
def problem(request: Request, key: str, part: str | None = None):
    active = resolve_part(request, part)
    with db(active) as conn:
        d = problem_detail(conn, key)
    # On the summary of a collapsed section, so a reader can see how much is
    # behind it before deciding to open it.
    n_results = sum(len(w["results"]) for w in d["workloads"])
    return page(request, "problem.html", active, toc=TOC_PROBLEM,
                n_results=n_results, **d)


@app.get("/submissions/{slug}", response_class=HTMLResponse)
def submission(request: Request, slug: str, part: str | None = None):
    active = resolve_part(request, part)
    with db(active) as conn:
        d = submission_detail(conn, slug)
    return page(request, "submission.html", active, **d)


def trajectory_chart(traj: list[dict], w: int = 720, h: int = 190) -> dict | None:
    """Lay out the trajectory as SVG coordinates.

    Server-side so the page stays dependency-free — no chart library, no build
    step, and the same numbers the API returns are the ones plotted.

    Plotted against S, not against speedup. Speedup is relative to whatever the
    agent's own harness measured as the reference on its own GPU, which is not
    the anchor the leaderboard scores against; S is. That also puts T_b at a
    fixed y = 0.5 gridline, so "crossed the anchor" is visible as a line
    crossing rather than something the reader has to work out.
    """
    # An eval with no recorded time cannot be placed on a time axis, and the
    # previous `minutes_in or 0` placed it at the origin — drawing a point at
    # minute zero for an eval nothing timed, and dragging the polyline back
    # through it. It is dropped from the plot instead and counted, so the page
    # can say how many evals are not shown rather than showing them wrongly.
    timed = [e for e in traj if e.get("minutes_in") is not None]
    n_untimed = len(traj) - len(timed)
    pts = [e for e in timed if e.get("mean_score") is not None]
    if len(pts) < 2:
        return None
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 26
    xs = [e["minutes_in"] for e in pts]
    x_max = max(xs) or 1.0
    ys = [e["mean_score"] for e in pts]
    # Always include the 0.5 anchor in the visible range, or a run that never
    # reaches it renders as a confident climb to nowhere in particular.
    y_lo, y_hi = min(min(ys), 0.5), max(max(ys), 0.5)
    span = (y_hi - y_lo) or 0.1
    y_lo, y_hi = y_lo - span * 0.15, y_hi + span * 0.15

    def px(m):
        return pad_l + (m / x_max) * (w - pad_l - pad_r)

    def py(s):
        return pad_t + (1 - (s - y_lo) / (y_hi - y_lo)) * (h - pad_t - pad_b)

    points = [{"x": round(px(e["minutes_in"]), 1),
               "y": round(py(e["mean_score"]), 1), **e} for e in pts]
    # Evals with no score still happened, and hiding them would make a run that
    # broke twice look monotonic. Drawn on the axis as marks, not as points.
    marks = [{"x": round(px(e["minutes_in"]), 1),
              "n": e["n"], "harness_error": e.get("harness_error"),
              "regression": e.get("regression")}
             for e in timed if e.get("mean_score") is None]
    return {"w": w, "h": h, "points": points, "marks": marks,
            "n_untimed": n_untimed,
            "path": " ".join(f"{'M' if i == 0 else 'L'}{p['x']},{p['y']}"
                             for i, p in enumerate(points)),
            "y_anchor": round(py(0.5), 1) if y_lo <= 0.5 <= y_hi else None,
            "x_axis": round(h - pad_b, 1), "pad_l": pad_l,
            # The axis label is FORMATTED HERE, not in the template: it is the
            # same wall-clock span as every `mins` cell on the page and has to
            # read the same way (`1 h 30 min`, `<1 min`), and an SVG <text>
            # cannot take the `_q` markup that `mins` returns. `x_max` is in
            # minutes; `mins_text` takes seconds.
            "x_max_label": mins_text(x_max * 60),
            "y_lo": round(y_lo, 3), "y_hi": round(y_hi, 3)}


@app.get("/submissions/{slug}/problems/{key}", response_class=HTMLResponse)
def run(request: Request, slug: str, key: str, part: str | None = None):
    active = resolve_part(request, part)
    with db(active) as conn:
        d = run_detail(conn, slug, key)
    # `part` means two different things here and they must not collide: in a
    # RunDetail it is the part this run was MEASURED on, in a page context it
    # is the dataset being viewed, which is what `part_url()` reads off the
    # context to build links. The run's own goes under `run_part`.
    d["run_part"] = d.pop("part")
    # Every heading on the run page now renders unconditionally, the transcript
    # one included: "no transcript was recorded" and "this board does not carry
    # the transcripts" are different facts, and a section that disappears
    # states neither. Transcripts are not tracked in git, so the second case is
    # the normal state of any deploy. test_sidenav.py holds both directions.
    return page(request, "run.html", active, toc=TOC_RUN_ALL,
                chart=trajectory_chart(d["trajectory"]), **d)


app.include_router(V1)
app.include_router(submit.router)


@app.get("/methodology", response_class=HTMLResponse)
def methodology(request: Request, part: str | None = None):
    active = resolve_part(request, part)
    with db(active) as conn:
        bounds = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("bound_sources") or "{}")
        excluded = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("excluded_submissions") or "{}")
        deferred = rows(conn, """SELECT key, n_workloads, deferred_reason,
                                        deferred_mechanism, deferred_error
                                   FROM problem WHERE deferred=1 ORDER BY key""")
        # Facts about each invalid-bound problem rather than a single asserted
        # cause: at least two distinct mechanisms are in play (D18, D21), so a
        # per-row explanation copied from the first one found would be wrong
        # for the others.
        bad_keys = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("problems_with_invalid_bound") or "[]")
        invalid_bound_info = {}
        for k in bad_keys:
            row = conn.execute(
                """SELECT n_workloads, median_headroom FROM problem WHERE key=?""",
                (k,)).fetchone()
            if row:
                invalid_bound_info[k] = {"n_workloads": row["n_workloads"],
                                         "headroom": row["median_headroom"]}
        # The other tail. `problems_with_invalid_bound` is the set where a
        # bound was too TIGHT to be true; these are the ones so loose that the
        # score has almost no roofline content left in it. Nothing in the
        # pipeline flags them, because a weak lower bound violates no rule --
        # which is exactly why they belong on the page that explains what the
        # number means.
        headroom = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("headroom_bands") or "null")
        # Flagged, across every recorded result including runs kept off the
        # ranking. Counted here rather than asserted in the template: the
        # paragraph that renders it used to say "0" as prose, and on 2026-08-10
        # that stopped being true in the same hour a guard was added. A claim
        # that is a negative has to be able to stop being one by itself.
        flagged_row = conn.execute(
            "SELECT COALESCE(SUM(flagged),0) AS n, COUNT(*) AS total FROM result"
        ).fetchone()
        flagged = {"n": flagged_row["n"], "total": flagged_row["total"]}
        flagged_by = [dict(r) for r in conn.execute(
            """SELECT p.key, COUNT(*) AS n FROM result r
               JOIN problem p ON p.key = r.problem_key
               WHERE r.flagged = 1 GROUP BY p.key ORDER BY n DESC""")]
        loosest = [dict(r) for r in conn.execute(
            """SELECT key, n_workloads, median_headroom FROM problem
               WHERE median_headroom IS NOT NULL
               ORDER BY median_headroom DESC LIMIT 8""")]
    # The known-wrong-bounds section renders only when there ARE some, so its
    # nav entry has to be conditional too -- the same rule the run page's
    # transcript entry follows. Latent until now for the wrong reason: the
    # template guarded on `meta.problems_with_invalid_bound`, which is the JSON
    # STRING "[]" when the list is empty, and a non-empty string is truthy. Two
    # bugs cancelling: an always-true guard under an always-present nav entry.
    toc = [e for e in TOC_METHODOLOGY
           if e["id"] != "bad-bounds" or invalid_bound_info]
    return page(request, "methodology.html", active, bound_sources=bounds,
                deferred=deferred, excluded=excluded, nav="methodology",
                toc=toc, invalid_bound_info=invalid_bound_info,
                headroom=headroom, loosest=loosest,
                flagged=flagged, flagged_by=flagged_by)


@app.get("/healthz", response_model=Health)
def healthz(request: Request, part: str | None = None):
    active = resolve_part(request, part)
    path = part_databases().get(active)
    if path is None:
        return JSONResponse({"ok": False, "db": None, "part": active,
                             "error": f"no database for {active}"},
                            status_code=503)
    m = meta(active)
    # The part guard reported where a monitor will see it. `leaderboard_rows()`
    # drops a mismatched submission from the ranking, which is the safe
    # behaviour and also the invisible one; a non-empty list here means the
    # ingest guard was bypassed and the database needs rebuilding, not that the
    # board needs reading more carefully.
    with db(active) as conn:
        mismatch = part_mismatches(conn)
    return JSONResponse({"ok": not mismatch, "db": str(path), "part": m.get("part"),
                         "manifest_version": m.get("manifest_version"),
                         "freshness": m["freshness"],
                         "part_mismatch": mismatch})
