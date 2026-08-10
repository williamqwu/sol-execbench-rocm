# Leaderboard v2 — the contract

This file is the **interface contract** between four pieces of work that are
being done in parallel: the schema + ingest, the app queries + response models,
the templates + static assets, and the backend part plumbing. Nothing here is
optional and nothing here may be renamed without changing this file first.

Everything below obeys the repo's prime directives. In particular:

* **No number appears that was not measured.** Where a harness did not record
  something, the UI says which harness and what it did not record. It never
  substitutes a plausible value, and it never renders "not recorded" the same
  way it renders "zero".
* **MI355X has no measurements.** Nothing in this work produces an MI355X
  number. It produces the *switch*, and the switch lands on an honest empty
  state.

---

## 0. Amendments — 2026-08-06, after the build

Written before the work; the sections below are the contract as agreed. Where
the build learned something the contract did not know, the amendment is here
and the section is corrected in place. Every claim in this block was re-checked
against the running code and the database on 2026-08-06.

* **§3's "five states" is seven.** The grid in `run.html` distinguishes seven,
  and the legend names all seven.
* **§3's hover card cannot show axes** — not because it does not try, but
  because the manifest carries no per-workload axes. `data-axes` renders empty.
  The values exist in the dataset; see §3 for where and what the fix is.
* **§6's claim that `CLOCK_LOCK_PRESETS` has no MI350X entry was wrong when
  written.** The entry has existed since commit 2cdb7b0 (2026-08-03 20:36 UTC)
  and `verify_artifacts.py --task 01` passes: *11 checks, 0 failed, 0 require
  human judgement*, re-run 2026-08-06. Corrected in §6.
* **§6's storage contract cannot produce an MI355X database as written.**
  `ingest.py` has no way to be pointed at a second manifest. Corrected in §6.
* **§7 understates the models.** `PartInfo`'s exact item shape and the
  `RunDetail.part` / page-`part` collision are now written down, because three
  files code against them and a rename would break all three silently.

---

## 1. Trials — the same setup, run more than once

### The real instance

`pilot8` and `opus5-budget100` are the same setup — Claude-Opus-5, the same
agent harness, the same sandbox — under **different budget constraints**
($8/problem vs $100/problem). They overlap on four problems:

    L1__030_attention_output_projection_with_residual
    L1__046_attention_softmax_with_softcapping_and_dropout
    L2__050_vae_decoder_mid_block_attention_resnet
    Quant__004_fp8_moe_expert_linear

All four of `opus5-budget100`'s problems are also in `pilot8`. That is exactly
the "same setup, different constraint" case, and it is why the trial switcher
has real data on day one rather than being a feature with nothing behind it.

### The model: a trial is a submission, a group is the setup

Do **not** add a trial dimension to `result`. A trial is a whole run — its own
kernels, its own trajectory, its own cost — and that is precisely what a
`submission` row already is. What is missing is the link between them.

```sql
-- submission gains:
group_slug       TEXT,     -- the setup: 'agent-opus5'. NULL = ungrouped, a group of one.
group_name       TEXT,     -- 'Claude-Opus-5 · agent harness'
trial_label      TEXT,     -- the constraint that distinguishes it: '$8 / problem'
trial_n          INTEGER,  -- 1-based within the group, ordered by created_utc
constraint_json  TEXT,     -- {"budget_usd_per_session": 8.0, "timeout_s": null, ...}
board_visible    INTEGER NOT NULL DEFAULT 1,
exclusion_reason TEXT,
part             TEXT      -- the part this run was measured on
```

`board_visible = 0` is how `pilot8` comes back. It is **ingested** — its
kernels, trajectory, cost and results all land — but `leaderboard_rows()`
filters it out, so no ranking, count or total on the board changes. Today
`pilot8` is not ingested at all, which means the board cannot show it as a
trial and a reader who follows `docs/agent-baseline.md` to it finds nothing.
Excluding a run from the ranking and deleting its evidence are different
decisions; only the first one was made.

**Verification requirement:** after this change the board's six visible rows,
their ranks, and every aggregate must be **byte-identical** to before. If a
number moves, `board_visible` is being read in the wrong place.

### Grouping rule (in `ingest.py`, explicit, not inferred)

A hardcoded table, because inferring "same setup" from a model name is a guess:

```python
TRIAL_GROUPS = {
    "pilot8":            ("agent-opus5", "Claude-Opus-5 · agent sandbox harness"),
    "opus5-budget100":   ("agent-opus5", "Claude-Opus-5 · agent sandbox harness"),
}
```

`trial_label` comes from the run's own `budget_usd_per_session`, so it is read
from the artifact, not typed in. `trial_n` is assigned by `created_utc` order
within the group.

### What the UI does

On `/submissions/<slug>/problems/<key>`, when the submission's group has more
than one trial: a **trial switcher** listing every trial in the group, each
annotated with its constraint and its outcome on *this* problem. A trial that
never touched this problem is shown disabled with "not in this trial", not
hidden — the reader needs to know the trial exists and did not cover it.

The switcher navigates (real links to the other trial's run page). It does not
merge trials into one view: two runs under different budgets are two
measurements, and averaging them would invent a third.

---

## 2. Timestamps — started and finished, in the viewer's timezone

### Where they come from

There is no field in any artifact that says "this session started at T". What
exists is the harness-eval series, whose files are named
`eval-<epoch_nanoseconds>.json`. So:

```sql
CREATE TABLE run_window (
    submission_id INTEGER NOT NULL REFERENCES submission(id),
    problem_key   TEXT NOT NULL,
    started_utc   TEXT,
    finished_utc  TEXT,
    source        TEXT NOT NULL,   -- see below
    PRIMARY KEY (submission_id, problem_key));
```

`source` is one of, and the UI **must** print which:

| source | meaning |
|---|---|
| `first_last_eval` | first and last harness evaluation the agent ran. This is a window *inside* the session, not the session: the agent was working before its first eval and after its last. |
| `session` | the harness recorded a real session start/end. Nothing does today. |
| `retime_only` | the only timestamp is the authoritative re-time on GPU 0 — when it was *scored*, not when it was worked on. |

`opus5-budget100` and `pilot8` get `first_last_eval`. `glm-run1` gets whatever
`retimed/*.json` provenance carries, as `retime_only`, or **no row at all** if
it carries nothing. No row is correct; a fabricated window is not.

### Rendering in local time

The server does not know the viewer's timezone and must not guess. Emit UTC in
the DOM and convert in the browser:

```html
<time class="localtime" datetime="2026-08-04T19:46:36.951160+00:00">2026-08-04 19:46 UTC</time>
```

A small script in `base.html` rewrites `.localtime` with
`toLocaleString(undefined, {...})` and puts the UTC value in `title`. With JS
off the page still shows a correct, labelled UTC timestamp — never an
unlabelled one, which would silently read as local.

Show, on the run page: **started**, **finished**, **elapsed** (finished −
started), and the `source` caveat in the same breath. Where `run_effort.
wall_seconds` also exists and disagrees with the window, show both and say why
they differ — the window is eval-to-eval, wall time is the whole session.

---

## 3. Per-workload as a grid

The per-workload table is a list of test cases for one problem — 16 for most,
43 for `FlashInfer-Bench__007`. A table of 43 rows buries the shape of the
result. Default to a **heatmap grid**, GitHub-contribution style, and keep the
table one click away.

* One cell per workload, in `workload.rowid` order (the manifest's order).
* Colour by `S`, on the same scale the `.solbar` already uses:
  `S >= 0.5` (beat the anchor) in the accent ramp, `S < 0.5` in the under ramp,
  five steps each. **Failed** is a distinct hatched/red cell, **not attempted**
  is an empty outlined cell, **not scoreable** is dimmed, **bound invalid** is
  its own marker. A failed workload and an unattempted one must not look alike
  at a glance.
* Each cell is a `<button>`/`<a>` carrying the workload's axes, status, T_SOL,
  T_b, T_k, speedup and S in a hover card.
* A legend that names every state, in the same order as the ramp.
* `<details>`/toggle expands the existing detail table, unchanged. The table
  keeps working with JS off; the grid is the enhancement.

The grid is **rendered server-side** (cells and their colour class), so
"view source" still shows the data and no request is needed to see it.

### Amendment — seven states, not five

The build found two more that the five above collapse:

| class | state |
|---|---|
| `g-b1`..`g-b5` | passed, faster than the T_b anchor (5 steps) |
| `g-u1`..`g-u5` | passed, slower than the T_b anchor (5 steps) |
| `g-fail` | a kernel ran and did not pass |
| `g-unmeasured` | **submitted, never measured** |
| `g-miss` | not attempted |
| `g-nosc` | not scoreable |
| `g-inv` | bound invalid |

`g-unmeasured` is the one this contract missed, and it is D23: a kernel was
submitted and the authoritative re-time never completed
(`run_kernel.retime_ok = 0`), so the problem has no result rows at all. It
scores zero exactly like "not attempted" and it is not the same event — one is
a missing attempt, the other a missing measurement. Rendering 30 such cells as
"not attempted" under a banner saying a kernel was submitted made the page
contradict itself, which is how it was found.

The ramp steps are also **not uniform**. They are cut from the observed
distribution, and `run.html` carries the cut and the counts it was cut from,
with the warning that they have a shelf life. A uniform 0.1-wide cut put 89.1%
of scored cells into two steps.

### Amendment — the hover card carries no axes, and cannot

Each cell does emit `data-axes`, so the markup matches the contract. It is
always empty. Verified against the live database: `axes_json` is the string
`'{}'` for **all 3,957** `workload` rows, with no other distinct value.

The reason is the ingest path, not the template. `ingest.py` builds `workload`
rows from `artifacts/09/manifest-v1.json`, and a manifest workload record has
exactly ten keys — `scoreable`, `sol_bottleneck`, `t_b_ms`, `t_b_variant`,
`t_sol_cycles`, `t_sol_cycles_solar`, `t_sol_cycles_traffic`, `t_sol_ms`,
`t_sol_source`, `tolerance`. No `axes`. So `json.dumps(w.get("axes") or {})`
writes `'{}'` every time.

**The values are not lost, only off the path.** The dataset's
`workload.jsonl` carries them per workload —
`{"batch_size": 4, "seq_len_q": 256, "seq_len_kv": 256}` for the first workload
of `L1__001` — and `ingest.py` already opens that problem's directory for
`definition.json`, whose `axes` block it stores on the `problem` row. But that
block is the axis *schema* (name, `var`/`const`/`expr`, description), not this
workload's values, so the problem row cannot stand in for the missing cell
data.

Two ways to close it, and they are not equivalent:

1. Read `workload.jsonl` alongside `definition.json` in the same loop and key
   the axes by `uuid`. Cheapest, but it makes the database depend on `data/`
   being materialised — which is gitignored and does not travel with the repo.
2. Add `axes` to the manifest workload record and regenerate. Then the
   manifest is self-describing and ingest keeps its single input. This changes
   a frozen artifact, so it is a v1.1 item, not a UI item.

Until one of them happens the hover card should not imply the axes are
unknown — they are known and simply not carried. Do not fill the attribute
with a placeholder.

---

## 4. Code panes — highlighting and copy

* Every `<pre class="code">` on the run page gains `data-lang="python"` (or
  `cpp`/`hip`/`text`) and a **copy button** that writes the *raw* source to the
  clipboard — the raw text, not the highlighted DOM with its inserted spans.
  Keep the raw text in a sibling `<script type="text/plain">` or read
  `textContent` before highlighting.
* Highlighting is a **self-contained** tokenizer in
  `leaderboard/static/highlight.js`. No CDN, no build step — the same
  constraint the rest of this app is under. Python and C-family are enough;
  anything else renders unhighlighted rather than wrongly.
* The tokenizer must be **correctness-first**: it may under-highlight, it may
  never corrupt the source. Escape before inserting, and never re-highlight an
  already-highlighted block.
* Line numbers on the left, not selectable (`user-select:none`) so copying a
  region by mouse does not drag the numbers in.

## 5. The trajectory chart becomes interactive

`trajectory_chart()` already computes the coordinates server-side. Keep that —
it stays dependency-free and the plotted numbers stay the API's numbers. Add:

* Each `<circle>` gets `data-eval="{{ p.n }}"`; each trajectory table `<tr>`
  gets `data-eval="{{ e.n }}"`.
* Hovering or focusing a point shows a **tooltip** (positioned HTML, not the
  SVG `<title>` tooltip, which is slow and unstyleable) with eval number,
  time, S, Δ vs best, passed/workloads, and kernel size.
* Hovering a point highlights the matching table row, and hovering a row
  highlights the matching point. Both directions.
* Keyboard reachable: points are focusable, arrow keys move between them.
* The axis ticks (evals with no score) participate in the same linking.

Retain the existing `<title>` elements as the no-JS fallback.

---

## 6. The part switch — MI350X ↔ MI355X

### The rule this has to encode

A score measured on MI350X and a score measured on MI355X are **not
comparable**: different power cap, different sustained clock, therefore
different F_LOCK, therefore different T_SOL and different T_b. Mixing them in
one table would be the most damaging kind of wrong — every number plausible,
nothing detectably broken.

So the part is not a filter over one dataset. **It selects the dataset.**

### Storage

One database per part:

    leaderboard/db/solbench-MI350X.db
    leaderboard/db/solbench-MI355X.db     <- does not exist, and must not be faked

`ingest.py --part MI350X` writes `db/solbench-MI350X.db` and stamps
`meta.part`. The part defaults to the manifest's own `part` field; passing
`--part` that disagrees with the manifest is an **error**, not an override.

Back-compatibility: if `leaderboard/solbench.db` exists it is still read, and
its part is taken from its own `meta.part`. `SOLBENCH_DB` continues to work and
pins a single database (used by the tests and by `worker.py`).

### Resolution order for the active part

1. `?part=` query parameter (explicit, shareable in a URL)
2. `part` cookie (set whenever 1 is used — the switch is sticky across pages)
3. `SOLBENCH_PART` environment variable
4. the only part that has a database, if there is exactly one
5. `MI350X`

An unknown part name is a 400, not a silent fallback to the default.

### The empty state

Known parts come from `src/solexbench_rocm/parts.py: PARTS` — MI350X, MI355X,
MI300X. A part in that registry with no database is a **first-class page**, not
a 404 and not an empty table: it says that nothing has been measured on that
part, that the port itself needs no work, and it links to the runbook
(`docs/TODO-MI355X.md`). MI300X is CDNA3 and is out of scope for this port — the
switch lists only parts the port targets.

### The switch itself

A segmented control in the header, beside the existing `.part-pill`, which it
replaces. Each option shows the part name and, underneath, either its
measurement count or "not measured". Selecting a part reloads the current page
with `?part=`, preserving the rest of the path and query. Every internal link
must carry the active part forward — a `part_url()` Jinja global, used
everywhere, so a reader who switches to MI355X and clicks a problem does not
silently land back on MI350X data.

The footer's F_LOCK / device / ROCm line is per-part and comes from that part's
own `meta`. It must never show MI350X's F_LOCK under an MI355X heading.

### Backend

`scripts/sol_bounds.py` and `solar/gen_arch_yaml.py` already take `--part`.

**Correction.** This section previously said the missing piece was that
`CLOCK_LOCK_PRESETS` has no MI350X entry, and pointed at TODO.md and the task-01
gate. That was wrong. The entry has been there since **2cdb7b0** (2026-08-03
20:36 UTC) —
`ClockPreset(gpu_clk_mhz=1600, dram_clk_mhz=None, achieved_gpu_clk_mhz=1300)`,
`device_config.py:113` — and `verify_artifacts.py --task 01` passes with
*11 checks, 0 failed*. Across all ten tasks the only failing gate is task 03's
D18/D21 bound check. Nothing in the part switch depends on the preset table.

What *is* in scope: `docs/TODO-MI355X.md`, a runbook precise enough that an agent
landing on an MI355X node can execute it without re-deriving the plan.

### Amendment — the storage contract cannot be executed as written

`ingest.py --part MI355X` cannot produce `db/solbench-MI355X.db`, and not for
want of measurements. The manifest path is a **module constant**:

```python
MANIFEST = ROOT / "artifacts" / "09" / "manifest-v1.json"   # ingest.py:50
```

There is no `--manifest` flag. And `--part` deliberately **asserts** rather than
sets: it is compared against the part the manifest says it was measured on, and
a disagreement is an error, not an override — which is the right design and is
exactly why it cannot be used to relabel v1's MI350X manifest into an MI355X
database. So the two flags between them can produce one database, from one
manifest, for one part.

What is missing, concretely:

1. A `--manifest PATH` argument, defaulting to today's constant so no existing
   invocation changes.
2. Deriving the output database name from the manifest's own part when `--db`
   is not given, so `db/solbench-<part>.db` follows from the input rather than
   from a flag someone has to remember.

Both are small. Neither should be written speculatively against an MI355X
manifest that does not exist: there is no way to test the second path until
`tasks/01` has run on that part, and an untested branch that writes databases
is worse than a documented gap. The empty state above is what ships until then,
and it is honest.

---

## 7. Response models

Every new field is added to `leaderboard/models.py` with the honest type.
`int | None`, not `int = 0`, wherever a LEFT JOIN can produce NULL — that
mistake was caught once already, on 661 of 3343 endpoints, by the response
model doing its job. New models: `Trial`, `RunWindow`, `PartInfo`. `RunDetail`
gains `trials: list[Trial]`, `window: RunWindow | None`, `part: str | None`.

### The `parts[]` item shape — fixed, three consumers

`PartInfo` (`models.py`) is produced by `part_infos()` (`app.py`) and rendered
by `templates/base.html` (the switch) and `templates/part_missing.html` (the
empty state). Renaming a key breaks all three and breaks them quietly, because
Jinja resolves a missing attribute to undefined rather than raising:

```python
{"name": str, "available": bool, "n_results": int | None,
 "active": bool, "url": str}
```

**`n_results` is `None` where there is no database, and never `0`.** "Not
measured" and "measured nothing" are different statements and the switch has to
be able to say which one it means — so both templates test
`{% if p.n_results is not none %}`, not truthiness. A `0` here would render
"0 results" under MI355X and quietly assert a run that never happened. This is
the same rule as the F_LOCK footer: absence is rendered as absence.

`url` is this same page on that part, built by `switch_url()` with path and
query preserved and `part=` set, so the switch never drops the reader
somewhere else.

### `RunDetail.part` and the page-context `part` collide — the template gets `run_part`

Two different things want the name `part` in one template:

* **`RunDetail.part`** — the part this run was **measured** on, from its own
  re-time provenance.
* **the page context's `part`** — the active dataset, i.e. which database is
  being served, used by `part_url()` to build every internal link.

They are usually equal and are not the same claim, and where they differ the
difference is the interesting thing. So the view renames the run's own on the
way into the template (`app.py`, in the run handler):

```python
d["run_part"] = d.pop("part")
```

`run.html` therefore reads **`run_part`** for the measured part and leaves
`part` meaning the page context. It renders `run_part` as a tag, and when
`run_part != meta.part` it shows a banner: the bounds on this page came from a
different part's manifest than the run was measured on. The API is unchanged —
`/api/v1/submissions/{slug}/problems/{key}` still returns `part`; the rename is
template-side only. Do not "tidy" it by renaming the context variable instead:
`part_url()` and every link helper are keyed on `part`.

## 8. Tests

`tests/leaderboard/test_service.py` gains coverage for, at minimum:

* the board is unchanged by `board_visible` — same rows, ranks and aggregates
* a hidden submission is reachable by URL and by the trial switcher
* trials group correctly and `trial_n` is stable
* `run_window.source` is never invented: a submission with no timestamp
  evidence has no row
* the part resolver: query > cookie > env > sole > default; unknown = 400
* a part with no database renders the empty state, not a 500 and not a 404
* every `/api/v1` route still declares a response schema
* the local-time markup carries a parseable ISO-8601 UTC `datetime`
