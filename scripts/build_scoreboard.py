#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 10 — build the scoreboard and its dashboard.

    python scripts/build_scoreboard.py --run-id pilot-01
    python scripts/build_scoreboard.py --all-runs

Reads ``artifacts/10/scores/<run-id>/`` and writes:

    artifacts/10/scoreboard.json    the aggregate, machine-readable
    artifacts/10/dashboard.html     the same thing, self-contained, no network

The HTML embeds its data and uses no CDN, because the most likely reader opens it
over SSH port-forwarding or copies it off the node, and a dashboard that needs
the internet to render is a dashboard that renders blank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402
from solexbench_agents.aggregate import summarize_results  # noqa: E402

SCORES_ROOT = ROOT / "artifacts" / "10" / "scores"


def load_run(run_id: str) -> tuple[list[dict], dict]:
    run_dir = SCORES_ROOT / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no scores for run {run_id!r} at {run_dir}")
    results = []
    for path in sorted(run_dir.glob("*/*.json")):
        if path.name == "summary.json":
            continue
        results.append(json.loads(path.read_text()))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return results, summary


def build(run_ids: list[str]) -> dict:
    runs = {}
    for run_id in run_ids:
        results, scoring_summary = load_run(run_id)
        agg = summarize_results(results)
        runs[run_id] = {
            "scoring": {k: v for k, v in scoring_summary.items()
                        if k != "_provenance"},
            **agg,
        }
    return {
        **stamp("10-scoreboard"),
        "runs": runs,
        "run_ids": run_ids,
    }


def render_html(scoreboard: dict) -> str:
    data = json.dumps(scoreboard, indent=None, default=str)
    # </script> inside embedded JSON would end the tag early.
    data = data.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", data)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOL-ExecBench-AMD — agent scoreboard</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --line: #262b36; --ink: #e6e9ef;
    --dim: #98a1b3; --accent: #5b9dff; --good: #3fb950; --warn: #d29922;
    --bad: #f85149; --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 32px 24px 96px; }
  h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 17px; margin: 40px 0 12px; letter-spacing: -0.01em; }
  h3 { font-size: 14px; margin: 24px 0 8px; color: var(--dim);
       text-transform: uppercase; letter-spacing: 0.06em; }
  .sub { color: var(--dim); font-size: 13px; margin: 0 0 24px; }
  .panel { background: var(--panel); border: 1px solid var(--line);
           border-radius: 10px; padding: 18px 20px; margin-bottom: 16px; }
  .note { border-left: 3px solid var(--warn); background: #1d1a12;
          padding: 12px 16px; border-radius: 6px; font-size: 13.5px;
          margin-bottom: 20px; color: #e8d9b0; }
  .note b { color: #ffd479; }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--dim); font-weight: 600; font-size: 11.5px;
       text-transform: uppercase; letter-spacing: 0.05em; }
  tbody tr:hover { background: #1c2029; }
  td.mono, .mono { font-family: var(--mono); font-size: 12.5px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
           gap: 12px; margin-bottom: 8px; }
  .card { background: var(--panel); border: 1px solid var(--line);
          border-radius: 10px; padding: 14px 16px; }
  .card .k { color: var(--dim); font-size: 11.5px; text-transform: uppercase;
             letter-spacing: 0.05em; }
  .card .v { font-size: 26px; font-weight: 650; margin-top: 4px;
             letter-spacing: -0.02em; }
  .card .d { color: var(--dim); font-size: 12px; margin-top: 2px; }
  .bar { height: 7px; background: #21262f; border-radius: 4px; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: var(--accent); }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
          font-size: 11.5px; border: 1px solid var(--line); color: var(--dim); }
  .ok { color: var(--good); } .no { color: var(--bad); } .mid { color: var(--warn); }
  .empty { color: var(--dim); }
  .legend { color: var(--dim); font-size: 12.5px; margin-top: 8px; }
  select { background: var(--panel); color: var(--ink); border: 1px solid var(--line);
           border-radius: 6px; padding: 6px 10px; font-size: 13px; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SOL-ExecBench-AMD — agent scoreboard</h1>
  <p class="sub" id="subtitle"></p>
  <div class="row" style="margin-bottom:20px">
    <label for="runsel" class="mono">run</label>
    <select id="runsel"></select>
  </div>
  <div id="app"></div>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const SB = JSON.parse(document.getElementById('payload').textContent);

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? '<span class="empty">-</span>' : Number(v).toFixed(d);
const pct = (v) => (v === null || v === undefined)
  ? '<span class="empty">-</span>' : Number(v).toFixed(1) + '%';
const usd = (v) => (v === null || v === undefined)
  ? '<span class="empty">n/a</span>' : '$' + Number(v).toFixed(2);
const esc = (s) => String(s ?? '').replace(/[&<>]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function bar(fraction) {
  const w = Math.max(0, Math.min(1, fraction || 0)) * 100;
  return `<div class="bar"><span style="width:${w}%"></span></div>`;
}

function basisNote(run) {
  const bases = (run.scoring && run.scoring.score_bases) || {};
  const hasFull = (bases['sol_score_v1'] || 0) > 0;
  const tb = (run.scoring && run.scoring.t_b) || {};
  const tsol = (run.scoring && run.scoring.t_sol) || {};
  if (hasFull) return '';
  const bits = [];
  if (!tsol.used) bits.push(`<b>T_SOL</b> unavailable for this part${tsol.rejected ? ` (${esc(tsol.rejected)})` : ''}`);
  if (!tb.used) bits.push(`<b>T_b</b> unavailable${tb.rejected ? ` (${esc(tb.rejected)})` : ''}`);
  return `<div class="note">No <b>SOL score</b> column below is populated, and that is
    a statement about the bounds rather than the kernels: ${bits.join('; ')}.
    Until both land, speed is reported as measured speedup over each problem's own
    reference. That is <b>not</b> the SOL score and is not comparable to
    upstream's published figures &mdash; the reference is whatever the dataset
    shipped, while the anchor T_b is an optimized-PyTorch variant chosen by a
    sweep. Every record carries a <span class="mono">score_basis</span> field so
    the two can never be pooled by accident.</div>`;
}

function cards(run) {
  const hs = Object.entries(run.harnesses);
  const totalSolved = hs.reduce((a, [, h]) => a + h.solved, 0);
  const totalAttempt = hs.reduce((a, [, h]) => a + h.attempted, 0);
  const totalCost = hs.reduce((a, [, h]) => a + (h.cost.total_usd || 0), 0);
  const anyUnpriced = hs.some(([, h]) => h.cost.unpriced_sessions > 0);
  const totalMin = hs.reduce((a, [, h]) => a + (h.time.total_min || 0), 0);
  const wl = hs.reduce((a, [, h]) => a + h.workloads, 0);
  const wlp = hs.reduce((a, [, h]) => a + h.workloads_passed, 0);
  return `<div class="cards">
    <div class="card"><div class="k">problems solved</div>
      <div class="v">${totalSolved}<span style="font-size:15px;color:var(--dim)"> / ${totalAttempt}</span></div>
      <div class="d">all workloads passing</div></div>
    <div class="card"><div class="k">workload pass rate</div>
      <div class="v">${pct(wl ? 100 * wlp / wl : null)}</div>
      <div class="d">${wlp} of ${wl} workloads</div></div>
    <div class="card"><div class="k">agent spend</div>
      <div class="v">${usd(totalCost || null)}</div>
      <div class="d">${anyUnpriced ? `${hs.reduce((a, [, h]) => a + h.cost.unpriced_sessions, 0)} of ${totalAttempt} sessions report no price` : 'as reported by the harnesses'}</div></div>
    <div class="card"><div class="k">input tokens</div>
      <div class="v">${(hs.reduce((a, [, h]) => a + (h.tokens ? h.tokens.input : 0), 0) / 1e6).toFixed(1)}<span style="font-size:15px;color:var(--dim)"> M</span></div>
      <div class="d">effort measure that survives a timeout</div></div>
    <div class="card"><div class="k">agent wallclock</div>
      <div class="v">${fmt(totalMin / 60, 1)}<span style="font-size:15px;color:var(--dim)"> h</span></div>
      <div class="d">summed across GPUs</div></div>
  </div>`;
}

function harnessTable(run) {
  const rows = Object.entries(run.harnesses).map(([name, h]) => `
    <tr>
      <td><b>${esc(name)}</b><div class="d mono" style="color:var(--dim);font-size:11.5px">${esc(h.model || 'model not reported')}</div></td>
      <td>${h.solved} / ${h.attempted}</td>
      <td>${pct(h.solve_rate_pct)}${bar((h.solve_rate_pct || 0) / 100)}</td>
      <td>${pct(h.workload_pass_rate_pct)}</td>
      <td>${fmt(h.timing.speedup_vs_reference.median)}x<div class="d" style="color:var(--dim);font-size:11px">n=${h.timing.speedup_vs_reference.n}</div></td>
      <td>${h.timing.sol_score.n ? fmt(h.timing.sol_score.median, 3) : '<span class="empty">-</span>'}</td>
      <td>${usd(h.cost.total_usd)}${h.cost.unpriced_sessions
        ? `<div class="d" style="color:var(--warn);font-size:11px">${h.cost.unpriced_sessions} unpriced</div>` : ''}</td>
      <td>${h.tokens ? (h.tokens.input / 1e6).toFixed(1) + 'M' : '-'}</td>
      <td>${fmt(h.time.mean_min, 1)}</td>
      <td>${fmt(h.verify_attempts.mean, 1)}</td>
    </tr>`).join('');
  return `<h2>By harness</h2><div class="panel"><table>
    <thead><tr><th>harness</th><th>solved</th><th>solve rate</th>
      <th>workload pass</th><th>median speedup</th><th>median S</th>
      <th>spend</th><th>input tok</th><th>mean min</th><th>mean verifies</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <div class="legend">"solved" means every workload of the problem passed on the
      authoritative GPU. Median speedup is over passing workloads only, since the
      latency of a wrong kernel is not a measurement of anything. Spend is only
      what a harness reported: a session killed at the wallclock cap never emits
      its price, so tokens are shown beside it as the effort measure that
      survives.</div>
  </div>`;
}

function categoryTable(run) {
  const cats = ['L1', 'L2', 'Quant', 'FlashInfer-Bench'];
  const names = Object.keys(run.harnesses);
  const head = names.map(n => `<th>${esc(n)}</th>`).join('');
  const rows = cats.map(c => {
    const cells = names.map(n => {
      const bc = run.harnesses[n].by_category[c];
      if (!bc) return '<td><span class="empty">-</span></td>';
      const cls = bc.solve_rate_pct >= 66 ? 'ok' : bc.solve_rate_pct >= 33 ? 'mid' : 'no';
      return `<td><span class="${cls}">${bc.solved}/${bc.attempted}</span>
        <div class="d" style="color:var(--dim);font-size:11px">${pct(bc.workload_pass_rate_pct)} wl</div></td>`;
    }).join('');
    return `<tr><td>${esc(c)}</td>${cells}</tr>`;
  }).join('');
  return `<h2>By category</h2><div class="panel"><table>
    <thead><tr><th>category</th>${head}</tr></thead><tbody>${rows}</tbody></table>
    <div class="legend">Top figure is problems solved; below it the workload pass
      rate, which moves earlier than the solve rate because a problem needs every
      workload to pass.</div></div>`;
}

function failureTable(run) {
  const names = Object.keys(run.harnesses);
  const kinds = new Set();
  names.forEach(n => Object.keys(run.harnesses[n].failures).forEach(k => kinds.add(k)));
  const ordered = [...kinds].sort((a, b) => (a === 'passed' ? -1 : b === 'passed' ? 1 : a.localeCompare(b)));
  const rows = ordered.map(k => {
    const cells = names.map(n => `<td>${run.harnesses[n].failures[k] || 0}</td>`).join('');
    return `<tr><td class="mono">${esc(k)}</td>${cells}</tr>`;
  }).join('');
  return `<h2>Where it went wrong</h2><div class="panel"><table>
    <thead><tr><th>stage</th>${names.map(n => `<th>${esc(n)}</th>`).join('')}</tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="legend">Counted per workload for anything that reached evaluation,
      per problem for anything that did not. <span class="mono">no_solution</span>
      and <span class="mono">invalid_solution</span> are harness or budget
      outcomes; <span class="mono">incorrect_numerical</span> and
      <span class="mono">compile_error</span> are the model's.</div></div>`;
}

function languageTable(run) {
  const names = Object.keys(run.harnesses);
  const langs = new Set();
  names.forEach(n => Object.keys(run.harnesses[n].languages).forEach(l => langs.add(l)));
  if (!langs.size) return '';
  const rows = [...langs].sort().map(l => `<tr><td class="mono">${esc(l)}</td>${
    names.map(n => `<td>${run.harnesses[n].languages[l] || 0}</td>`).join('')}</tr>`).join('');
  const copyRows = names.map(n => {
    const c = run.harnesses[n].reference_copies || {};
    return `<tr><td>${esc(n)}</td><td>${c.exact || 0}</td><td>${c.near || 0}</td><td>${c.distinct || 0}</td></tr>`;
  }).join('');
  return `<h2>What they wrote</h2>
    <div class="panel"><h3>declared language</h3><table>
      <thead><tr><th>language</th>${names.map(n => `<th>${esc(n)}</th>`).join('')}</tr></thead>
      <tbody>${rows}</tbody></table></div>
    <div class="panel"><h3>was it just the reference again?</h3><table>
      <thead><tr><th>harness</th><th>exact copy</th><th>near copy</th><th>distinct</th></tr></thead>
      <tbody>${copyRows}</tbody></table>
      <div class="legend">Resubmitting the reference is correct, so it lifts the
        pass rate without demonstrating any kernel work, and it lands near the
        anchor by construction. Labelled rather than penalized.</div></div>`;
}

function problemTable(run) {
  const names = Object.keys(run.harnesses);
  const rows = run.problems.map(p => {
    const cells = names.map(n => {
      const h = p.harnesses[n];
      if (!h) return '<td><span class="empty">-</span></td>';
      const mark = h.solved
        ? `<span class="ok">${h.passed}/${h.workloads}</span>`
        : `<span class="no">${h.passed}/${h.workloads}</span>`;
      const extra = h.solved && h.median_speedup ? ` <span class="pill">${fmt(h.median_speedup)}x</span>` : '';
      const lang = (h.languages || []).join(',');
      return `<td>${mark}${extra}<div class="d mono" style="color:var(--dim);font-size:11px">${esc(lang || h.outcome)}</div></td>`;
    }).join('');
    const flag = p.solved_by_none ? ' <span class="pill no">unsolved</span>' : '';
    return `<tr><td class="mono">${esc(p.problem)}${flag}</td>${cells}</tr>`;
  }).join('');
  const unsolved = run.problems.filter(p => p.solved_by_none).length;
  return `<h2>Per problem</h2><div class="panel">
    <div class="legend" style="margin-bottom:10px">${unsolved} of ${run.problems.length}
      problems were solved by no harness. Those are the rows worth reading first:
      they are either genuinely hard kernels or a defect in the port, and the two
      look identical from the headline rate.</div>
    <table><thead><tr><th>problem</th>${names.map(n => `<th>${esc(n)}</th>`).join('')}</tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function integrityNote(run) {
  const hi = (run.scoring && run.scoring.harness_integrity) || {};
  if (!hi.comparable) {
    return `<div class="note">The scoring harness was <b>not fingerprinted</b> when
      this sweep started, so it cannot be shown that the code judging correctness
      stayed put while agents were running. ${esc(hi.note || '')}</div>`;
  }
  if (hi.match) return '';
  return `<div class="note">The scoring-critical source tree <b>changed</b> between
    the sweep starting and it being scored: ${(hi.changed || []).map(esc).join(', ') || 'see summary.json'}.
    An operator edit produces the same signal as tampering, so this is reported
    rather than judged &mdash; but these scores are not comparable with ones taken
    across an unchanged harness.</div>`;
}

function violationNote(run) {
  const all = Object.entries(run.harnesses)
    .flatMap(([n, h]) => (h.bound_violations || []).map(v => ({ ...v, harness: n })));
  if (!all.length) return '';
  return `<div class="note">${all.length} workload(s) came in <b>faster than their
    own Speed-of-Light bound</b>, which is impossible and means the bound is too
    loose rather than the kernel being exceptional. Not clamped, because clamping
    would hide it: ${all.slice(0, 4).map(v => `<span class="mono">${esc(v.harness)}/${esc(v.problem)}</span>`).join(', ')}${all.length > 4 ? ', ...' : ''}.</div>`;
}

function render(runId) {
  const run = SB.runs[runId];
  const p = SB._provenance || {};
  document.getElementById('subtitle').innerHTML =
    `${esc(p.part || 'unknown part')} &middot; F_LOCK ${esc(p.f_lock_mhz ?? '?')} MHz &middot; `
    + `torch ${esc((p.torch || {}).version || '?')} &middot; ROCm ${esc((p.rocm || {}).version || '?')} &middot; `
    + `built ${esc((p.utc || '').slice(0, 19))} &middot; <span class="mono">${esc((p.git_sha || '').slice(0, 12))}</span>`;
  document.getElementById('app').innerHTML = [
    basisNote(run), integrityNote(run), violationNote(run),
    cards(run), harnessTable(run), categoryTable(run),
    failureTable(run), languageTable(run), problemTable(run),
  ].join('');
}

const sel = document.getElementById('runsel');
SB.run_ids.forEach(id => {
  const o = document.createElement('option');
  o.value = id; o.textContent = id; sel.appendChild(o);
});
sel.addEventListener('change', () => render(sel.value));
render(SB.run_ids[SB.run_ids.length - 1]);
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", action="append", help="repeatable")
    ap.add_argument("--all-runs", action="store_true")
    ap.add_argument("--out-json",
                    default=str(ROOT / "artifacts" / "10" / "scoreboard.json"),
                    type=Path)
    ap.add_argument("--out-html",
                    default=str(ROOT / "artifacts" / "10" / "dashboard.html"),
                    type=Path)
    args = ap.parse_args()

    if args.all_runs:
        run_ids = sorted(p.name for p in SCORES_ROOT.iterdir() if p.is_dir()) \
            if SCORES_ROOT.is_dir() else []
    else:
        run_ids = args.run_id or []
    if not run_ids:
        sys.exit("nothing to build: pass --run-id or --all-runs")

    scoreboard = build(run_ids)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(scoreboard, indent=2, default=str))
    args.out_html.write_text(render_html(scoreboard))

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_html}")
    for run_id, run in scoreboard["runs"].items():
        print(f"\n{run_id}:")
        for name, h in run["harnesses"].items():
            cost = f"${h['cost']['total_usd']:.2f}" \
                if h["cost"]["total_usd"] is not None else "cost n/a"
            print(f"  {name:<12} solved {h['solved']}/{h['attempted']}  "
                  f"workloads {h['workloads_passed']}/{h['workloads']}  {cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
