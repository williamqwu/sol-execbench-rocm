#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A live status page for the task 10 pipeline. Standard library only.

    python scripts/status_server.py --port 8099
    # then, from your laptop:
    ssh -N -L 8099:localhost:8099 <node>   →   http://localhost:8099

Answers the questions a long unattended run actually raises, and which no log
tail answers well:

  * which stage is running, and which have completed
  * **which GPUs are doing work right now, and who is holding them** — the one
    that matters most here, because stage 1 is serial on the authoritative GPU by
    construction and therefore leaves seven cards idle, and that is very hard to
    see from a log that is scrolling
  * the T_b queue: how many done, and the per-problem durations, which are so
    skewed that a mean is misleading
  * the agent sweep, per harness, once it starts

No dependencies, no build step, and it reads only. It cannot perturb a
measurement, which matters because the thing it is watching is a timing run: the
page shells out to ``rocm-smi`` at most once per refresh interval and never
touches a GPU context.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "artifacts" / "10" / "pipeline"
LOGS = PIPELINE / "logs"

STAGES = [
    ("t_sol", "0 · T_SOL", "CPU only"),
    ("tb_authoritative", "1 · authoritative T_b", "idle node, one GPU"),
    (None, "2 · traffic-floor tier", "CPU, gated on T_b"),
    (None, "3 · freeze manifest", "both tiers + T_b"),
    ("anchor", "4 · anchor verification", "idle node"),
    ("sweep", "5 · agent sweep", "7 GPUs, hours"),
    (None, "6 · scoring", "authoritative GPU, serial"),
    (None, "7 · backfill S", "no GPU"),
    (None, "8 · scoreboard", "no GPU"),
]

_CACHE: dict = {"t": 0.0, "smi": []}


def gpu_map() -> dict[int, int]:
    """{torch index -> rocm-smi index}. Cached: it is resolved from PCI identity
    and does not change while the node is up.

    Without this the page would label the busy card wrongly. On this node torch 0
    is rocm-smi 3, so a naive reading shows "GPU 3 busy" while the pipeline
    reports it is using GPU 0, and both are right.
    """
    if _CACHE.get("map"):
        return _CACHE["map"]
    try:
        out = subprocess.run(["rocm-smi", "--showbus"], capture_output=True,
                             text=True, timeout=30).stdout
        by_bus = {int(m.group(2), 16): int(m.group(1)) for m in
                  re.finditer(r"GPU\[(\d+)\][^\n]*?PCI Bus:\s*[0-9A-Fa-f]{4}:"
                              r"([0-9A-Fa-f]{2}):", out)}
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import torch
        m = {}
        for t in range(torch.cuda.device_count()):
            bus = int(getattr(torch.cuda.get_device_properties(t), "pci_bus_id"))
            if bus in by_bus:
                m[t] = by_bus[bus]
        _CACHE["map"] = m
        return m
    except Exception:
        return {}


def smi(min_interval: float = 3.0) -> list[dict]:
    """Per-GPU utilisation, clock and power, throttled so refreshes are cheap."""
    now = time.time()
    if now - _CACHE["t"] < min_interval and _CACHE["smi"]:
        return _CACHE["smi"]
    rows: dict[int, dict] = {}
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--showpower", "--showgpuclocks"],
            capture_output=True, text=True, timeout=30).stdout
        for m in re.finditer(r"GPU\[(\d+)\][^:]*:\s*GPU use \(%\):\s*(\d+)", out):
            rows.setdefault(int(m.group(1)), {})["use"] = int(m.group(2))
        # `.*?` rather than `[^:]*`: the label itself contains the separating
        # colon ("GPU[0]  : Current Socket Graphics Package Power (W): 241.0"),
        # and the field name varies across rocm-smi builds between "Current
        # Socket Graphics Package" and "Average Graphics Package".
        for m in re.finditer(r"GPU\[(\d+)\].*?Power \(W\):\s*([\d.]+)", out):
            rows.setdefault(int(m.group(1)), {})["power"] = float(m.group(2))
        for m in re.finditer(r"GPU\[(\d+)\][^:]*:\s*sclk clock speed:\s*\((\d+)Mhz\)",
                             out):
            rows.setdefault(int(m.group(1)), {})["sclk"] = int(m.group(2))
        for m in re.finditer(r"GPU\[(\d+)\][^:]*:\s*sclk clock level:[^(]*\((\d+)Mhz\)",
                             out):
            rows.setdefault(int(m.group(1)), {}).setdefault("sclk", int(m.group(2)))
    except Exception as exc:
        _CACHE["smi"] = [{"error": str(exc)}]
        _CACHE["t"] = now
        return _CACHE["smi"]

    holders = gpu_holders()
    inv = {v: k for k, v in gpu_map().items()}
    result = []
    for smi_idx in sorted(rows):
        torch_idx = inv.get(smi_idx)
        result.append({
            "smi": smi_idx,
            "torch": torch_idx,
            **rows[smi_idx],
            "holders": holders.get(torch_idx, []),
        })
    _CACHE["smi"] = result
    _CACHE["t"] = now
    return result


def gpu_holders() -> dict[int, list[str]]:
    """{torch index -> [process labels]} for anything holding a HIP context.

    Read from ``/proc/<pid>/environ`` rather than from a GPU query, because what
    matters is *which job* is on a card, not just that one is. A GPU at 100% with
    no owner named here is an orphan, and an orphan inflates every later
    measurement on that device (STATE.md D22).
    """
    out: dict[int, list[str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fd = entry / "fd"
            if not any(f.resolve().name == "kfd" for f in fd.iterdir()
                       if f.is_symlink()):
                continue
            env = (entry / "environ").read_bytes().decode(errors="replace")
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace").strip()
        except (OSError, PermissionError):
            continue
        m = re.search(r"HIP_VISIBLE_DEVICES=([0-9,]+)", env)
        if not m:
            continue
        visible = [int(x) for x in m.group(1).split(",") if x != ""]
        # A process that can see every GPU is a driver, not a worker; attributing
        # it to all eight would make every card look busy.
        if len(visible) > 1:
            continue
        label = None
        for needle, shown in (
            ("compile_worker", "compile worker"),
            ("eval_driver", "eval_driver"),
            ("authoritative_tb", "authoritative_tb"),
            ("time_tb_candidates", "time_tb_candidates"),
            ("agent_verify", "agent_verify"),
            ("sol_bounds", "sol_bounds"),
            ("claude", "claude"),
            ("codex", "codex"),
        ):
            if needle in cmd:
                label = shown
                break
        if label is None:
            # Last resort: the executable, not the last argument. Inductor's
            # compile workers end their command line in a base64 pickler token,
            # so the last token is a hash and tells a reader nothing.
            first = cmd.split()[0] if cmd.split() else cmd
            label = Path(first).name or "?"
        out.setdefault(visible[0], []).append(label)
    return {gpu: _summarise(labels) for gpu, labels in out.items()}


def _summarise(labels: list[str]) -> list[str]:
    """Collapse a fan-out into counts.

    A single `torch.compile` timing run spawns dozens of inductor compile
    workers; listing each pid buries the one process that actually says what the
    GPU is doing.
    """
    from collections import Counter

    counts = Counter(labels)
    ordered = sorted(counts.items(), key=lambda kv: (kv[0] == "compile worker",
                                                     -kv[1]))
    return [name if n == 1 else f"{n}x {name}" for name, n in ordered]


def stage_status() -> list[dict]:
    log = (LOGS / "pipeline-session.log")
    text = log.read_text(errors="replace")[-40000:] if log.exists() else ""
    reached = re.findall(r"=== \[[\d:]+\] stage (\d+)[^=]*===", text)
    current = int(reached[-1]) if reached else None
    rows = []
    for i, (marker, label, needs) in enumerate(STAGES):
        done = bool(marker) and (PIPELINE / f"{marker}.done").exists()
        rows.append({
            "label": label, "needs": needs, "done": done,
            "running": current == i and not done,
        })
    return rows


def tb_progress() -> dict:
    d = ROOT / "artifacts" / "06"
    auth = sorted((d / "authoritative").glob("*.json")) \
        if (d / "authoritative").is_dir() else []
    cand = list((d / "candidates").glob("*.json")) if (d / "candidates").is_dir() else []
    log = LOGS / "01-tb-authoritative.log"
    pending = total = None
    if log.exists():
        text = log.read_text(errors="replace")
        m = re.findall(r"re-time\s+(\d+) problems, (\d+) pending", text)
        if m:
            total, pending = int(m[-1][0]), int(m[-1][1])
        last = re.findall(r"\[(\d+)/(\d+)\]\s+(\S+)\s+(\S+)", text)
    else:
        last = []
    # Durations from file mtimes: the log does not record per-problem wall time,
    # and the distribution is the interesting part -- most are under a minute and
    # a few are over an hour, so a mean says nothing.
    times = sorted(p.stat().st_mtime for p in auth)
    gaps = [(b - a) / 60 for a, b in zip(times, times[1:]) if 0 < b - a < 6 * 3600]
    gaps_sorted = sorted(gaps)
    return {
        "authoritative": len(auth),
        "candidates": len(cand),
        "retime_total": total,
        "retime_pending": pending,
        "recent": [{"n": a, "of": b, "status": c, "problem": e}
                   for a, b, c, e in last[-8:]],
        "median_min": gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else None,
        "p90_min": gaps_sorted[int(len(gaps_sorted) * 0.9)] if gaps_sorted else None,
        "max_min": gaps_sorted[-1] if gaps_sorted else None,
        "n_timed": len(gaps),
    }


def sweep_progress() -> dict:
    runs = ROOT / "artifacts" / "10" / "runs"
    out: dict = {"runs": {}}
    if not runs.is_dir():
        return out
    for run in sorted(runs.iterdir()):
        if not run.is_dir():
            continue
        cfg = run / "config.json"
        units = None
        if cfg.exists():
            try:
                units = json.loads(cfg.read_text()).get("units")
            except Exception:
                pass
        per: dict[str, dict] = {}
        for sess in run.glob("*/*/session.json"):
            h = sess.parent.parent.name
            slot = per.setdefault(h, {"done": 0, "solved": 0, "cost": 0.0})
            slot["done"] += 1
            try:
                d = json.loads(sess.read_text())
            except Exception:
                continue
            if d.get("produced_solution"):
                slot["solved"] += 1
            if d.get("cost_usd"):
                slot["cost"] += d["cost_usd"]
        if per:
            out["runs"][run.name] = {"units": units, "harnesses": per}
    return out


def snapshot() -> dict:
    return {
        "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": subprocess.run(["hostname"], capture_output=True,
                               text=True).stdout.strip(),
        "driver_alive": bool(subprocess.run(
            ["pgrep", "-f", "run_pipeline.sh"], capture_output=True).stdout),
        "stages": stage_status(),
        "gpus": smi(),
        "t_b": tb_progress(),
        "sweep": sweep_progress(),
    }


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>SOL-ExecBench-AMD · pipeline</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--ink:#e6e9ef;--dim:#98a1b3;
--accent:#5b9dff;--good:#3fb950;--warn:#d29922;--bad:#f85149;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:19px;margin:0 0 2px}h2{font-size:13px;text-transform:uppercase;
letter-spacing:.06em;color:var(--dim);margin:26px 0 10px}
.sub{color:var(--dim);font-size:12.5px;margin:0 0 18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:14px 16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase}
.mono{font-family:var(--mono);font-size:12px}
.bar{height:6px;background:#21262f;border-radius:3px;overflow:hidden}
.bar>span{display:block;height:100%;background:var(--accent)}
.ok{color:var(--good)}.no{color:var(--bad)}.mid{color:var(--warn)}
.dim{color:var(--dim)}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;
border:1px solid var(--line);color:var(--dim)}
.run{border-color:var(--accent);color:var(--accent)}
.idle{opacity:.45}
</style></head><body><div class="wrap">
<h1>SOL-ExecBench-AMD · task 10 pipeline</h1>
<p class="sub" id="sub">loading…</p><div id="app"></div>
</div><script>
const f=(v,d=1)=>v==null?'<span class="dim">–</span>':Number(v).toFixed(d);
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function gpus(d){
  const busy=d.gpus.filter(g=>(g.use||0)>5).length;
  const rows=d.gpus.map(g=>{
    const on=(g.use||0)>5;
    return `<tr class="${on?'':'idle'}">
      <td class="mono">torch ${g.torch??'?'} <span class="dim">/ smi ${g.smi}</span></td>
      <td>${g.use??0}%${'<div class="bar"><span style="width:'+(g.use||0)+'%"></span></div>'}</td>
      <td>${f(g.sclk,0)} MHz</td><td>${f(g.power,0)} W</td>
      <td class="mono dim">${(g.holders||[]).map(esc).join(' ')||'—'}</td></tr>`}).join('');
  return `<h2>GPUs — ${busy} of ${d.gpus.length} doing work</h2><div class="panel">
   <table><thead><tr><th>device</th><th>util</th><th>clock</th><th>power</th>
   <th>holder (pid:job)</th></tr></thead><tbody>${rows}</tbody></table>
   ${busy<=1?'<p class="sub" style="margin:10px 0 0">One GPU busy is <b>expected</b> during stage 1 and stage 4: authoritative timing runs on a single GPU by construction, because at one determinism setting this node\\'s eight GPUs hold clocks spanning 1318–1644 MHz. The other seven are idle by design, not by fault.</p>':''}
  </div>`}
function stages(d){
  return `<h2>Stages</h2><div class="panel"><table><thead><tr><th>stage</th>
  <th>needs</th><th>state</th></tr></thead><tbody>`+d.stages.map(s=>`<tr class="${s.done||s.running?'':'idle'}">
   <td>${esc(s.label)}</td><td class="dim">${esc(s.needs)}</td>
   <td>${s.done?'<span class="ok">done</span>':s.running?'<span class="pill run">running</span>':'<span class="dim">pending</span>'}</td>
  </tr>`).join('')+`</tbody></table></div>`}
function tb(d){
  const t=d.t_b,pct=t.retime_total?100*(t.retime_total-(t.retime_pending||0))/t.retime_total:null;
  const recent=t.recent.map(r=>`<tr><td class="mono">${r.n}/${r.of}</td>
   <td class="${r.status==='ok'?'ok':'no'}">${esc(r.status)}</td>
   <td class="mono dim" style="text-align:left">${esc(r.problem)}</td></tr>`).join('');
  return `<h2>Stage 1 · authoritative T_b queue</h2><div class="panel">
   <table><tbody>
    <tr><td>artifacts written</td><td>${t.authoritative} <span class="dim">of ${t.candidates} candidates</span></td></tr>
    <tr><td>re-time queue</td><td>${t.retime_pending==null?'<span class="dim">–</span>':t.retime_pending+' pending of '+t.retime_total}</td></tr>
    <tr><td>per-problem, median / p90 / max</td><td>${f(t.median_min)} / ${f(t.p90_min)} / ${f(t.max_min)} min <span class="dim">(n=${t.n_timed})</span></td></tr>
   </tbody></table>
   ${pct!=null?'<div class="bar" style="margin:10px 0"><span style="width:'+pct+'%"></span></div>':''}
   <p class="sub" style="margin:8px 0 0">The duration spread is the point: a mean would hide it. Most problems finish in well under a minute; a few large GEMMs re-run <span class="mono">max_autotune</span> and take over an hour.</p>
   ${recent?'<table style="margin-top:10px"><thead><tr><th>#</th><th>status</th><th style="text-align:left">problem</th></tr></thead><tbody>'+recent+'</tbody></table>':''}
  </div>`}
function sweep(d){
  const runs=Object.entries(d.sweep.runs||{});
  if(!runs.length)return '<h2>Stage 5 · agent sweep</h2><div class="panel"><p class="sub" style="margin:0">Not started. It needs the seven non-authoritative GPUs, and cannot overlap a timing stage — compile-heavy agents starve a timing run of CPU, which is what voided the first T_b measurement.</p></div>';
  return '<h2>Stage 5 · agent sweep</h2>'+runs.map(([id,r])=>{
    const rows=Object.entries(r.harnesses).map(([h,v])=>`<tr><td>${esc(h)}</td>
     <td>${v.done}</td><td>${v.solved}</td>
     <td>${v.cost?'$'+v.cost.toFixed(2):'<span class="dim">not reported</span>'}</td></tr>`).join('');
    return `<div class="panel" style="margin-bottom:10px"><b class="mono">${esc(id)}</b>
     <span class="dim">${r.units?'· '+r.units+' units':''}</span>
     <table style="margin-top:8px"><thead><tr><th>harness</th><th>sessions</th>
     <th>with a solution</th><th>reported cost</th></tr></thead><tbody>${rows}</tbody></table></div>`}).join('')}
async function tick(){
  try{
    const d=await (await fetch('status.json',{cache:'no-store'})).json();
    document.getElementById('sub').innerHTML=
      `${esc(d.host)} · ${esc(d.now)} · driver ${d.driver_alive?'<span class="ok">alive</span>':'<span class="no">not running</span>'} · refreshes every 5s`;
    document.getElementById('app').innerHTML=
      gpus(d)+stages(d)+tb(d)+sweep(d);
  }catch(e){document.getElementById('sub').textContent='status unavailable: '+e}
}
tick();setInterval(tick,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/status.json"):
            body = json.dumps(snapshot(), default=str).encode()
            ctype = "application/json"
        else:
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # keep the terminal readable
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 by default; this exposes process command "
                         "lines, so bind wider only deliberately")
    ap.add_argument("--once", action="store_true",
                    help="print one snapshot as JSON and exit")
    a = ap.parse_args()

    if a.once:
        print(json.dumps(snapshot(), indent=2, default=str))
        return 0

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"status page on http://{a.host}:{a.port}")
    print(f"  ssh -N -L {a.port}:localhost:{a.port} {snapshot()['host']}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
