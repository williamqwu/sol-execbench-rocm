#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render REPORT.md into the standalone HTML page published as an Artifact.

Deliberately minimal: the report uses only headings, tables, fenced code,
blockquotes, lists, hr, and inline emphasis/code, so a full markdown
dependency would be more surface than the job needs. Regenerate with

    python artifacts/11/compile-diag/build_page.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "REPORT.md"
OUT = HERE / "report.html"


def inline(s: str) -> str:
    """Inline spans.

    Code spans are lifted out to placeholders *before* emphasis runs, not
    split on: `**bold with `code` inside**` is common in the report, and
    splitting would leave the ``**`` stranded in two different fragments with
    neither able to match.
    """
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(held) - 1}\x00"

    s = re.sub(r"`([^`]+)`", hold, s)
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<em>\1</em>", s)
    s = s.replace("--", "&#8211;")
    return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], s)


def cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def render(md: str) -> tuple[str, list[tuple[str, str]]]:
    lines = md.split("\n")
    out: list[str] = []
    toc: list[tuple[str, str]] = []
    i, n = 0, len(lines)

    while i < n:
        ln = lines[i]

        # fenced code
        if ln.startswith("```"):
            lang = ln[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append(f'<div class="scroll"><pre class="code" data-lang="{html.escape(lang)}">'
                       f'<code>{html.escape(chr(10).join(buf))}</code></pre></div>')
            continue

        # table
        if ln.startswith("|") and i + 1 < n and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            head = cells(ln)
            i += 2
            body = []
            while i < n and lines[i].startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tr = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                for r in body)
            out.append(f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
                       f"<tbody>{tr}</tbody></table></div>")
            continue

        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2)
            slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"[`*]", "", txt).lower()).strip("-")
            num = re.match(r"^(\d+(?:\.\d+)?)\.?\s+(.*)$", re.sub(r"[`*]", "", txt))
            if lvl == 2 and num:
                toc.append((slug, num.group(2)))
                out.append(f'<h2 id="{slug}"><span class="secnum">{num.group(1)}</span>'
                           f"<span>{inline(num.group(2))}</span></h2>")
            else:
                out.append(f'<h{lvl} id="{slug}">{inline(txt)}</h{lvl}>')
            i += 1
            continue

        if ln.strip() in ("---", "***"):
            out.append("<hr>")
            i += 1
            continue

        # blockquote
        if ln.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{inline(' '.join(buf))}</blockquote>")
            continue

        # list
        if re.match(r"^\s*[*+-]\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.\s+", ln))
            items: list[str] = []
            while i < n and (re.match(r"^\s*[*+-]\s+", lines[i])
                             or re.match(r"^\s*\d+\.\s+", lines[i])
                             or (lines[i].startswith("  ") and lines[i].strip() and items)):
                if re.match(r"^\s*(?:[*+-]|\d+\.)\s+", lines[i]):
                    items.append(re.sub(r"^\s*(?:[*+-]|\d+\.)\s+", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        if not ln.strip():
            i += 1
            continue

        # paragraph
        buf = [ln]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|\||>|```|\s*[*+-]\s|\s*\d+\.\s|---$)", lines[i]):
            buf.append(lines[i])
            i += 1
        para = " ".join(buf)
        cls = ' class="standfirst"' if para.startswith("*") and para.endswith("*") else ""
        out.append(f"<p{cls}>{inline(para)}</p>")

    return "\n".join(out), toc


# The five deep-dive problems, as the tolerance-gap strip plots them.
# atol / divergence are the worst measured workload of each, from REPORT.md 2.4.
GAPS = [
    ("L2__009", "decoder layer", "fp32", 1.375e-07, 5.8e-06, 0.415),
    ("L1__067", "GQA attention", "fp32", 3.446e-08, 2.03e-06, 0.249),
    ("Quant__004", "fp8 MoE expert", "bf16", 4.85e-03, 2.03e-01, 0.734),
    ("L1__062", "KV-cache rope bwd", "bf16", 8.26e-03, 1.25e-01, 0.977),
    ("L2__058", "mamba2 scan", "bf16", 4.666e-03, 2.34e-02, 0.985),
]


def strip_svg() -> str:
    import math
    rows = []
    ratios = [d / a for _, _, _, a, d, _ in GAPS]
    hi = max(ratios)
    for name, what, fl, atol, div, mr in GAPS:
        r = div / atol
        frac = math.log10(r) / math.log10(hi)
        rows.append(f"""
        <div class="gap-row">
          <div class="gap-id"><span class="gap-key">{name}</span>
            <span class="gap-what">{what} &middot; {fl}</span></div>
          <div class="gap-track" role="img"
               aria-label="{name}: measured divergence {div:.2e} is {r:.0f} times the tolerance {atol:.2e}">
            <div class="gap-fill" style="width:{max(frac * 100, 4):.1f}%"></div>
            <span class="gap-mult">{r:.0f}&times;</span>
          </div>
          <div class="gap-mr {'is-near' if mr > 0.9 else ''}">{mr:.3f}</div>
        </div>""")
    return "".join(rows)


def main() -> int:
    body, toc = render(SRC.read_text())
    nav = "".join(
        f'<li><a href="#{s}"><span>{i + 1}</span>{html.escape(t)}</a></li>'
        for i, (s, t) in enumerate(toc))

    OUT.write_text(TEMPLATE.replace("{{BODY}}", body)
                   .replace("{{NAV}}", nav)
                   .replace("{{STRIP}}", strip_svg()))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes), {len(toc)} sections")
    return 0


TEMPLATE = r"""<title>torch.compile fails 71 problems: a tolerance defect, not a compiler defect</title>
<style>
:root{
  --paper:#eaeff1; --surface:#fbfcfd; --sunk:#e2e9ec;
  --ink:#111a1d; --ink-soft:#33474e; --muted:#5b6f76; --faint:#8ba0a7;
  --rule:#c8d5d9; --rule-soft:#dbe4e7;
  --accent:#0c6a74; --accent-soft:#d3e7e9;
  --fail:#a82a19; --fail-soft:#f5dcd8;
  --pass:#2a6642; --warn:#7f5900;
  --shadow:0 1px 2px rgba(17,26,29,.06), 0 8px 24px -16px rgba(17,26,29,.28);
  --measure:69ch;
  --f-display:"Iowan Old Style","Source Serif 4","Charter",Palatino,Georgia,serif;
  --f-body:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --f-mono:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0d1417; --surface:#141d21; --sunk:#0a1013;
  --ink:#e4ecee; --ink-soft:#bdcdd1; --muted:#8ba0a6; --faint:#647a80;
  --rule:#26343a; --rule-soft:#1c282d;
  --accent:#54bec8; --accent-soft:#153037;
  --fail:#ef8375; --fail-soft:#331915;
  --pass:#74c294; --warn:#d7a833;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --paper:#0d1417; --surface:#141d21; --sunk:#0a1013;
  --ink:#e4ecee; --ink-soft:#bdcdd1; --muted:#8ba0a6; --faint:#647a80;
  --rule:#26343a; --rule-soft:#1c282d;
  --accent:#54bec8; --accent-soft:#153037;
  --fail:#ef8375; --fail-soft:#331915;
  --pass:#74c294; --warn:#d7a833;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--f-body); font-size:16.5px; line-height:1.62;
  font-variant-numeric:tabular-nums;
}
.wrap{max-width:min(var(--measure),calc(100vw - 3rem)); margin:0 auto; padding:0 0 6rem}

/* ---- masthead: a data strip, not a hero ---- */
.mast{border-bottom:1px solid var(--rule); background:var(--surface); margin-bottom:3.5rem}
.mast-in{max-width:min(78ch,calc(100vw - 3rem)); margin:0 auto; padding:3rem 0 2.25rem;
  display:flex; flex-direction:column; gap:1.25rem}
.eyebrow{font-family:var(--f-mono); font-size:.7rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--accent); display:flex; flex-wrap:wrap; gap:.9rem}
.eyebrow span{color:var(--faint)}
h1{font-family:var(--f-display); font-weight:600; font-size:clamp(1.85rem,4.6vw,2.9rem);
  line-height:1.14; letter-spacing:-.015em; margin:0; text-wrap:balance}
h1 em{font-style:normal; color:var(--accent)}
.dek{margin:0; font-size:1.09rem; color:var(--ink-soft); max-width:62ch; text-wrap:pretty}

.facts{display:grid; grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr));
  gap:1px; background:var(--rule-soft); border:1px solid var(--rule-soft);
  border-radius:3px; overflow:hidden; margin-top:.5rem}
.fact{background:var(--surface); padding:.8rem .95rem}
.fact b{display:block; font-family:var(--f-mono); font-size:1.32rem; font-weight:600;
  letter-spacing:-.02em; line-height:1.2}
.fact small{display:block; color:var(--muted); font-size:.755rem; line-height:1.35; margin-top:.25rem}
.fact.f-bad b{color:var(--fail)} .fact.f-ok b{color:var(--pass)} .fact.f-warn b{color:var(--warn)}

/* ---- the tolerance-gap strip: the finding in one graphic ---- */
.strip{margin:0 0 3.25rem; padding:1.5rem 1.4rem 1.25rem; background:var(--surface);
  border:1px solid var(--rule); border-radius:4px; box-shadow:var(--shadow)}
.strip h3{margin:0 0 .3rem; font-family:var(--f-body); font-size:.83rem; font-weight:650;
  letter-spacing:.03em; text-transform:uppercase; color:var(--ink)}
.strip .legend{margin:0 0 1.15rem; font-size:.83rem; color:var(--muted); max-width:58ch}
.gap-row{display:grid; grid-template-columns:minmax(7.5rem,auto) 1fr 3.4rem;
  align-items:center; gap:.85rem; padding:.42rem 0; border-top:1px solid var(--rule-soft)}
.gap-row:first-of-type{border-top:0}
.gap-key{display:block; font-family:var(--f-mono); font-size:.815rem; font-weight:600}
.gap-what{display:block; font-size:.715rem; color:var(--faint)}
.gap-track{position:relative; height:1.45rem; background:var(--sunk); border-radius:2px;
  display:flex; align-items:center}
.gap-track::before{content:""; position:absolute; left:0; top:-.1rem; bottom:-.1rem; width:2px;
  background:var(--accent)}
.gap-fill{height:100%; border-radius:2px;
  background:linear-gradient(90deg,var(--accent-soft),var(--fail-soft));
  border-right:2px solid var(--fail)}
.gap-mult{position:absolute; left:.55rem; font-family:var(--f-mono); font-size:.76rem;
  font-weight:600; color:var(--ink-soft)}
.gap-mr{font-family:var(--f-mono); font-size:.8rem; text-align:right; color:var(--fail)}
.gap-mr.is-near{color:var(--warn)}
.strip .foot{margin:.9rem 0 0; font-size:.755rem; color:var(--faint); line-height:1.5}

/* ---- contents ---- */
.toc{margin:0 0 3.5rem; padding:1.25rem 1.4rem; background:var(--surface);
  border:1px solid var(--rule); border-radius:4px}
.toc h3{margin:0 0 .7rem; font-size:.72rem; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); font-weight:650; font-family:var(--f-mono)}
.toc ol{margin:0; padding:0; list-style:none; display:grid; gap:.28rem}
.toc a{display:flex; gap:.7rem; text-decoration:none; color:var(--ink-soft); font-size:.92rem;
  padding:.14rem .2rem; border-radius:2px}
.toc a span{font-family:var(--f-mono); color:var(--accent); font-size:.8rem; min-width:1.1rem}
.toc a:hover{background:var(--accent-soft); color:var(--ink)}

/* ---- prose ---- */
h2{font-family:var(--f-display); font-size:1.62rem; font-weight:600; line-height:1.2;
  letter-spacing:-.012em; margin:3.6rem 0 1.1rem; padding-top:1.5rem;
  border-top:2px solid var(--rule); display:flex; gap:.85rem; align-items:baseline;
  text-wrap:balance; scroll-margin-top:1.5rem}
.secnum{font-family:var(--f-mono); font-size:.85rem; font-weight:600; color:var(--accent);
  letter-spacing:.02em}
h3{font-family:var(--f-display); font-size:1.19rem; font-weight:600; line-height:1.3;
  margin:2.5rem 0 .7rem; letter-spacing:-.008em; text-wrap:balance; scroll-margin-top:1.5rem}
h4{font-size:.95rem; font-weight:650; margin:1.8rem 0 .5rem; color:var(--ink-soft)}
p{margin:0 0 1.05rem; text-wrap:pretty}
.standfirst{color:var(--muted); font-size:.93rem; line-height:1.6; border-left:2px solid var(--rule);
  padding-left:1rem; margin-bottom:2rem}
.standfirst em{font-style:normal}
strong{font-weight:650; color:var(--ink)}
em{font-style:italic}
ul,ol{margin:0 0 1.15rem; padding-left:1.3rem; display:grid; gap:.42rem}
li{padding-left:.15rem}
li::marker{color:var(--faint)}
hr{border:0; height:1px; background:var(--rule-soft); margin:2.5rem 0}
blockquote{margin:1.4rem 0; padding:.9rem 1.1rem; background:var(--surface);
  border-left:3px solid var(--warn); border-radius:0 3px 3px 0; font-size:.94rem;
  color:var(--ink-soft)}
blockquote p{margin:0}
a{color:var(--accent); text-decoration-thickness:1px; text-underline-offset:2px}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:2px}

code{font-family:var(--f-mono); font-size:.855em; background:var(--sunk);
  padding:.1em .34em; border-radius:2px; color:var(--ink-soft);
  overflow-wrap:break-word}
strong code{color:var(--ink); font-weight:600}

/* ---- breakout: tables and code run wider than the measure ---- */
.scroll{overflow-x:auto; margin:1.5rem 0;
  width:min(84ch,calc(100vw - 3rem));
  margin-left:calc(min(0px, (var(--measure) - min(84ch,100vw - 3rem)) / 2));
  -webkit-overflow-scrolling:touch}
pre.code{margin:0; padding:1rem 1.15rem; background:var(--surface);
  border:1px solid var(--rule); border-left:3px solid var(--accent); border-radius:0 3px 3px 0;
  font-family:var(--f-mono); font-size:.795rem; line-height:1.62; color:var(--ink-soft)}
pre.code code{background:none; padding:0; font-size:inherit; color:inherit}
table{border-collapse:collapse; width:100%; font-size:.815rem; font-family:var(--f-mono);
  background:var(--surface); border:1px solid var(--rule)}
thead th{text-align:left; font-weight:650; color:var(--ink); background:var(--sunk);
  padding:.55rem .7rem; border-bottom:1px solid var(--rule); white-space:nowrap;
  font-size:.755rem; letter-spacing:.02em}
td{padding:.48rem .7rem; border-top:1px solid var(--rule-soft); vertical-align:top;
  color:var(--ink-soft)}
tbody tr:hover td{background:var(--accent-soft)}
td code,th code{background:none; padding:0; font-size:1em}
td strong{color:var(--ink)}

footer{margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--rule);
  font-size:.78rem; color:var(--faint); line-height:1.6}

@media (max-width:640px){
  body{font-size:16px}
  .mast-in{padding:2.25rem 0 1.75rem}
  .gap-row{grid-template-columns:1fr 3.2rem; row-gap:.3rem}
  .gap-track{grid-column:1/-1}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
</style>

<header class="mast">
  <div class="mast-in">
    <div class="eyebrow">
      SOL-ExecBench-ROCm <span>&middot;</span> MI350X / gfx950 <span>&middot;</span>
      ROCm 7.2.0 <span>&middot;</span> torch 2.9.1 <span>&middot;</span> 2026-08-11
    </div>
    <h1>The gate asks a kernel to reproduce eager&rsquo;s rounding.
      <em>torch.compile cannot, and should not.</em></h1>
    <p class="dek">Why <code>torch.compile</code> and <code>max-autotune</code> fail ~70 problems on
      the leaderboard &mdash; traced from the tolerance derivation through five problems reproduced
      on GPU, adjudicated against float64 goldens, and checked by fifteen adversarial verifiers.</p>
    <div class="facts">
      <div class="fact f-bad"><b>523</b><small>workloads rejected<br>INCORRECT_NUMERICAL</small></div>
      <div class="fact f-bad"><b>71</b><small>problems, v2_compile<br>L1 24 &middot; L2 41 &middot; Quant 6</small></div>
      <div class="fact f-warn"><b>96.3%</b><small>of tolerances are<br>pure epsilon floor</small></div>
      <div class="fact f-warn"><b>2061</b><small>workloads never<br>actually compiled</small></div>
      <div class="fact f-ok"><b>5 of 6</b><small>cases where compiled<br>beats eager vs float64</small></div>
    </div>
  </div>
</header>

<main class="wrap">

<section class="strip">
  <h3>How far past the bound, and by how much</h3>
  <p class="legend">The teal line is the workload&rsquo;s own tolerance. The bar runs to the measured
    compiled-vs-eager divergence, on a log scale. The right column is <code>matched_ratio</code>
    against the required <strong>0.990</strong>.</p>
  {{STRIP}}
  <p class="foot">Worst workload of each problem, measured on GPUs 4&ndash;7 in the
    <code>solbench</code> container. Eager&ndash;vs&ndash;eager on every one of these is exactly
    <code>0.000e+00</code>, which is what floors the tolerance at one ULP in the first place.</p>
</section>

<nav class="toc">
  <h3>Contents</h3>
  <ol>{{NAV}}</ol>
</nav>

{{BODY}}

<footer>
  Generated from <code>artifacts/11/compile-diag/REPORT.md</code> by
  <code>build_page.py</code>. Underlying measurements and scripts are in
  <code>artifacts/11/compile-diag/</code>; nothing outside that directory was modified and
  <code>leaderboard/solbench.db</code> was read only. No number on this page was estimated
  unless it says so.
</footer>

</main>
"""

if __name__ == "__main__":
    raise SystemExit(main())
