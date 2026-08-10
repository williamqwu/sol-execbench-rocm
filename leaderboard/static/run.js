/* SPDX-License-Identifier: Apache-2.0
 *
 * Interaction for the run page: the workload grid, the trajectory chart, and
 * the one tooltip they share.
 *
 * Everything here is an enhancement over markup that already answers the
 * question. The grid cells carry `title`, the chart keeps its <title>
 * elements, and the detail table is inside a <details>, which opens without
 * script. If this file fails to load the page is slower to read, not wrong --
 * which is the only acceptable arrangement for a page whose job is to show
 * measurements.
 */
(function () {
  "use strict";

  function esc(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;";
    });
  }

  /* ------------------------------------------------------------- tooltip */
  /* Positioned HTML rather than the native `title` / SVG <title>: those wait
     about a second, cannot be styled, and cannot hold the six numbers a
     workload needs. The server-rendered `title` attributes stay in the markup
     for the no-JS case, and are moved out of the way below so a reader does
     not get both. */
  var tip = null;

  function tipEl() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "tip";
      tip.setAttribute("role", "tooltip");
      tip.hidden = true;
      document.body.appendChild(tip);
    }
    return tip;
  }

  function rowsHTML(title, pairs, foot) {
    var h = '<div class="tip-h">' + esc(title) + "</div>";
    pairs.forEach(function (p) {
      if (p[1] === null || p[1] === undefined || p[1] === "") return;
      h += '<div class="tip-r"><span class="tip-k">' + esc(p[0]) +
           '</span><span class="tip-v' + (p[2] ? " " + p[2] : "") + '">' +
           esc(p[1]) + "</span></div>";
    });
    if (foot) h += '<div class="tip-f">' + esc(foot) + "</div>";
    return h;
  }

  function showTip(anchor, html) {
    var t = tipEl();
    t.innerHTML = html;
    t.hidden = false;
    var a = anchor.getBoundingClientRect();
    var b = t.getBoundingClientRect();
    var vw = document.documentElement.clientWidth;
    var x = a.left + window.scrollX + a.width / 2 - b.width / 2;
    var y = a.top + window.scrollY - b.height - 9;
    /* Flip below when there is no room above -- a tooltip clipped by the
       sticky header is a tooltip nobody can read. */
    if (a.top - b.height - 9 < 62) y = a.bottom + window.scrollY + 9;
    x = Math.max(window.scrollX + 8,
                 Math.min(x, window.scrollX + vw - b.width - 8));
    t.style.left = Math.round(x) + "px";
    t.style.top = Math.round(y) + "px";
  }

  function hideTip() { if (tip) tip.hidden = true; }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") hideTip();
  });
  window.addEventListener("scroll", hideTip, { passive: true });

  /* Arrow keys walk a list of peers; Home/End jump to its ends. Roving
     tabindex, so a 43-cell grid is one tab stop and not forty-three. Applied
     from script only: without it every cell keeps its natural focusability,
     which is the better no-JS default. */
  function roving(items) {
    if (!items.length) return;
    items.forEach(function (el, i) { el.setAttribute("tabindex", i ? "-1" : "0"); });
    items.forEach(function (el, i) {
      el.addEventListener("keydown", function (e) {
        var j = null;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") j = i + 1;
        else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = i - 1;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = items.length - 1;
        if (j === null) return;
        j = Math.max(0, Math.min(items.length - 1, j));
        e.preventDefault();
        items.forEach(function (o) { o.setAttribute("tabindex", "-1"); });
        items[j].setAttribute("tabindex", "0");
        items[j].focus();
      });
    });
  }

  /* --------------------------------------------------------- the workload grid */
  var cells = Array.prototype.slice.call(
    document.querySelectorAll(".wl-grid .cell"));

  function cellTip(c) {
    var d = c.dataset;
    return rowsHTML(
      "workload " + d.i + (d.uuid ? " · " + d.uuid.slice(0, 8) : ""),
      [["axes", d.axes],
       ["status", d.state],
       ["T_SOL", d.tsol ? d.tsol + " ms" : null],
       ["T_b", d.tb ? d.tb + " ms" : null],
       ["T_k", d.tk ? d.tk + " ms" : null],
       ["vs T_b", d.speedup, d.beat === "1" ? "ok" : ""],
       ["S", d.s, d.beat === "1" ? "ok" : ""]],
      d.foot || null);
  }

  function markRow(id, on) {
    if (!id) return;
    var tr = document.getElementById(id);
    if (tr) tr.classList.toggle("hot", on);
  }

  cells.forEach(function (c) {
    /* The native tooltip would fire alongside this one. Kept as an attribute
       the script owns rather than deleted, so the information is still in the
       DOM for anything reading it. */
    var t = c.getAttribute("title");
    if (t !== null) {
      c.setAttribute("data-title", t);
      c.setAttribute("aria-label", t);
      c.removeAttribute("title");
    }
    var enter = function () { showTip(c, cellTip(c)); markRow(c.dataset.row, true); };
    var leave = function () { hideTip(); markRow(c.dataset.row, false); };
    c.addEventListener("mouseenter", enter);
    c.addEventListener("focus", enter);
    c.addEventListener("mouseleave", leave);
    c.addEventListener("blur", leave);
    /* Clicking a cell is the way over to the row it stands for. The grid shows
       the shape of the result; the table is where the numbers are. */
    c.addEventListener("click", function () {
      var det = document.querySelector(".wl-details");
      if (det) det.open = true;
      var tr = document.getElementById(c.dataset.row);
      if (!tr) return;
      tr.scrollIntoView({ block: "center", behavior: "smooth" });
      tr.classList.add("flash");
      setTimeout(function () { tr.classList.remove("flash"); }, 1600);
    });
  });
  roving(cells);

  /* Hovering a table row lights the matching cell, so a reader who found the
     row first can see where it sits in the grid. */
  document.querySelectorAll(".wl-details tr[id]").forEach(function (tr) {
    var c = document.querySelector('.wl-grid .cell[data-row="' + tr.id + '"]');
    if (!c) return;
    tr.addEventListener("mouseenter", function () { c.classList.add("hot"); });
    tr.addEventListener("mouseleave", function () { c.classList.remove("hot"); });
  });

  /* ------------------------------------------------------- trajectory chart */
  var pts = Array.prototype.slice.call(
    document.querySelectorAll(".traj [data-eval]"));

  function evalRows(n) {
    return document.querySelectorAll('tr[data-eval="' + n + '"]');
  }

  function pointTip(p) {
    var d = p.dataset;
    return rowsHTML(
      "eval " + d.eval + (d.at ? " · +" + d.at + " min" : ""),
      [["when", d.local || d.utc],
       ["mean S", d.s],
       /* The sign class comes from the server, which decided it from the
          float. Deriving it here from the printed string got it wrong on
          exactly the value that needs it most: parseFloat("-0.0000") is
          negative zero, `-0 < 0` is false, so a delta the table painted red
          came out green in this tooltip -- the same number, two colours, in
          the two views this file exists to wire together. */
       ["Δ vs best", d.delta, d.deltaCls || ""],
       ["passed", d.passed],
       ["kernel", d.lines]],
      d.note || null);
  }

  pts.forEach(function (p) {
    /* The <title> child is the no-JS tooltip and the accessible name. Once
       this runs it would double up with the HTML tooltip, so it moves to
       aria-label: same text, same role, no second popup. */
    var t = p.querySelector("title");
    if (t) {
      p.setAttribute("aria-label", t.textContent);
      t.remove();
    }
    p.setAttribute("role", "button");
    /* A timestamp is only rendered in UTC on the server; the chart says it in
       the reader's own timezone for the same reason every other time on this
       page does. */
    if (p.dataset.utc) {
      var d = new Date(p.dataset.utc);
      if (!isNaN(d)) p.dataset.local = d.toLocaleString();
    }
    var enter = function () {
      showTip(p, pointTip(p));
      p.classList.add("hot");
      evalRows(p.dataset.eval).forEach(function (tr) { tr.classList.add("hot"); });
    };
    var leave = function () {
      hideTip();
      p.classList.remove("hot");
      evalRows(p.dataset.eval).forEach(function (tr) { tr.classList.remove("hot"); });
    };
    p.addEventListener("mouseenter", enter);
    p.addEventListener("focus", enter);
    p.addEventListener("mouseleave", leave);
    p.addEventListener("blur", leave);
    /* The caption under the chart says a point lights up "its row in the table
       below". Since 2026-08-10 that table starts collapsed, so clicking a point
       has to open it -- otherwise the caption describes something the reader
       cannot see happening, which is worse than not linking them at all. Same
       affordance the workload grid already has. */
    p.addEventListener("click", function () {
      var det = p.closest("section, div, body").querySelector(".tbl-details")
             || document.querySelector(".tbl-details");
      if (det) det.open = true;
      var tr = evalRows(p.dataset.eval)[0];
      if (!tr) return;
      tr.scrollIntoView({ block: "center", behavior: "smooth" });
      tr.classList.add("flash");
      setTimeout(function () { tr.classList.remove("flash"); }, 1600);
    });
  });
  roving(pts);

  /* The other direction: a row lights its point, including the axis ticks,
     which are evals too -- an eval that produced no score is still a place the
     run went, and leaving the ticks out of the linking would quietly demote
     them to decoration. */
  document.querySelectorAll("tr[data-eval]").forEach(function (tr) {
    var p = document.querySelector('.traj [data-eval="' + tr.dataset.eval + '"]');
    if (!p) return;
    tr.addEventListener("mouseenter", function () { p.classList.add("hot"); });
    tr.addEventListener("mouseleave", function () { p.classList.remove("hot"); });
  });
})();
