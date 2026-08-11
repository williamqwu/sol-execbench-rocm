/* SPDX-License-Identifier: Apache-2.0
 *
 * Problem page: table filters, and the two column switches on the workload
 * table.
 *
 * Everything here is enhancement. With JS off the page renders every row and
 * every column the server sent -- the optional columns are hidden by a class
 * the stylesheet only applies when this script has set it, so "no script"
 * degrades to "all columns shown", never to "columns missing and no way to
 * get them back".
 */
(function () {
  "use strict";

  /* ------------------------------------------------------------ filters */
  /* Substring over the row's own text, case-insensitive, all terms must match
     so "b200 4096" narrows twice. The count beside the box says what was hidden
     -- a filter with no count is a table that looks short for no reason. */
  function wire(input) {
    var table = document.querySelector(input.getAttribute("data-filter"));
    if (!table) return;
    var body = table.tBodies[0];
    var count = document.querySelector(
      '[data-count-for="' + input.getAttribute("data-filter") + '"]');
    var rows = Array.prototype.slice.call(body.rows);
    var total = rows.length;

    var apply = function () {
      var terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      var shown = 0;
      rows.forEach(function (tr) {
        var hay = (tr.textContent || "").toLowerCase();
        var ok = terms.every(function (t) { return hay.indexOf(t) !== -1; });
        tr.hidden = !ok;
        if (ok) shown++;
      });
      if (count) {
        count.textContent = shown === total
          ? total + (total === 1 ? " row" : " rows")
          : shown + " of " + total + " rows";
      }
    };
    input.addEventListener("input", apply);
    apply();
  }
  document.querySelectorAll("input.tfilter").forEach(wire);

  /* ------------------------------------------------- optional columns */
  /* Remembered in localStorage rather than per page: whether a reader wants
     NVIDIA's B200 figures, or the derivation columns, is a standing preference
     about how they read this board and not a decision to retake on every
     problem. Nothing about a measurement changes; only which columns render. */
  function columnSwitch(id, cls, key, extra) {
    var box = document.getElementById(id);
    var table = document.getElementById("wl-table");
    if (!box || !table) return;
    var on = false;
    try { on = localStorage.getItem(key) === "1"; } catch (e) {}
    var apply = function () {
      box.checked = on;
      table.classList.toggle(cls, !on);   /* the class HIDES, see style.css */
      if (extra) extra(on);
    };
    box.addEventListener("change", function () {
      on = box.checked;
      try { localStorage.setItem(key, on ? "1" : "0"); } catch (e) {}
      apply();
    });
    apply();
  }

  /* Each band's caption travels with its columns: shown only while they are,
     because a paragraph about numbers nobody can see is noise -- and a visible
     caveat over hidden columns reads as a caveat about the AMD numbers. */
  function reveal(sel) {
    var el = document.querySelector(sel);
    return function (on) { if (el) el.hidden = !on; };
  }
  columnSwitch("sw-b200", "no-b200", "cols.b200", reveal(".note-b200"));
  columnSwitch("sw-deriv", "no-deriv", "cols.deriv", reveal(".note-deriv"));

  /* ------------------------------------------------- the reference pane */
  /* Clamped to `data-lines` lines with a button, and clamped HERE rather than
     in the template: a truncation that only exists when the script that undoes
     it is running cannot leave a reader with a listing they cannot finish. The
     height is measured off the rendered pane, so it follows the font and the
     line-height instead of assuming them, and it is re-measured after
     highlight.js has rebuilt the pane with its gutter. */
  (function () {
    var box = document.querySelector(".srcclamp");
    if (!box) return;
    var pre = box.querySelector("pre.code");
    if (!pre) return;
    var want = parseInt(box.getAttribute("data-lines") || "25", 10);

    var measure = function () {
      var lh = parseFloat(getComputedStyle(pre).lineHeight) || 20;
      var pad = pre.offsetHeight - pre.clientHeight + 32;   /* border + padding */
      var lines = (pre.textContent.match(/\n/g) || []).length + 1;
      if (lines <= want + 3) return false;   /* not worth a button */
      box.style.setProperty("--clamp-h", Math.round(want * lh + pad) + "px");
      return true;
    };
    if (!measure()) return;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "srcmore";
    var lines = (pre.textContent.match(/\n/g) || []).length + 1;
    var set = function (open) {
      box.classList.toggle("on", !open);
      btn.textContent = open ? "show fewer lines"
                             : "show all " + lines + " lines";
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    };
    btn.addEventListener("click", function () {
      set(!box.classList.contains("on"));
    });
    box.parentNode.insertBefore(btn, box.nextSibling);
    set(false);
  })();
})();
