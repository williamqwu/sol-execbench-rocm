/* SPDX-License-Identifier: Apache-2.0
 *
 * Syntax highlighting and copy buttons for the code panes. Self-contained: no
 * CDN and no build step, the same constraint the rest of this app is under.
 *
 * The rule is correctness first. Under-highlighting is fine -- an unterminated
 * string, an unknown language, a construct no rule covers all degrade to plain
 * text -- but the pane must never show something the source does not say.
 * Three properties give that, and they are structural rather than careful:
 *
 *   1. EVERY chunk the scanner emits, matched or not, goes through esc().
 *      The only unescaped text ever inserted is a fixed <span class="t-.."> and
 *      its closing tag, from this file.
 *   2. The scanner reads the raw string once, left to right, and never re-reads
 *      its own output. `data-hl` makes that true across calls too: a block that
 *      has been highlighted is never highlighted again, which is the classic
 *      way a highlighter eats its own markup.
 *   3. The copy button carries the raw text captured BEFORE any of this ran, so
 *      what lands on the clipboard is the source and not the DOM's rendering of
 *      it -- no line numbers, no inserted spans.
 */
(function () {
  "use strict";

  function esc(s) {
    return s.replace(/[&<>]/g, function (c) {
      return c === "&" ? "&amp;" : c === "<" ? "&lt;" : "&gt;";
    });
  }

  function words(s) {
    var set = Object.create(null);
    s.split(" ").forEach(function (w) { set[w] = true; });
    return set;
  }

  var PY_KW = words(
    "False None True and as assert async await break class continue def del " +
    "elif else except finally for from global if import in is lambda nonlocal " +
    "not or pass raise return try while with yield match case");
  var PY_LIT = words("self cls True False None NotImplemented Ellipsis __name__");
  var C_KW = words(
    "alignas alignof asm auto bool break case catch char class const constexpr " +
    "continue decltype default delete do double dynamic_cast else enum explicit " +
    "extern false float for friend goto if inline int long mutable namespace new " +
    "noexcept nullptr operator private protected public register reinterpret_cast " +
    "return short signed sizeof static static_cast struct switch template this " +
    "throw true try typedef typename union unsigned using virtual void volatile " +
    "while __global__ __device__ __host__ __shared__ __constant__ __restrict__ " +
    "__launch_bounds__ __forceinline__ __syncthreads");

  /* Each rule is tried in order at the current offset; the first that matches
     wins. Order is the grammar here -- comments and strings come before
     everything, so a `#` inside a string is not a comment and a quote inside a
     comment does not open a string. Every regex is sticky, so a rule can only
     ever match AT the offset, never somewhere ahead of it. */
  function ident(kw, lit) {
    return {
      re: /[A-Za-z_][A-Za-z0-9_]*/y,
      cls: function (t) {
        return kw[t] ? "t-kw" : lit[t] ? "t-lit" : null;
      }
    };
  }

  /* `def foo` / `class Foo` as one match, so the name is styled without the
     scanner having to remember what it saw last. */
  var PY_DEF = {
    re: /\b(def|class)([ \t]+)([A-Za-z_][A-Za-z0-9_]*)/y,
    render: function (m) {
      return '<span class="t-kw">' + esc(m[1]) + "</span>" + esc(m[2]) +
             '<span class="t-def">' + esc(m[3]) + "</span>";
    }
  };

  var NUMBER = {
    re: /(?:0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(?:\.[\d_]*)?(?:[eE][+-]?\d+)?|\.\d[\d_]*(?:[eE][+-]?\d+)?)[jJfFuUlL]*/y,
    cls: "t-num"
  };
  /* A run of punctuation, whitespace or anything else uninteresting, consumed
     in one step. Purely a speed rule: without it the fallback advances one
     character per iteration through every operator and indent in the file. */
  var PLAIN = { re: /[^A-Za-z0-9_"'`#@\/*]+/y, cls: null };

  var LANGS = {
    python: [
      { re: /#[^\n]*/y, cls: "t-com" },
      { re: /[rRbBuUfF]{0,3}(?:"""[\s\S]*?"""|'''[\s\S]*?''')/y, cls: "t-str" },
      { re: /[rRbBuUfF]{0,3}(?:"(?:\\[\s\S]|[^"\\\n])*"|'(?:\\[\s\S]|[^'\\\n])*')/y, cls: "t-str" },
      { re: /@[A-Za-z_][A-Za-z0-9_.]*/y, cls: "t-def" },
      PY_DEF,
      NUMBER,
      ident(PY_KW, PY_LIT),
      PLAIN
    ],
    cpp: [
      { re: /\/\/[^\n]*/y, cls: "t-com" },
      { re: /\/\*[\s\S]*?\*\//y, cls: "t-com" },
      { re: /#[ \t]*[a-z_]+/y, cls: "t-def" },
      { re: /"(?:\\[\s\S]|[^"\\\n])*"/y, cls: "t-str" },
      { re: /'(?:\\[\s\S]|[^'\\\n])*'/y, cls: "t-str" },
      NUMBER,
      ident(C_KW, words("")),
      PLAIN
    ]
  };
  /* HIP is C++ with a different set of intrinsics, and CUDA appears in the
     upstream references. Same rules; naming them separately means a pane can
     say what it is without the tokenizer pretending they are different. */
  LANGS.hip = LANGS.cpp;
  LANGS.c = LANGS.cpp;
  LANGS.cuda = LANGS.cpp;

  function tokenize(src, rules) {
    var out = "", i = 0, n = src.length;
    while (i < n) {
      var hit = false;
      for (var r = 0; r < rules.length; r++) {
        var rule = rules[r];
        rule.re.lastIndex = i;
        var m = rule.re.exec(src);
        if (!m || !m[0].length) continue;
        if (rule.render) {
          out += rule.render(m);
        } else {
          var cls = typeof rule.cls === "function" ? rule.cls(m[0]) : rule.cls;
          out += cls ? '<span class="' + cls + '">' + esc(m[0]) + "</span>"
                     : esc(m[0]);
        }
        i += m[0].length;
        hit = true;
        break;
      }
      /* No rule applied: emit the character and move on. This is the branch
         that makes the tokenizer total -- there is no input it can refuse, and
         nothing it cannot classify gets dropped. */
      if (!hit) { out += esc(src.charAt(i)); i++; }
    }
    return out;
  }

  /* The key the reader actually has. The fallback message said "press ⌘C"
     unconditionally, and this board is served from a Linux node to Linux and
     Windows browsers far more often than to a Mac -- so the one message that
     appears when copying has FAILED was naming a key most readers do not have.
     Sniffed once, from the platform string, defaulting to the majority case. */
  var COPY_KEY = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
    ? "⌘C" : "Ctrl+C";

  function copyText(text, btn) {
    var done = function (ok) {
      btn.textContent = ok ? "copied" : "press " + COPY_KEY;
      btn.classList.toggle("ok", ok);
      setTimeout(function () {
        btn.textContent = "copy";
        btn.classList.remove("ok");
      }, 1400);
    };
    /* navigator.clipboard needs a secure context. This board is served over
       plain HTTP to other machines on the node's network as often as it is
       served to localhost, so the textarea path is the normal case there, not
       a legacy fallback. */
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { done(true); },
                                                function () { done(false); });
      return;
    }
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;top:0;left:-9999px";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    done(ok);
  }

  function enhance(pre) {
    if (pre.getAttribute("data-hl")) return;
    pre.setAttribute("data-hl", "1");

    /* Before anything is inserted. This is the source, and it is the only
       thing the copy button is ever allowed to hand over. */
    var raw = pre.textContent;

    /* A trailing newline ends the last line, it does not begin another. The
       gutter used to drop it and the body used to keep it, which is the same
       off-by-one twice, in opposite directions: a numberless blank line at the
       foot of every pane in the corpus, since every source but 26 of 1,660
       ends in \n. Strip it once, here, and both sides are built from the same
       string. `raw` keeps its newline -- that is the file, and it is what the
       copy button hands over; a source that arrives on the clipboard without
       its final newline is a diff nobody asked for. */
    var shown = raw.replace(/\n$/, "");

    var lang = (pre.getAttribute("data-lang") || "text").toLowerCase();
    var rules = LANGS[lang] || null;
    var body = rules ? tokenize(shown, rules) : esc(shown);

    var lines = shown.split("\n");
    var gutter = lines.map(function (_, i) { return i + 1; }).join("\n");

    pre.classList.add("has-ln");
    pre.innerHTML = '<span class="gutter" aria-hidden="true">' + gutter +
                    '</span><code class="src">' + body + "</code>";

    var wrap = document.createElement("div");
    wrap.className = "codewrap";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "copy";
    btn.setAttribute("aria-label", "Copy this source to the clipboard");
    btn.addEventListener("click", function () { copyText(raw, btn); });
    wrap.appendChild(btn);
  }

  function init() {
    /* Opt-in by `data-lang`: a pane that has not declared what it holds is
       left exactly as the server rendered it. */
    document.querySelectorAll("pre.code[data-lang]").forEach(enhance);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
