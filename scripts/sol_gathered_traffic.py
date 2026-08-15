#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D18 at the tier: price a gathered allocation at what the workload names.

The declared-traffic tier (`scripts/sol_traffic_floor.py`) prices every
declared input at its full allocation. That is the right price for a tensor the
kernel streams and the wrong one for a tensor the kernel *gathers*.

A paged KV cache is declared `[num_pages, page_size, num_kv_heads, head_dim]`
with `num_pages` in the hundreds of thousands, and the workload names a handful
of pages in `kv_indices`. On the worst workload of
`FlashInfer-Bench__015_gqa_paged_prefill_causal_h32_kv8_d128_ps1`
(`934884ed`, `num_pages=552310`, `num_kv_indices=2`) the tier charges

    k_cache + v_cache = 2 x 552,310 x 1 x 8 x 128 x 2 B = 2,262,241,280 B
    everything else (q, output, lse, indptrs, indices)  =       301,208 B
    total 2,262,542,488 B / 7.99992e12 B/s               = 0.282821 ms

against a manifest-v2 published T_SOL of **0.282820964 ms** -- the bound IS the
whole-allocation streaming time, to nine significant figures. A real Triton
kernel ran it in 0.00664 ms, so the "lower bound" is 43x above a measured time.
Priced at the two pages the workload names, the same formula gives 288,920 B
and 3.61e-5 ms, which the measurement clears by 184x.

**The pairing is derived, not tabulated.** `gathered_axes` reads the problem's
own reference and finds gathers `T[i]` where `T` comes from an input whose
leading dimension is an axis and `i` comes from a 1-D integer input. Six
problems in the dataset declare that shape today (`FlashInfer-Bench__012`,
`013`, `014`, `015`, `018`, `019`); nothing names them here, and a seventh
would be picked up without an edit.

**Both directions of error are bounded, and they are not symmetric.** A pairing
that exists and is not found leaves the allocation price in place -- today's
behaviour, no regression. A pairing found wrongly prices a streamed tensor at
an index count, which makes the bound SMALLER: a lower bound that is too small
is loose, and loose is the safe direction, because no kernel can beat it and no
score it produces can exceed 1. The direction this removes -- a bound too large
-- is the one that puts scores above 1 and is not a bound at all.

---

**D18 in a second guise: a masked STREAM, not a gathered ALLOCATION.**

D18 was an allocation the workload only names part of. The same tier makes the
same class of error one step further out: it charges a full read of a tensor the
kernel *streams* over rows the problem's own causal mask makes dead.

`FlashInfer-Bench__014/__015` declare `q` as `[total_q, 32, 128]` bf16 and the
reference (`reference.py:64-67`) does

    delta = num_kv_tokens - num_q_tokens
    for q_idx in range(num_q_tokens):
        max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
        if max_kv_idx <= 0:
            continue                # output row stays 0, lse row stays -inf

over an `output` pre-filled with `torch.zeros` and an `lse` pre-filled with
`-inf`. For every query row whose causal window is empty the correct answer is
`(0, -inf)` *independent of `q`*, so a correct kernel need never read that row.
Per sequence exactly `min(q_len, kv_len)` rows survive. Measured from the
workloads' own `qo_indptr`/`kv_indptr`: **25 of 15783, 2 of 16384, 3 of 16384
and 1 of 10447** live rows on the four workloads real kernels falsified.

`causal_masked_axis` derives that shape from the reference, `masked_live_rows`
counts the live rows from the workload's own index vectors, and
`gathered_traffic(..., live=...)` prices the stream at them. Ratio of declared
to corrected over the 68 workloads of the two problems: min 1.0000, median
1.9584, mean 1.8357, max 1.9844 -- ~2x for the trivial reason that `q` and
`output` are the same shape and dtype, so charging a dead `q` doubles the
bound. **64 of the 68 move; 4 do not** -- the degenerate `total_q` in {1, 2}
workloads, where every row is live and the ratio is exactly 1.0000. All 64
move DOWN, none up, and measured against every submission on those two
problems the corrected bound leaves **0 of 68** residual violations where the
declared one left 4.

**The OUTPUT and the `lse` keep their full price.** A correct kernel really does
have to write `(0, -inf)` into every dead row; only the read side is dead. That
conservative reading leaves the four falsifying kernels at an implied 4.16-5.09
TB/s, against a measured achievable HBM figure of 4.89 TB/s and a measured
~7.3 TB/s ceiling at their working-set size -- physically plausible, where the
declared model demanded 8.24-10.10 TB/s, which a GPU probe has measured to be
unreachable at that working-set size. The stricter reading that also discounts
the dead output rows (125x, not 2x) is NOT taken.

**The detector is narrow on purpose, and it is not a name list.** It fires only
on the exact empty-window skip above, which pins the closed form
`min(q_len, kv_len)`; anything else falls through to today's price. That matters
for the neighbouring problems: `FlashInfer-Bench__016/__017` mask by
`masked_fill_` and produce NaN rather than skipping, and `__019` computes every
row, so none of them may take this rule even if a future trace gave them dead
rows. They do not match the pattern and so they do not get it.

**The cost, stated honestly.** Unlike D18's pairing -- a declared *axis* --
this rule needs the *contents* of `qo_indptr`/`kv_indptr` out of the workload's
safetensors blob at derivation time, which this tier did not read before.
Measured: 131 ms for `causal_masked_axis` over all 235 references (it parses
each one a second time, alongside `gathered_axes`) and 37 ms to read the 136
blob+key pairs the 68 workloads name. End to end that takes the MI355X tier
build from 1.80-1.85 s to 1.91-2.07 s over three runs each -- ~8%, on a build
that is already two seconds. It still needs no GPU: `safetensors.numpy` is
used precisely so no torch import and no device ever enters a bound
derivation. Where the blob is absent or does not validate, no correction is
applied and the workload is counted in `workloads_masked_rows_unresolved`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

#: Integer dtypes an index vector may be declared with, from the dataset's own
#: dtype strings. Deliberately not floats: a float 1-D input is not an index.
INT_DTYPES = ("int64", "int32", "int16", "int8", "uint8")


#: Attributes that return METADATA about a tensor, not its contents. Following
#: them made `device = q.device` give `output = torch.zeros(..., device=device)`
#: the origin `q`, and `output[global_q_idx]` then looked like a gather of `q`.
METADATA_ATTRS = frozenset({"device", "dtype", "shape", "ndim", "size",
                            "numel", "stride", "element_size", "layout",
                            "requires_grad", "is_cuda"})

#: Calls that SCALARISE: they turn tensor content into one Python number. A
#: scalar cannot be the index vector of a gather -- `output[q_start + q_idx]`
#: with `q_start = int(qo_indptr[b].item())` writes ONE row, and pricing
#: `total_q` at `len_indptr` because of it is exactly the wrong direction.
SCALARISING_CALLS = frozenset({"item", "tolist"})
SCALARISING_BUILTINS = frozenset({"int", "float", "bool", "len", "range"})


def _content_names(node: ast.AST) -> set[str]:
    """Names whose VALUE flows into `node`, ignoring subscript keys.

    The distinction is the whole reason the rule works. In
    `page_ids = kv_indices[kv_start:kv_end]` the *content* of `page_ids` comes
    from `kv_indices` alone; `kv_start` and `kv_end` say which part, not what.
    Walking the subscript key as if it were content made `page_ids` derive from
    both `kv_indices` and `kv_indptr`, two different index vectors, so every
    gather in the paged references came out ambiguous and the rule fired on
    nothing but false positives.
    """
    out: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Name):
            out.add(cur.id)
            continue
        if isinstance(cur, ast.Subscript):
            stack.append(cur.value)          # the key is not content
            continue
        if isinstance(cur, ast.Attribute) and cur.attr in METADATA_ATTRS:
            continue                          # metadata, not content
        if isinstance(cur, ast.Call):
            f = cur.func
            if isinstance(f, ast.Attribute) and f.attr in SCALARISING_CALLS:
                continue
            if isinstance(f, ast.Name) and f.id in SCALARISING_BUILTINS:
                continue
        stack.extend(ast.iter_child_nodes(cur))
    return out


def _scalar_names(node: ast.AST) -> set[str]:
    """`_content_names`, but following the scalarising calls it stops at.

    `q_start = int(qo_indptr[b].item())` is exactly what `_content_names` is
    built to cut: a scalar cannot be the index vector of a gather. But it is
    precisely how a reference says *which* index vector segments a stream, so
    the causal-mask derivation needs the other answer. Two namers, two
    questions; `gathered_axes` must keep using `_content_names`.
    """
    out: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Name):
            out.add(cur.id)
            continue
        if isinstance(cur, ast.Subscript):
            stack.append(cur.value)          # the key is not content
            continue
        if isinstance(cur, ast.Attribute) and cur.attr in METADATA_ATTRS:
            continue                          # metadata, not content
        stack.extend(ast.iter_child_nodes(cur))
    return out


def _origins(tree: ast.AST, params: set[str],
             names: Callable[[ast.AST], set[str]] = _content_names,
             ) -> dict[str, set[str]]:
    """{local name -> the input parameters its value may derive from}.

    A deliberately shallow data flow. Seed every parameter with itself, then
    propagate through simple `name = <expr>` assignments over `_content_names`,
    to a fixpoint. Loops, branches and method calls need no special handling:
    the question is only "could this value have come from that input", and a
    superset answer is safe -- an origin set that is too large can only produce
    an ambiguous pairing, and `gathered_axes` discards ambiguous pairings.

    `page_ids = kv_indices[kv_start:kv_end].to(torch.long)` gives
    `page_ids -> {kv_indices}`, and `k_cache_flat = k_cache.squeeze(1).to(...)`
    gives `k_cache_flat -> {k_cache}`, which is what the gather needs.
    """
    origins: dict[str, set[str]] = {p: {p} for p in params}
    for _ in range(8):
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            src: set[str] = set()
            for name in names(node.value):
                src |= origins.get(name, set())
            if src - origins.get(target.id, set()):
                origins.setdefault(target.id, set()).update(src)
                changed = True
        if not changed:
            break
    return origins


def _lead_axis(definition: dict) -> dict[str, str]:
    """{input name -> the axis its FIRST declared dimension is}, rows-first."""
    axes = definition.get("axes") or {}
    return {name: spec["shape"][0]
            for name, spec in (definition.get("inputs") or {}).items()
            if spec.get("shape") and isinstance(spec["shape"][0], str)
            and spec["shape"][0] in axes}


def _index_axis(definition: dict) -> dict[str, str]:
    """{index-vector input -> its axis}: 1-D, integer dtype, axis-dimensioned."""
    axes = definition.get("axes") or {}
    return {name: spec["shape"][0]
            for name, spec in (definition.get("inputs") or {}).items()
            if spec.get("dtype") in INT_DTYPES
            and spec.get("shape") and len(spec["shape"]) == 1
            and isinstance(spec["shape"][0], str)
            and spec["shape"][0] in axes}


def gathered_axes(definition: dict) -> dict[str, str]:
    """{allocation axis -> the axis counting how many rows are named}.

    An axis `A` is priced at axis `G` when the reference contains a subscript
    `T[i]` for which

      * `T` derives from a declared input tensor whose FIRST declared
        dimension is the axis `A`;
      * `i` derives from a declared input tensor of integer dtype whose shape
        is exactly `[G]` -- an index vector, so `G` is the most rows of `A`
        the workload can name;
      * the subscript is an index, not a slice, and it is the first subscript
        position -- a gather of whole rows.

    The slice exclusion is load-bearing. `q_f32[q_start:q_end]` reads a
    contiguous run whose bounds come from `qo_indptr`, a 1-D int32 input; that
    is a slice, not a gather, and pricing `total_q` at `len_indptr` would be
    nonsense. Only `k_cache_flat[page_ids]` and `v_cache_flat[page_ids]`
    survive on the paged problems.

    A subscript that resolves to more than one candidate allocation axis or
    more than one candidate index vector is ambiguous and is dropped, as is an
    axis two different subscripts would price at two different counts.
    """
    src = definition.get("reference")
    if not src:
        return {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}

    inputs = definition.get("inputs") or {}
    lead_axis = _lead_axis(definition)
    index_axis = _index_axis(definition)
    if not lead_axis or not index_axis:
        return {}

    origins = _origins(tree, set(inputs))
    out: dict[str, str] = {}
    ambiguous: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if isinstance(key, ast.Tuple):
            key = key.elts[0] if key.elts else None
        if key is None or isinstance(key, ast.Slice):
            continue
        base: set[str] = set()
        for name in _content_names(node.value):
            base |= origins.get(name, set())
        idx: set[str] = set()
        for name in _content_names(key):
            idx |= origins.get(name, set())
        alloc = {lead_axis[b] for b in base if b in lead_axis}
        gath = {index_axis[i] for i in idx if i in index_axis}
        if len(alloc) != 1 or len(gath) != 1:
            continue
        a, g = alloc.pop(), gath.pop()
        if a == g:
            continue
        if out.get(a, g) != g:
            ambiguous.add(a)
        out[a] = g
    for a in ambiguous:
        out.pop(a, None)
    return out


def _add_terms(node: ast.AST) -> list[ast.AST]:
    """Flatten an `a + b + c` chain, so the match is on terms not on shape."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _add_terms(node.left) + _add_terms(node.right)
    return [node]


def _window_offset(node: ast.AST, loop_var: str) -> str | None:
    """`loop_var + 1 + delta` -> `"delta"`. Anything else -> None."""
    terms = _add_terms(node)
    if len(terms) != 3:
        return None
    ones = [t for t in terms
            if isinstance(t, ast.Constant) and t.value == 1]
    var = [t for t in terms
           if isinstance(t, ast.Name) and t.id == loop_var]
    rest = [t for t in terms
            if isinstance(t, ast.Name) and t.id != loop_var]
    if len(ones) != 1 or len(var) != 1 or len(rest) != 1:
        return None
    return rest[0].id


def _row_extent_bases(name: str, assigns: dict[str, list[ast.AST]],
                      origins: dict[str, set[str]]) -> set[str]:
    """Inputs whose row count `name` is, via `X.shape[0]` or `len(X)`."""
    out: set[str] = set()
    for value in assigns.get(name, []):
        base: ast.AST | None = None
        if (isinstance(value, ast.Subscript)
                and isinstance(value.slice, ast.Constant)
                and value.slice.value == 0
                and isinstance(value.value, ast.Attribute)
                and value.value.attr == "shape"):
            base = value.value.value
        elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
              and value.func.id == "len" and len(value.args) == 1):
            base = value.args[0]
        if base is None:
            continue
        for n in _content_names(base):
            out |= origins.get(n, set())
    return out


def _segmenting_indptr(tree: ast.AST, content: dict[str, set[str]],
                       scalar: dict[str, set[str]],
                       index_axis: dict[str, str]) -> dict[str, set[str]]:
    """{input -> the index vectors that supply a SLICE of its rows}.

    `q_batch = q_f32[q_start:q_end]` with `q_start = int(qo_indptr[b].item())`
    is how a reference says "`qo_indptr` cuts `q` into sequences". The bounds
    are scalars, so this is the one place `_scalar_names` is needed: through
    `int(...)` and `.item()`, which `_content_names` deliberately stops at.
    """
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if isinstance(key, ast.Tuple):
            key = key.elts[0] if key.elts else None
        if not isinstance(key, ast.Slice):
            continue
        base: set[str] = set()
        for n in _content_names(node.value):
            base |= content.get(n, set())
        bounds: set[str] = set()
        for part in (key.lower, key.upper):
            if part is None:
                continue
            for n in _scalar_names(part):
                bounds |= scalar.get(n, set())
        vectors = {b for b in bounds if b in index_axis}
        if len(base) != 1 or len(vectors) != 1:
            continue
        out.setdefault(base.pop(), set()).update(vectors)
    return out


def causal_masked_axis(definition: dict) -> dict | None:
    """The stream axis whose rows an empty causal window makes dead, or None.

    Fires only on the reference shape D18's second guise is defined by:

        for <row> in range(<q_count>):
            <m> = min(<row> + 1 + <delta>, <kv_count>)
            if <m> <= 0:
                continue

    with `<delta> = <kv_count> - <q_count>`, `<q_count>` the row count of a
    declared multi-row input sliced by one index vector, and `<kv_count>` the
    row count of a declared index vector sliced by another. Those five facts
    together pin the closed form: the live rows of a sequence are exactly
    `min(q_len, kv_len)`, so no symbolic evaluation of the mask is needed and
    none is attempted.

    Every clause is load-bearing as an EXCLUSION, which is the point --
    `FlashInfer-Bench__016/__017` mask with `masked_fill_` and emit NaN on an
    empty window (no skip, so a dead row is not free), and `__019` has a
    per-row loop with no skip in it at all. Neither matches, so neither is
    repriced, and a future reference that rewrites the arithmetic falls back to
    today's price rather than to a closed form that no longer holds.

    Returns `{"axis", "stream", "stream_indptr", "window_axis", "window",
    "window_indptr"}`, or None where nothing matches or more than one distinct
    match does.
    """
    src = definition.get("reference")
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    inputs = definition.get("inputs") or {}
    lead_axis = _lead_axis(definition)
    index_axis = _index_axis(definition)
    if not lead_axis or not index_axis:
        return None

    content = _origins(tree, set(inputs))
    scalar = _origins(tree, set(inputs), _scalar_names)
    segment = _segmenting_indptr(tree, content, scalar, index_axis)

    assigns: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            assigns.setdefault(node.targets[0].id, []).append(node.value)

    found: list[dict] = []
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
            continue
        it = loop.iter
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "range" and len(it.args) == 1
                and isinstance(it.args[0], ast.Name)):
            continue
        row, q_count = loop.target.id, it.args[0].id
        for node in ast.walk(loop):
            if not isinstance(node, ast.If):
                continue
            if [type(s) for s in node.body] != [ast.Continue]:
                continue
            test = node.test
            if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and len(test.ops) == 1 and isinstance(test.ops[0], ast.LtE)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == 0):
                continue
            for value in assigns.get(test.left.id, []):
                spec = _match_window(value, row, q_count, assigns, lead_axis,
                                     index_axis, segment, content)
                if spec is not None and spec not in found:
                    found.append(spec)
    return found[0] if len(found) == 1 else None


def _match_window(value: ast.AST, row: str, q_count: str,
                  assigns: dict[str, list[ast.AST]],
                  lead_axis: dict[str, str], index_axis: dict[str, str],
                  segment: dict[str, set[str]],
                  origins: dict[str, set[str]]) -> dict | None:
    """`min(row + 1 + delta, kv_count)` resolved to declared inputs, or None."""
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "min" and len(value.args) == 2):
        return None
    for a, b in ((0, 1), (1, 0)):
        window, offset = value.args[a], value.args[b]
        if not isinstance(window, ast.Name):
            continue
        kv_count = window.id
        delta = _window_offset(offset, row)
        if delta is None:
            continue
        # delta must be exactly `kv_count - q_count`; a different alignment is
        # a different mask and would not have `min(q_len, kv_len)` live rows.
        if not any(isinstance(d, ast.BinOp) and isinstance(d.op, ast.Sub)
                   and isinstance(d.left, ast.Name) and d.left.id == kv_count
                   and isinstance(d.right, ast.Name) and d.right.id == q_count
                   for d in assigns.get(delta, [])):
            continue
        streams = {s for s in _row_extent_bases(q_count, assigns, origins)
                   if s in lead_axis and s not in index_axis}
        windows = {w for w in _row_extent_bases(kv_count, assigns, origins)
                   if w in index_axis}
        if len(streams) != 1 or len(windows) != 1:
            continue
        stream, window_vec = streams.pop(), windows.pop()
        s_ptr = segment.get(stream, set())
        w_ptr = segment.get(window_vec, set())
        if len(s_ptr) != 1 or len(w_ptr) != 1:
            continue
        stream_indptr, window_indptr = s_ptr.copy().pop(), w_ptr.copy().pop()
        # The two indptrs must be different vectors over the same batch axis:
        # one cut, two extents, `b` indexing both.
        if (stream_indptr == window_indptr
                or index_axis[stream_indptr] != index_axis[window_indptr]):
            continue
        if lead_axis[stream] == index_axis[window_vec]:
            continue
        return {"axis": lead_axis[stream], "stream": stream,
                "stream_indptr": stream_indptr,
                "window_axis": index_axis[window_vec], "window": window_vec,
                "window_indptr": window_indptr}
    return None


#: {(resolved path, tensor key) -> the vector}, so a blob shared by every
#: workload of a problem is read once per derivation, not once per workload.
_INDEX_CACHE: dict[tuple[str, str], list[int] | None] = {}


def _index_vector(workload_inputs: dict, name: str,
                  root: Path | None) -> list[int] | None:
    """One declared index vector's CONTENTS, from the workload's own blob.

    This is the cost D18's fix did not have. Returns None -- never a guess --
    when the workload does not name a safetensors tensor for it, when the file
    is absent (the FlashInfer trace blobs are fetched separately, see
    `scripts/fetch_flashinfer_traces.py`), or when safetensors is not
    importable. A missing blob leaves the uncorrected price in place.
    """
    spec = (workload_inputs or {}).get(name)
    if not isinstance(spec, dict) or spec.get("type") != "safetensors":
        return None
    path, key = spec.get("path"), spec.get("tensor_key")
    if not path or not key:
        return None
    resolved = Path(path)
    if root is not None and not resolved.is_absolute():
        resolved = Path(root) / path
    ck = (str(resolved), key)
    if ck in _INDEX_CACHE:
        return _INDEX_CACHE[ck]
    out: list[int] | None = None
    try:
        from safetensors.numpy import load_file      # no torch, no device
        tensors = load_file(str(resolved))
        arr = tensors.get(key)
        if arr is not None and arr.ndim == 1:
            out = [int(v) for v in arr.tolist()]
    except Exception:
        out = None
    _INDEX_CACHE[ck] = out
    return out


def masked_live_rows(spec: dict, axes: dict, workload_inputs: dict,
                     root: Path | None = None) -> int | None:
    """Rows of the masked axis a correct kernel must READ, or None.

    `sum_b min(q_len_b, kv_len_b)` over the workload's own index vectors --
    exactly the rows the reference's `if max_kv_idx <= 0: continue` leaves
    alive. Validated against the declared axes before it is believed: the two
    vectors must be the same length, non-decreasing, and end at the axes the
    definition says they count (`qo_indptr[-1] == total_q`, which the reference
    itself asserts, and `kv_indptr[-1] == num_kv_indices`). Any of those
    failing means the blob is not what the definition describes, and the answer
    is None -- no correction -- rather than a count from a file that does not
    match.
    """
    q = _index_vector(workload_inputs, spec["stream_indptr"], root)
    w = _index_vector(workload_inputs, spec["window_indptr"], root)
    if q is None or w is None or len(q) != len(w) or len(q) < 2:
        return None
    if q[-1] != axes.get(spec["axis"]) or w[-1] != axes.get(spec["window_axis"]):
        return None
    live = 0
    for b in range(len(q) - 1):
        dq, dw = q[b + 1] - q[b], w[b + 1] - w[b]
        if dq < 0 or dw < 0:
            return None
        live += min(dq, dw)
    return live


def gathered_traffic(definition: dict, axes: dict,
                     rewrite: dict[str, str] | None = None,
                     live: dict[str, int] | None = None) -> int | None:
    """`declared_traffic`, with a gathered allocation priced at what it names.

    Identical to `sol_cross_checks.declared_traffic` -- same dtype table, same
    "scalars ride in a kernel argument", same "return None rather than make a
    partial claim" on an unresolved symbol or unknown dtype -- except that a
    dimension `gathered_axes` identified is replaced by the axis counting the
    rows the workload names.

    `live` carries the second correction: `{axis -> live row count}` from
    `masked_live_rows`, applied to the LEADING dimension of an input only.
    Leading, because the mask kills whole rows; input, because the write side
    of a masked row is not free -- a correct kernel still has to put `(0,
    -inf)` there, and that is why the output and the `lse` keep their full
    price here exactly as they do under a gather.

    Returns None, not a partial count, if the gathered axis is not resolvable
    for this workload: pricing half the correction in would be neither the old
    number nor the new one.
    """
    from sol_cross_checks import DTYPE_BYTES

    if rewrite is None:
        rewrite = gathered_axes(definition)
    live = live or {}
    total = 0
    for group in ("inputs", "outputs"):
        # OUTPUTS ARE NEVER REWRITTEN. A gather is a claim about reading: the
        # kernel need only read the rows the index names. The matching write is
        # a SCATTER into a tensor that still has to exist in full --
        # `L1__009`'s `grad_hidden_states` is `[batch_seq_len, hidden_dim]`,
        # zero everywhere the scatter does not land, so every row of it is
        # written. Pricing an output at the index count would under-count real
        # traffic for no reason; only the read side of the pair is corrected.
        gathered = rewrite if group == "inputs" else {}
        masked = live if group == "inputs" else {}
        for spec in (definition.get(group) or {}).values():
            shape = spec.get("shape")
            if not shape:
                continue
            n = 1
            for pos, dim in enumerate(shape):
                if pos == 0 and dim in masked and dim not in gathered:
                    # Capped at the allocation for the same reason the gather
                    # is: a correction may never raise a bound.
                    m = masked[dim]
                    if dim in axes:
                        m = min(m, axes[dim])
                    n *= m
                elif dim in gathered:
                    if rewrite[dim] not in axes:
                        return None
                    # Capped at the allocation. An index vector may name a slot
                    # twice -- `FlashInfer-Bench__015`/`75ab4c21` has
                    # `num_kv_indices = 28` against `num_pages = 2`, one
                    # distinct page repeated -- and a kernel never has to read
                    # more distinct rows than exist. Without the cap the
                    # "correction" raised that bound by 106,496 B, which is the
                    # one direction this whole change exists to remove.
                    m = axes[rewrite[dim]]
                    if dim in axes:
                        m = min(m, axes[dim])
                    n *= m
                elif isinstance(dim, int):
                    n *= dim
                elif dim in axes:
                    n *= axes[dim]
                elif str(dim).isdigit():
                    n *= int(dim)
                else:
                    return None
            width = DTYPE_BYTES.get(spec.get("dtype"))
            if width is None:
                return None
            total += n * width
    return total
