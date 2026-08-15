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
"""

from __future__ import annotations

import ast

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


def _origins(tree: ast.AST, params: set[str]) -> dict[str, set[str]]:
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
            for name in _content_names(node.value):
                src |= origins.get(name, set())
            if src - origins.get(target.id, set()):
                origins.setdefault(target.id, set()).update(src)
                changed = True
        if not changed:
            break
    return origins


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
    axes = definition.get("axes") or {}
    lead_axis = {name: spec["shape"][0]
                 for name, spec in inputs.items()
                 if spec.get("shape") and isinstance(spec["shape"][0], str)
                 and spec["shape"][0] in axes}
    index_axis = {name: spec["shape"][0]
                  for name, spec in inputs.items()
                  if spec.get("dtype") in INT_DTYPES
                  and spec.get("shape") and len(spec["shape"]) == 1
                  and isinstance(spec["shape"][0], str)
                  and spec["shape"][0] in axes}
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


def gathered_traffic(definition: dict, axes: dict,
                     rewrite: dict[str, str] | None = None) -> int | None:
    """`declared_traffic`, with a gathered allocation priced at what it names.

    Identical to `sol_cross_checks.declared_traffic` -- same dtype table, same
    "scalars ride in a kernel argument", same "return None rather than make a
    partial claim" on an unresolved symbol or unknown dtype -- except that a
    dimension `gathered_axes` identified is replaced by the axis counting the
    rows the workload names.

    Returns None, not a partial count, if the gathered axis is not resolvable
    for this workload: pricing half the correction in would be neither the old
    number nor the new one.
    """
    from sol_cross_checks import DTYPE_BYTES

    if rewrite is None:
        rewrite = gathered_axes(definition)
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
        for spec in (definition.get(group) or {}).values():
            shape = spec.get("shape")
            if not shape:
                continue
            n = 1
            for dim in shape:
                if dim in gathered:
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
