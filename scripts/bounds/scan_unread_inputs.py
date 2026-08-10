#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Which references declare an input they never read? (D42, cause A)

    python scripts/bounds/scan_unread_inputs.py --out artifacts/11/unread-inputs.json

`L1__042`'s bound is 65/32 of the traffic a correct kernel must move, and the
extra 33/32 is two declared inputs that `run()` never touches -- it uses
`topk_idx.shape[0]` and never reads `expert_mask` at all. The declared-traffic
tier prices both at full size.

That is mechanically detectable, so this asks how many problems have it. Three
classes, and they are NOT the same defect:

  never_read     the parameter name never appears in the body. Its bytes are
                 pure over-count.
  metadata_only  the name appears only under `.shape` / `.size()` / `len()` /
                 `.dtype` / `.device` / `.ndim`. A kernel reads no elements of
                 it, so its bytes are over-count too -- this is `topk_idx`.
  indexed_only   the name appears only as the *base* of a subscript or a gather.
                 NOT reported as a defect: a slice still moves bytes, just fewer
                 than the whole tensor, and how many depends on the workload.
                 Counted separately because it is where `L1__018` and `L1__057`
                 live and it needs a per-problem derivation, not a scan.

**Scoped to the body of `run()`, deliberately.** The D37 scoping error came from
an AST scan over the whole reference file, which counted `get_inputs` and the
`__main__` block and reported seven grouped-convolution problems where there
are six. A name used only to *construct* an input is not a name the timed
function reads.

This sizes the work. It does not correct a bound: a parameter that is never
read tells you a term should not be in the sum, not what the sum should be.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "SOL-ExecBench" / "benchmark"

#: Attributes that read a tensor's metadata rather than its elements. A kernel
#: that touches only these moves no bytes of that tensor.
META_ATTRS = {"shape", "size", "dtype", "device", "ndim", "numel", "is_cuda",
              "requires_grad", "stride", "dim", "element_size", "layout"}


def tensor_params(fn: ast.FunctionDef) -> set[str]:
    """Parameters annotated as tensors.

    A scalar that is never read costs nothing -- `rms_norm_eps: float` moves no
    bytes whether the kernel reads it or not -- so leaving scalars in the
    finding list would pad it with entries that cannot change a bound. Every
    reference in this benchmark annotates its signature; a parameter with no
    annotation is kept rather than dropped, because an unannotated tensor is
    the case that matters and guessing it away would be the wrong direction to
    err in.
    """
    out = set()
    for arg in list(fn.args.args) + list(fn.args.kwonlyargs):
        ann = arg.annotation
        if ann is None:
            out.add(arg.arg)
            continue
        txt = ast.unparse(ann)
        if "Tensor" in txt:
            out.add(arg.arg)
    return out


def classify(fn: ast.FunctionDef) -> dict[str, str]:
    """param -> never_read | metadata_only | indexed_only | read."""
    params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    verdict = {p: "never_read" for p in params}

    def worse(cur: str, new: str) -> str:
        order = ["never_read", "metadata_only", "indexed_only", "read"]
        return new if order.index(new) > order.index(cur) else cur

    class V(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            # `x.shape`, `x.size()`, `len(x)` handled below -- metadata only.
            if (isinstance(node.value, ast.Name) and node.value.id in verdict
                    and node.attr in META_ATTRS):
                verdict[node.value.id] = worse(verdict[node.value.id],
                                               "metadata_only")
                return                      # do NOT descend into node.value
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (isinstance(node.func, ast.Name) and node.func.id == "len"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in verdict):
                p = node.args[0].id
                verdict[p] = worse(verdict[p], "metadata_only")
                for kw in node.keywords:
                    self.visit(kw)
                return
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id in verdict:
                verdict[node.value.id] = worse(verdict[node.value.id],
                                               "indexed_only")
                self.visit(node.slice)      # the INDEX is a full read of itself
                return
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id in verdict:
                verdict[node.id] = worse(verdict[node.id], "read")

    for stmt in fn.body:
        V().visit(stmt)
    return verdict


def run_fn(src: str) -> ast.FunctionDef | None:
    """The `run` at module level. Not any function called `run` anywhere."""
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == "run":
            return node                                      # type: ignore[return-value]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/09/manifest-v1.2.json")
    ap.add_argument("--out", default="artifacts/11/unread-inputs.json")
    a = ap.parse_args()

    man = json.loads((ROOT / a.manifest).read_text())["problems"]

    findings = []
    scanned = unparsed = 0
    for key, prob in sorted(man.items()):
        cat, name = key.split("__", 1)
        ref = BENCH / cat / name / "reference.py"
        if not ref.exists():
            continue
        fn = run_fn(ref.read_text())
        if fn is None:
            unparsed += 1
            continue
        scanned += 1
        v = classify(fn)
        tensors = tensor_params(fn)
        dead = sorted(p for p, k in v.items()
                      if k in ("never_read", "metadata_only") and p in tensors)
        if not dead:
            continue

        # Does this problem's bound actually rest on the traffic tier? A dead
        # input on a problem SOLAR bounds is a curiosity; on one the traffic
        # tier bounds, it is bytes in a published number.
        wl = prob["workloads"]
        on_traffic = sum(
            1 for w in wl.values()
            if w.get("t_sol_cycles_traffic") and w.get("t_sol_cycles_solar")
            and w["t_sol_cycles_traffic"] > w["t_sol_cycles_solar"])
        findings.append({
            "problem": key,
            "dead_inputs": dead,
            "classes": {p: v[p] for p in dead},
            "indexed_only": sorted(p for p, k in v.items()
                                   if k == "indexed_only" and p in tensors),
            "scalars_ignored": sorted(
                p for p, k in v.items()
                if k in ("never_read", "metadata_only") and p not in tensors),
            "workloads_bounded_by_traffic_tier": on_traffic,
            "of_workloads": len(wl),
            "affects_a_published_bound": on_traffic > 0,
        })

    live = [f for f in findings if f["affects_a_published_bound"]]
    payload = {
        "question": "L1__042's bound over-counts by exactly two declared "
                    "inputs that run() never reads. How many problems have "
                    "that shape, and how many of those have a bound that "
                    "actually rests on the declared-traffic tier?",
        "scope": "the body of the module-level `run` only -- not get_inputs, "
                 "not __main__. The D37 scoping error came from scanning the "
                 "whole file and reported seven where there are six.",
        "problems_scanned": scanned,
        "problems_with_no_module_level_run": unparsed,
        "problems_with_a_dead_input": len(findings),
        "of_those_bounded_by_the_traffic_tier": len(live),
        "caveat": "`indexed_only` is reported per problem but NOT counted as a "
                  "defect. A slice moves bytes -- fewer than the whole tensor, "
                  "and how many depends on the workload. L1__018 and L1__057 "
                  "are that class and each needed its own derivation.",
        "does_not_correct_anything": "A parameter that is never read says a "
                                     "term should not be in the sum. It does "
                                     "not say what the sum should be.",
        "affecting_a_published_bound": live,
        "all_findings": findings,
    }

    sys.path.insert(0, str(ROOT / "scripts"))
    from provenance import write_artifact                     # noqa: E402
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(out, "11-unread-inputs", payload)
    print(f"wrote {out}")
    print(f"scanned {scanned} problems, {unparsed} with no module-level run()")
    print(f"{len(findings)} declare an input run() never reads; "
          f"{len(live)} of those are bounded by the traffic tier")
    for f in live:
        print(f"  {f['problem'][:58]:58} {f['workloads_bounded_by_traffic_tier']:2}"
              f"/{f['of_workloads']:2} wl   {', '.join(f['dead_inputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
