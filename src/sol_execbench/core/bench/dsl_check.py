# SPDX-License-Identifier: Apache-2.0
#
# NEW FILE, contributed from the AMDPilot v2 fleet side (issue amdpilotv2#19).
# It adds no behaviour to any existing code path on its own: nothing in this
# module raises, and nothing in it can reject a submission. See "LABEL ONLY".

"""The four DSL rules: what a submission can be *seen* to be written in.

SOL-ExecBench has always measured one axis — is the kernel fast and correct.
The fleet that drives it added a second one: which kernel language the agent
was *told* to write in (`triton`, `aiter`, `flydsl`, `assembly`, or no
constraint at all). The instruction is rendered into the prompt and recorded
per run; whether the submission actually stayed inside it has never been
checked anywhere, by anybody, because the only place that can check it is the
place that already reads the source it is about to compile and time. That is
here.

LABEL ONLY. THIS IS THE WHOLE DESIGN, NOT A PHASE-ONE COMPROMISE.
------------------------------------------------------------------
`dsl_labels` returns evidence and a label. `check_dsl_constraint` returns a
comparison. **Neither raises, neither returns a verdict, and no caller in this
repository may reject a submission on what they say** until a census of the
existing corpus has been run and read by a human. The reason is arithmetic: a
gate with a 2% false-positive rate applied to a 220-problem sweep scores four
correct kernels zero, and a zero is indistinguishable downstream from a kernel
that did not work. A label is recoverable. A zero is not.

The fleet's own census (`solbench-tasks/sbt/dslcheck.py` in amdpilotv2) is the
prior art here and it is worth stating what it cost: twelve packets in a
220-problem sweep were mislabelled by a first version of the reachability rule,
every one of them a *real* Triton kernel launched through a form the checker
did not know — `kernel.warmup(...)`, `CompiledKernel.run`, a launcher built by
`functools.partial` at module scope, a kernel handed to `torch.compile`. Nine
of the twelve were written that way deliberately, to skip the ~20 us of Python
that `kernel[grid](...)` costs. In other words: the submissions most likely to
be misread as non-compliant are the ones written by the strongest optimisers.
The reachability half of this module is a deliberate port of that fixed rule,
copied rather than imported — `solbench-tasks` is a fleet package and is not
installable inside the measurement image — and the two are expected to be kept
in step by hand, which is why both say so.

THE FOUR RULES
--------------
Each rule asks two questions, and both must answer yes. *Is the thing present?*
and *is it reached from the entry point?* Presence alone is how a two-line
`torch.matmul` wrapper with an unused `import triton` at the top gets labelled
compliant, and reachability alone is how a submission that never imports the
library at all does.

* **triton** — a `FunctionDef` whose decorator resolves, through a real binding
  of the `triton` module or of `triton.jit` itself, to `triton.jit`; AND a
  launch of that function reachable from the entry point. `import triton
  .language as tl` binds `tl` to the *submodule*, which has no `.jit`, so it
  satisfies neither half — that is the `L2__061` shape and it is why the rule
  is written against bindings and never against spellings. A locally defined
  `def jit(f): return f` above the kernel binds the same name and is not the
  same object, and does not count.
* **aiter** — an `aiter` binding whose symbols are **called** from the reachable
  graph. Imported and unused is not "written in aiter"; it is a leftover line.
* **flydsl** — an `import flydsl` binding, plus a `@flyc.kernel` device body,
  plus a `@flyc.jit` launcher, plus a `flyc.compile`/`.launch` call site. All
  four, because FlyDSL's device and host halves are separate decorators and
  either one alone is a fragment rather than a kernel.
* **assembly** — an inline-asm construct with a **non-empty constraint string**,
  or a `.s`/`.S` source file. `asm volatile("")` with an empty constraint is a
  compiler barrier, appears in ordinary C++ that has nothing to do with hand
  assembly, and must not satisfy this rule.

WHAT THESE RULES CANNOT YET BE VALIDATED AGAINST, STATED PLAINLY
-----------------------------------------------------------------
Two of the four ship without a corpus to check them against, and that is a
property of the rule, not of the effort spent on it:

* **assembly ships inline-asm-only.** Verified against this tree at
  `bf3d083f`: `src/sol_execbench/driver/templates/build_ext.py` collects
  `.cu .hip .cpp .cc .cxx .c` and nothing else, so there is **no `.s`/`.S`
  build path in this repository at all**. A submission cannot currently ship a
  standalone assembly source and have it compiled, so the `.s` half of the rule
  is written for a build path that does not exist yet and has never fired.
* **flydsl has no corpus.** Zero of the harvested kernels across the three
  archived sweeps import `flydsl`. The rule is written from the framework's
  documented decorator pair and is unvalidated by construction. Its first true
  positive will also be its first test.

Neither is a reason to leave them out — a vocabulary with two of its four
values undefined is worse than one with two of them unexercised — but neither
may be treated as measured. `dsl_labels` reports `validated: False` for both.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ── the vocabulary ────────────────────────────────────────────────────────────
#: The four constrained values, in the order the axis is written down. There is
#: no fifth value for "no constraint": an unconstrained run carries no declared
#: DSL at all, and an absent field is what "nobody stated one" looks like.
DSLS: tuple[str, ...] = ("triton", "aiter", "flydsl", "assembly")

#: The submission's entry point, as the harness calls it.
ENTRY_FILE = "kernel.py"
ENTRY_FUNCTION = "run"

#: Files that are staged beside a submission and are not part of it.
NOT_THE_SUBMISSION = ("reference.py",)

#: Triton's low-level launch path. `k[grid](...)` costs about 20 us of Python
#: and a submission optimising a small kernel will avoid it; both of these are
#: launches and a checker that only knows the subscript form will call the
#: fastest submissions non-compliant.
LAUNCH_ATTRIBUTES = ("warmup", "run")

#: Kernel libraries a submission can dispatch into instead of authoring.
DISPATCH_LIBRARIES = ("aiter", "hipblaslt", "flash_attn")

#: How far an alias chain is followed before it is treated as a cycle.
_ALIAS_DEPTH = 8

#: The synthetic unit name for a file's module-level statements. They run on
#: import, so a launch written there is reached.
MODULE_SCOPE = "<module>"

#: The two rules with no corpus behind them. Reported on every result so a
#: reader never has to remember which of the four have been exercised.
UNVALIDATED = ("flydsl", "assembly")


# ── source normalisation, shared with the static screen ───────────────────────

def normalise_sources(sources: Any) -> dict[str, str]:
    """`{path: content}` from whatever shape the caller has.

    Accepts exactly what `reward_hack.static_source_screen` accepts — an
    iterable of objects with `.path`/`.content`, or of `(path, content)` pairs —
    plus a plain mapping, because the census script and the tests have one.
    Deliberately the same normaliser rather than a second one: the two screens
    run at the same chokepoint on the same object, and two readings of "what
    the submission is" is the beginning of two answers.
    """
    if isinstance(sources, dict):
        return {str(k): (v or "") for k, v in sources.items()}
    out: dict[str, str] = {}
    for src in sources or ():
        if isinstance(src, (tuple, list)) and len(src) == 2:
            path, content = src
        else:
            path = getattr(src, "path", "?")
            content = getattr(src, "content", "")
        out[str(path)] = content or ""
    return out


# ── one file, read on its own ─────────────────────────────────────────────────

@dataclass
class FileFacts:
    """What one source file says about itself, before anything is joined."""

    path: str
    #: Names bound to the `triton` MODULE. Not `triton.language`: `import
    #: triton.language as tl` binds the submodule and `tl.jit` does not exist.
    triton_names: set[str] = field(default_factory=set)
    #: Names bound directly to `triton.jit` (`from triton import jit`).
    jit_names: set[str] = field(default_factory=set)
    #: Any triton import at all, including `triton.language` alone.
    imports_triton: bool = False
    #: Names bound to the `flydsl` module or its `flyc` handle.
    flydsl_names: set[str] = field(default_factory=set)
    imports_flydsl: bool = False
    #: Local name -> dispatch library.
    dispatch_names: dict[str, str] = field(default_factory=dict)
    functions: dict[str, ast.AST] = field(default_factory=dict)
    jit_functions: set[str] = field(default_factory=set)
    #: Functions carrying `@flyc.kernel` and `@flyc.jit` respectively.
    flydsl_device_functions: set[str] = field(default_factory=set)
    flydsl_launcher_functions: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    assigned_from: dict[str, ast.AST] = field(default_factory=dict)
    sibling_symbols: dict[str, str] = field(default_factory=dict)
    sibling_modules: dict[str, str] = field(default_factory=dict)
    #: Inline-asm call sites with a NON-EMPTY constraint string.
    asm_sites: list[str] = field(default_factory=list)
    syntax_error: str | None = None

    def imported_siblings(self) -> set[str]:
        return set(self.sibling_modules.values()) | set(
            self.sibling_symbols.values())


def _module_name(node: ast.AST) -> str:
    """`triton.language` from an Attribute/Name chain, dotted, or ""."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _root(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _bound_functions(body: list[ast.stmt]) -> list[ast.AST]:
    """Every `def` binding a module-level name, however deep in a block.

    A submission that defines its `run` twice, once in each arm of an
    `if _FAST is not None:`, binds the module's `run` either way. Reading only
    `tree.body` finds neither, and the submission is then recorded as having no
    entry point at all.
    """
    found: list[ast.AST] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node)
        elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While,
                               ast.With, ast.AsyncWith, ast.AsyncFor)):
            for name in ("body", "orelse", "finalbody", "handlers"):
                inner = getattr(node, name, None) or []
                if name == "handlers":
                    for handler in inner:
                        found += _bound_functions(handler.body)
                else:
                    found += _bound_functions(inner)
    return found


def _prune(node: ast.stmt) -> list[ast.stmt]:
    """One statement reduced to the part of it that runs on import."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # The decorators still run at import; the body does not.
        return [ast.Expr(value=d) for d in getattr(node, "decorator_list", [])]
    if isinstance(node, (ast.If, ast.Try, ast.For, ast.While,
                         ast.With, ast.AsyncWith, ast.AsyncFor)):
        inner: list[ast.stmt] = []
        for name in ("body", "orelse", "finalbody"):
            for child in getattr(node, name, None) or []:
                inner += _prune(child)
        for handler in getattr(node, "handlers", None) or []:
            for child in handler.body:
                inner += _prune(child)
        for extra in ("test", "iter", "items"):
            value = getattr(node, extra, None)
            if isinstance(value, ast.AST):
                inner.append(ast.Expr(value=value))  # type: ignore[arg-type]
        return inner
    return [node]


def _module_scope_nodes(tree: ast.Module) -> list[ast.stmt]:
    """The module's statements with every def/class body pruned out.

    Module scope is walked as a reachable unit because its launches run at
    import. Leaving the function bodies in would make every helper in the file
    reachable and "defined but never launched" would stop meaning anything.
    """
    kept: list[ast.stmt] = []
    for node in tree.body:
        kept += _prune(node)
    return kept


def _name_candidates(value: ast.AST) -> list[str]:
    """Every bare name an assignment's right-hand side could bind to.

    `kern = _fast if masked else _even` gives two, and a real submission writes
    exactly that line above the only launch in the file.
    """
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, ast.IfExp):
        return _name_candidates(value.body) + _name_candidates(value.orelse)
    if isinstance(value, ast.BoolOp):
        return [n for v in value.values for n in _name_candidates(v)]
    if isinstance(value, (ast.Tuple, ast.List)):
        return [n for v in value.elts for n in _name_candidates(v)]
    return []


def _assignments(body) -> list[tuple[str, list[str]]]:
    """`(target, candidate names)` for every simple name-to-name assignment."""
    out: list[tuple[str, list[str]]] = []
    for node in body:
        if not isinstance(node, ast.Assign):
            continue
        names = _name_candidates(node.value)
        if not names:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out.append((target.id, names))
    return out


#: An inline-asm construct in a Python string destined for a C++/HIP extension,
#: or in a `.cpp`/`.hip` source staged beside it. The constraint string is the
#: discriminator and it is why this is a small parser rather than a substring
#: test: `asm volatile("")` and `__asm__ __volatile__("" ::: "memory")` are
#: compiler barriers that appear in ordinary code, and matching the word `asm`
#: would label every submission that contains one as hand-written assembly.
_ASM_CALL = re.compile(
    r'(?:__asm__|asm)\s*(?:__volatile__|volatile)?\s*\(\s*(?P<body>.*?)\)\s*;',
    re.DOTALL)

#: gfx950 (MI355X) mnemonics. Presence of one is what separates a real inline
#: kernel body from an `"nop"` or a memory clobber.
_GFX_MNEMONICS = re.compile(
    r'\b(v_mfma\w*|v_dot\w*|v_pk_\w+|ds_read\w*|ds_write\w*|buffer_load\w*|'
    r'buffer_store\w*|global_load\w*|global_store\w*|s_waitcnt|s_barrier|'
    r'v_fma\w*|v_mac\w*|v_add\w*|v_mul\w*|v_cvt\w*)\b')


def _asm_findings(text: str, path: str) -> list[str]:
    """Inline-asm sites with a non-empty constraint AND a real mnemonic.

    Both halves are required. A non-empty constraint alone admits
    `asm("" : "=r"(x))`, which is a compiler hint; a mnemonic alone admits a
    comment or a docstring that talks about `v_mfma`. Returns human-readable
    site descriptions, never a verdict.
    """
    found: list[str] = []
    for match in _ASM_CALL.finditer(text or ""):
        body = match.group("body")
        # The constraint string is everything after the first unquoted colon.
        # An asm with no colon at all has no operands and no clobbers, which is
        # the barrier form.
        head, sep, constraints = body.partition(":")
        if not sep or not constraints.strip(" \t\n:"):
            continue
        if not _GFX_MNEMONICS.search(head):
            continue
        found.append(f"{path}: inline asm with operands and a gfx mnemonic")
    return found


def analyse(text: str, name: str, siblings: set[str]) -> FileFacts:
    """Every fact one file states, with no cross-file resolution yet."""
    facts = FileFacts(path=name)
    facts.asm_sites = _asm_findings(text, name)
    try:
        tree = ast.parse(text, filename=name)
    except (SyntaxError, ValueError) as exc:
        # Reported, never treated as an empty file. A submission that does not
        # parse is not a submission with no Triton in it; it is a submission
        # this reader could not read, and those are different rows.
        facts.syntax_error = f"{type(exc).__name__}: {exc}"
        return facts

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or _root(alias.name)
                if _root(alias.name) == "triton":
                    facts.imports_triton = True
                    if alias.asname is None or alias.name == "triton":
                        facts.triton_names.add(bound)
                if _root(alias.name) == "flydsl":
                    facts.imports_flydsl = True
                    facts.flydsl_names.add(bound)
                if _root(alias.name) in DISPATCH_LIBRARIES:
                    facts.dispatch_names[bound] = _root(alias.name)
                if _root(alias.name) in siblings:
                    facts.sibling_modules[bound] = _root(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _root(module) == "triton":
                facts.imports_triton = True
                for alias in node.names:
                    if alias.name == "jit" and module == "triton":
                        facts.jit_names.add(alias.asname or "jit")
            if _root(module) == "flydsl":
                facts.imports_flydsl = True
                for alias in node.names:
                    facts.flydsl_names.add(alias.asname or alias.name)
            if _root(module) in DISPATCH_LIBRARIES:
                for alias in node.names:
                    facts.dispatch_names[alias.asname or alias.name] = \
                        _root(module)
            if _root(module) in siblings and not node.level:
                for alias in node.names:
                    facts.sibling_symbols[alias.asname or alias.name] = \
                        _root(module)

    for node in _bound_functions(tree.body):
        facts.functions[node.name] = node          # type: ignore[attr-defined]
        if _decorated_with_jit(node, facts):
            facts.jit_functions.add(node.name)     # type: ignore[attr-defined]
        for attr, bucket in (("kernel", facts.flydsl_device_functions),
                             ("jit", facts.flydsl_launcher_functions)):
            if _decorated_with_flydsl(node, facts, attr):
                bucket.add(node.name)              # type: ignore[attr-defined]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            facts.classes.add(node.name)
            for child in _bound_functions(node.body):
                facts.functions[f"{node.name}.{child.name}"] = child  # type: ignore[attr-defined]

    for target, names in _assignments(tree.body):
        facts.aliases.setdefault(target, []).extend(names)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    facts.assigned_from[target.id] = node.value

    facts.functions[MODULE_SCOPE] = ast.Module(
        body=_module_scope_nodes(tree), type_ignores=[])
    return facts


def _decorated_with_jit(node: ast.AST, facts: FileFacts) -> bool:
    """Whether a decorator on this function resolves to `triton.jit`.

    Through a binding, never by spelling. `@jit` counts only when `jit` was
    imported from triton; `@triton.jit` counts only when `triton` is bound to
    the triton module. A local `def jit(f): return f` above the kernel — which
    is how a submission would fake this if anything ever scored on it — binds
    the same name and is not in `facts.jit_names`, so it does not count.

    `@triton.autotune(...)` and `@triton.heuristics(...)` are call-form
    decorators that WRAP a jit, and `@triton.jit(do_not_specialize=[...])` IS
    one, so the stack is read whole and each decorator resolved on its own.
    """
    for decorator in getattr(node, "decorator_list", []):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id in facts.jit_names:
            return True
        if isinstance(target, ast.Attribute) and target.attr == "jit":
            if isinstance(target.value, ast.Name) and \
                    target.value.id in facts.triton_names:
                return True
    return False


def _decorated_with_flydsl(node: ast.AST, facts: FileFacts,
                           attribute: str) -> bool:
    """`@flyc.kernel` / `@flyc.jit`, resolved through a real flydsl binding.

    Same discipline as the triton rule and for the same reason: `jit` is a
    common attribute name and a submission that writes `@self.jit` or
    `@numba.jit` must not fall into the flydsl bucket.
    """
    for decorator in getattr(node, "decorator_list", []):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == attribute:
            owner = _root(_module_name(target))
            if owner in facts.flydsl_names:
                return True
    return False


# ── the import closure: which files are the submission ────────────────────────

def import_closure(files: dict[str, FileFacts], entry: str) -> set[str]:
    """The entry module and every module it transitively imports.

    Everything else in a staged tree is the agent's scratch. One archived
    submission staged twelve `exp*.py` files with live `@triton.jit` kernels
    beside a `kernel.py` that is pure torch and imports none of them; counting
    those made a torch submission look like Triton. Scratch is reported and
    decides nothing.
    """
    seen: set[str] = set()
    stack = [entry]
    while stack:
        module = stack.pop()
        if module in seen or module not in files:
            continue
        seen.add(module)
        stack += sorted(files[module].imported_siblings())
    return seen


@dataclass
class Reach:
    """The reachable call graph from the entry point, and what is in it."""

    visited: set[tuple[str, str]] = field(default_factory=set)
    launched: set[str] = field(default_factory=set)
    dispatched: set[str] = field(default_factory=set)
    flydsl_calls: set[str] = field(default_factory=set)
    forms: set[str] = field(default_factory=set)
    entry_found: bool = False


def _local_aliases(node: ast.AST) -> dict[str, list[str]]:
    """`a = b` between bare names anywhere inside one callable unit."""
    out: dict[str, list[str]] = {}
    for target, names in _assignments(list(ast.walk(node))):
        out.setdefault(target, []).extend(names)
    return out


def _resolve_jit(name: str, module: str, files: dict[str, FileFacts],
                 local: dict[str, list[str]], depth: int = 0) -> str | None:
    """A bare name to the qualified jit it is bound to, through assignments."""
    facts = files.get(module)
    if facts is None or depth > _ALIAS_DEPTH:
        return None
    if name in facts.jit_functions:
        return f"{module}.{name}"
    sibling = facts.sibling_symbols.get(name)
    if sibling and name in files.get(sibling, FileFacts("")).jit_functions:
        return f"{sibling}.{name}"
    for nxt in (local.get(name) or []) + (facts.aliases.get(name) or []):
        if nxt == name:
            continue
        found = _resolve_jit(nxt, module, files, local, depth + 1)
        if found:
            return found
    return None


def _resolve_jit_expression(node: ast.AST, module: str,
                            files: dict[str, FileFacts],
                            local: dict[str, list[str]]) -> str | None:
    """A jit kernel behind a `Name` or a `sibling.name` attribute, or None."""
    if isinstance(node, ast.Name):
        return _resolve_jit(node.id, module, files, local)
    if isinstance(node, ast.Attribute):
        facts = files.get(module)
        if facts is None:
            return None
        owner = _root(_module_name(node))
        sibling = facts.sibling_modules.get(owner)
        if sibling and node.attr in \
                files.get(sibling, FileFacts("")).jit_functions:
            return f"{sibling}.{node.attr}"
    return None


def _walk(files: dict[str, FileFacts], module: str, function: str,
          reach: Reach) -> None:
    """Depth-first over the call graph, one callable unit at a time.

    Bounded by `reach.visited`, keyed on `(module, function)`: mutual recursion
    is normal in a submission and must not hang the reader.
    """
    if (module, function) in reach.visited:
        return
    facts = files.get(module)
    if facts is None or function not in facts.functions:
        return
    reach.visited.add((module, function))

    unit = facts.functions[function]
    local = _local_aliases(unit)
    if function == MODULE_SCOPE:
        # Importing a module runs its siblings' module scope too.
        for sibling in sorted(facts.imported_siblings()):
            _walk(files, sibling, MODULE_SCOPE, reach)

    for node in ast.walk(unit):
        # The low-level launch path: `k.warmup(...)`, `k.run(...)`. Nine of the
        # twelve packets a first version of this rule mislabelled are here, and
        # none of them by accident.
        if isinstance(node, ast.Attribute) and node.attr in LAUNCH_ATTRIBUTES:
            jit = _resolve_jit_expression(node.value, module, files, local)
            if jit:
                reach.launched.add(jit)
                reach.forms.add(f"{node.attr}-attribute")

        # `kernel[grid](args)`. Subscripting a jit function is Triton's
        # launcher protocol and means nothing else, so the subscript counts
        # wherever it appears and not only where it is called on the spot: a
        # launcher can be built by `partial(k[(1,)], num_warps=1)` in one
        # statement and used in another.
        if isinstance(node, ast.Subscript):
            jit = _resolve_jit_expression(node.value, module, files, local)
            if jit:
                reach.launched.add(jit)
                reach.forms.add("subscript-grid")
            continue

        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # A dispatch into a kernel library, CALLED and not merely imported.
        if isinstance(func, ast.Name) and func.id in facts.dispatch_names:
            reach.dispatched.add(f"{facts.dispatch_names[func.id]}:{func.id}")
        if isinstance(func, ast.Attribute):
            dotted = _module_name(func)
            owner = _root(dotted)
            if owner in facts.dispatch_names:
                reach.dispatched.add(f"{facts.dispatch_names[owner]}:{dotted}")
            # FlyDSL's host half: `flyc.compile(...)` / `handle.launch(...)`.
            if owner in facts.flydsl_names and func.attr in ("compile", "launch"):
                reach.flydsl_calls.add(dotted)

        # A step deeper. `X.apply(...)` on a module-level class is the
        # autograd.Function shape and reaches `forward`/`backward`.
        if isinstance(func, ast.Attribute):
            owner_name = func.value.id if isinstance(func.value, ast.Name) else ""
            if owner_name in facts.classes:
                targets = ("forward", "backward") if func.attr == "apply" \
                    else (func.attr,)
                for target in targets:
                    _walk(files, module, f"{owner_name}.{target}", reach)
            sibling = facts.sibling_modules.get(_root(_module_name(func)))
            if sibling:
                _walk(files, sibling, func.attr, reach)

    # Any REFERENCE to a function this submission defines, called or not.
    # `torch.compile(_run_impl)` and `partial(_gelu_backward, ...)` hand the
    # function to somebody else to call; the kernel underneath still runs.
    for node in ast.walk(unit):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in facts.functions:
                _walk(files, module, node.id, reach)
            sibling = facts.sibling_symbols.get(node.id)
            if sibling:
                _walk(files, sibling, node.id, reach)


def _entry_units(files: dict[str, FileFacts], entry: str) -> list[tuple[str, str]]:
    """Where execution starts, in every form the corpus writes it.

    `kernel.py::run` when it is defined there — including inside an if/else,
    which binds the module name just the same. `run = _run` when it is an
    alias. `from candidate import run` when the submission re-exported it. And
    always module scope, which runs on import.
    """
    units: list[tuple[str, str]] = [(entry, MODULE_SCOPE)]
    facts = files.get(entry)
    if facts is None:
        return units
    if ENTRY_FUNCTION in facts.functions:
        units.append((entry, ENTRY_FUNCTION))
        return units
    pending, seen = [ENTRY_FUNCTION], set()
    while pending:
        name = pending.pop()
        if name in seen or len(seen) > _ALIAS_DEPTH:
            continue
        seen.add(name)
        sibling = facts.sibling_symbols.get(name)
        if sibling:
            units.append((sibling, name))
            continue
        if name in facts.functions:
            units.append((entry, name))
            continue
        pending += facts.aliases.get(name) or []
        # `run = torch.compile(_run_impl, dynamic=False)`. The entry point is
        # real and the harness calls it; only its spelling is a wrapper.
        wrapped = facts.assigned_from.get(name)
        if wrapped is not None:
            pending += [child.id for child in ast.walk(wrapped)
                        if isinstance(child, ast.Name)
                        and child.id in facts.functions]
    return units


# ── the four rules ────────────────────────────────────────────────────────────

def dsl_labels(sources: Any) -> dict[str, Any]:
    """What this submission can be SEEN to be written in. Never raises.

    Returns a dict with, per DSL in `DSLS`, an evidence block carrying
    `satisfied` (bool), `present` (bool), `reached` (bool), the specific
    findings behind both, and `validated` — False for the two rules that ship
    without a corpus. Plus `labels`, the sorted list of DSLs satisfied, and
    `read`/`unread`/`unparsed`, so a caller can tell "found nothing" apart from
    "could not look".

    Pure and dict-in-dict-out on purpose: every case is then testable without a
    filesystem, a GPU, a container or a run.
    """
    raw = normalise_sources(sources)
    python = {name: text for name, text in raw.items()
              if name.endswith(".py") and name not in NOT_THE_SUBMISSION}
    unread = sorted(name for name in raw if name not in python)

    stems = {name.rsplit("/", 1)[-1][:-3] for name in python}
    files = {name.rsplit("/", 1)[-1][:-3]: analyse(text, name, stems)
             for name, text in python.items()}

    entry = ENTRY_FILE[:-3]
    closure = import_closure(files, entry) if files else set()

    reach = Reach()
    units = _entry_units(files, entry) if files else []
    reach.entry_found = any(unit != MODULE_SCOPE for _, unit in units)
    for module, unit in units:
        _walk(files, module, unit, reach)

    jits = {f"{m}.{n}" for m in sorted(closure) for n in files[m].jit_functions}
    scratch_jits = {f"{m}.{n}" for m in sorted(set(files) - closure)
                    for n in files[m].jit_functions}
    launched = {n for n in reach.launched if _root(n) in closure}
    imports_triton = any(files[m].imports_triton for m in closure)
    imports_aiter = sorted({n for m in closure
                            for n, lib in files[m].dispatch_names.items()
                            if lib == "aiter"})
    called_aiter = sorted(n for n in reach.dispatched if n.startswith("aiter:"))
    flydsl_devices = {f"{m}.{n}" for m in sorted(closure)
                      for n in files[m].flydsl_device_functions}
    flydsl_launchers = {f"{m}.{n}" for m in sorted(closure)
                        for n in files[m].flydsl_launcher_functions}
    imports_flydsl = any(files[m].imports_flydsl for m in closure)
    # Assembly is read off EVERY staged source and not only the import closure:
    # a `.hip` file is compiled by the extension builder, which does not consult
    # Python's import graph at all, so closure membership is the wrong question
    # for it. The `.s`/`.S` half is written for a build path this repository
    # does not have yet — see the module docstring.
    asm_sites = sorted(s for f in
                       [analyse(t, n, set()) if n.endswith(".py")
                        else FileFacts(n, asm_sites=_asm_findings(t, n))
                        for n, t in raw.items()]
                       for s in f.asm_sites)
    asm_sources = sorted(n for n in raw if n.endswith((".s", ".S")))

    rules: dict[str, dict[str, Any]] = {
        "triton": {
            "present": bool(jits),
            "reached": bool(launched),
            "satisfied": bool(jits and launched),
            "validated": "triton" not in UNVALIDATED,
            "jits_defined": sorted(jits),
            "jits_launched_from_entry": sorted(launched),
            "jits_outside_the_import_graph": sorted(scratch_jits),
            "launch_forms": sorted(reach.forms),
            "imports_triton": imports_triton,
            "why_not": (
                None if jits and launched else
                "no @triton.jit resolves through a real triton binding"
                if not jits else
                "kernels are defined but none is launched from the entry point"),
        },
        "aiter": {
            "present": bool(imports_aiter),
            "reached": bool(called_aiter),
            "satisfied": bool(called_aiter),
            "validated": "aiter" not in UNVALIDATED,
            "imported": imports_aiter,
            "called_from_entry": called_aiter,
            "why_not": (
                None if called_aiter else
                "aiter is not imported anywhere in the import closure"
                if not imports_aiter else
                "aiter is imported but nothing from it is called from the "
                "entry point; an unused import is a leftover line"),
        },
        "flydsl": {
            "present": bool(imports_flydsl),
            "reached": bool(reach.flydsl_calls),
            "satisfied": bool(imports_flydsl and flydsl_devices
                              and flydsl_launchers and reach.flydsl_calls),
            "validated": "flydsl" not in UNVALIDATED,
            "imports_flydsl": imports_flydsl,
            "device_functions": sorted(flydsl_devices),
            "launcher_functions": sorted(flydsl_launchers),
            "call_sites": sorted(reach.flydsl_calls),
            "why_not": (
                None if (imports_flydsl and flydsl_devices
                         and flydsl_launchers and reach.flydsl_calls)
                else "all four of import, @flyc.kernel, @flyc.jit and a "
                     "compile/launch call site are required; see the module "
                     "docstring for why this rule has no corpus behind it"),
        },
        "assembly": {
            "present": bool(asm_sites or asm_sources),
            "reached": bool(asm_sites or asm_sources),
            "satisfied": bool(asm_sites or asm_sources),
            "validated": "assembly" not in UNVALIDATED,
            "inline_sites": asm_sites,
            "asm_sources": asm_sources,
            "why_not": (
                None if (asm_sites or asm_sources) else
                "no inline asm with a non-empty constraint string and a gfx "
                "mnemonic, and no .s/.S source (which this repository has no "
                "build path for in any case)"),
        },
    }

    return {
        "labels": sorted(d for d in DSLS if rules[d]["satisfied"]),
        "rules": rules,
        "read": sorted(python),
        # Staged, and NOT parsed by this reader. Reported so a submission whose
        # only source is a `.hip` reads as "not looked at" rather than as
        # "nothing found" — silence is not a finding.
        "unread": unread,
        "import_closure": sorted(f"{m}.py" for m in closure),
        "entry_point_found": reach.entry_found,
        "unparsed": {f.path: f.syntax_error
                     for f in files.values() if f.syntax_error},
        "unvalidated_rules": list(UNVALIDATED),
    }


def check_dsl_constraint(sources: Any, declared: Any = None) -> dict[str, Any]:
    """Compare what was declared against what can be seen. NEVER raises.

    `declared` is what the run said the agent was constrained to — a string, an
    iterable of strings, or None for an unconstrained run. None is not a
    failure and not a pass: it is `declared: null`, `agrees: null`, because a
    run with no constraint has nothing to disagree with.

    A declared value this module has no rule for is a THIRD case, kept distinct
    from both: it is reported in `unruled` and as `per_declared[value] = null`,
    and it drives `agrees` to null rather than to False. Silently dropping it
    would make a run constrained to an unknown DSL read as an unconstrained
    run, which is a different fact.

    The return value has **no verdict field and no exit code**, and that is
    load-bearing. Nothing in this repository may reject a submission on this
    result until the census in `scripts/dsl_census.py` has been run over the
    archived sweeps and read by a human; the first thing a premature gate would
    do is score a correct kernel zero for a launch form the reader does not
    know, and this rule has already been wrong that way twelve times in one
    sweep on its way to being right.
    """
    seen = dsl_labels(sources)
    if declared is None:
        stated: list[str] = []
    elif isinstance(declared, str):
        stated = [declared]
    else:
        stated = [str(d) for d in declared]

    # Three outcomes per declared value, never two. `ruled` is a value this
    # module has a rule for and can therefore pass or fail; `unruled` is one it
    # has none for, and the only honest answer for those is "no rule", not
    # False and not a silent drop.
    #
    # Dropping them -- which is what a bare `[d for d in stated if d in DSLS]`
    # does -- makes `declared=["cutlass"]` indistinguishable from an
    # unconstrained run, so the two live in separate fields. That conflation is
    # the same shape as the `_is_cpp` defect this change set fixes in the
    # driver: one answer serving two questions.
    ruled = [d for d in stated if d in DSLS]
    unruled = [d for d in stated if d not in DSLS]

    per_declared: dict[str, bool | None] = {
        d: (seen["rules"][d]["satisfied"] if d in DSLS else None) for d in stated
    }

    if not stated:
        agrees = None
        note = ("no DSL was declared for this run, so there is nothing to "
                "disagree with. The labels below are a description, not a "
                "comparison.")
    elif unruled:
        # Unknown, and said so. Even when every RULED value is visible, the
        # declaration as a whole has not been checked, and reporting True here
        # would be a pass this module did not earn.
        agrees = None
        note = (f"no rule for {sorted(set(unruled))}: this reader has rules "
                f"only for {list(DSLS)}, so those values can be neither passed "
                f"nor failed. `agrees` is null because the declaration as a "
                f"whole was not checked; see per_declared for the values that "
                f"were. This is a LABEL. It rejects nothing.")
    else:
        agrees = all(per_declared[d] for d in ruled)
        note = ("every declared DSL is visible in the submission" if agrees
                else "at least one declared DSL is not visible in the "
                     "submission — see rules[<dsl>]['why_not']. This is a "
                     "LABEL. It rejects nothing.")
    return {
        "declared": stated or None,
        # The declared values this module can actually answer for, and those it
        # cannot. A caller that reads only `agrees` is never silently told that
        # a value it asked about was ignored.
        "constrained": ruled or None,
        "unruled": unruled,
        "per_declared": per_declared,
        "labels": seen["labels"],
        "agrees": agrees,
        "note": note,
        "evidence": seen,
    }
