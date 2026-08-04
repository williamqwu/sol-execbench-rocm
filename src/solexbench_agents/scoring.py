# SPDX-License-Identifier: Apache-2.0
"""Pure scoring logic: no GPU, no filesystem sweep, fully unit-testable.

The one formula that matters is upstream's:

    S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))

``S = 1`` at the Speed-of-Light bound, ``S = 0.5`` at the optimized-PyTorch
anchor. It needs three inputs and this repo currently has two of them on this
part, so :func:`score_record` is explicit about which basis it used rather than
substituting something plausible for the missing one. A score whose basis is
unstated is worse than no score: it invites comparison against numbers derived
differently.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class ScoreBasis(str, Enum):
    """What a record's headline number actually means.

    Ordered weakest to strongest. Records of different bases are never averaged
    together; the scoreboard groups by basis and says so.
    """

    CORRECTNESS_ONLY = "correctness_only"
    """Pass/fail against AMD-derived tolerances. No timing claim."""

    SPEEDUP_VS_REFERENCE = "speedup_vs_reference"
    """Also T_k against the problem's own reference, measured in the same run.
    This is *not* T_b: the anchor is an optimized-PyTorch variant selected by a
    sweep (task 06), while the reference is whatever the dataset shipped. Using
    one as the other would silently rescale every score."""

    SOL_HEADROOM = "sol_headroom"
    """Also the fraction of hardware headroom reclaimed against a T_SOL derived
    for *this part*. Comparable across problems; still not the SOL score."""

    SOL_SCORE_V1 = "sol_score_v1"
    """The real thing: T_SOL and T_b both present and part-correct."""


def sol_score(t_k_ms: float, t_sol_ms: float, t_b_ms: float) -> float | None:
    """Upstream's S. None when the inputs cannot support it.

    Guards, each of which corresponds to a way the bound can be wrong rather
    than the kernel:

    - ``t_b <= t_sol`` makes the denominator zero or negative. That is not a
      fast kernel, it is a broken bound -- the anchor cannot be at or beyond the
      Speed-of-Light limit -- and returning a number would hide it.
    - ``t_k < t_sol`` means the kernel beat the analytic lower bound, which is
      impossible and indicates the bound is too loose. S would exceed 1. The
      caller is expected to surface it; deviation D12 is what this looks like in
      practice.
    """
    if t_k_ms is None or t_sol_ms is None or t_b_ms is None:
        return None
    if t_b_ms <= t_sol_ms:
        return None
    if t_k_ms <= 0:
        return None
    return 1.0 / (1.0 + (t_k_ms - t_sol_ms) / (t_b_ms - t_sol_ms))


def headroom_fraction(t_k_ms: float, t_ref_ms: float, t_sol_ms: float) -> float | None:
    """``(T_ref - T_k) / (T_ref - T_SOL)`` -- the share of the gap closed.

    Upstream reports S correlating with this at r = 0.981, and task 09 asks for
    that correlation to be reproduced on AMD before release. It is computable
    from a reference timing alone, so it is available before T_b exists, which is
    why it is the strongest basis this repo can currently reach.
    """
    if t_k_ms is None or t_ref_ms is None or t_sol_ms is None:
        return None
    denom = t_ref_ms - t_sol_ms
    if denom <= 0:
        return None
    return (t_ref_ms - t_k_ms) / denom


def resolve_basis(*, correct: bool, t_k_ms: float | None,
                  t_ref_ms: float | None, t_sol_ms: float | None,
                  t_b_ms: float | None) -> ScoreBasis:
    """The strongest basis the available inputs support."""
    if not correct or not t_k_ms:
        return ScoreBasis.CORRECTNESS_ONLY
    if t_ref_ms and t_sol_ms and t_b_ms:
        return ScoreBasis.SOL_SCORE_V1
    if t_ref_ms and t_sol_ms:
        return ScoreBasis.SOL_HEADROOM
    if t_ref_ms:
        return ScoreBasis.SPEEDUP_VS_REFERENCE
    return ScoreBasis.CORRECTNESS_ONLY


# --- is this solution just the reference again? ----------------------------

@dataclass(frozen=True)
class CopyVerdict:
    """How close a solution is to simply resubmitting the reference."""

    kind: str          # "exact" | "near" | "distinct"
    similarity: float  # 0..1 on normalized source
    detail: str = ""

    @property
    def is_copy(self) -> bool:
        return self.kind in ("exact", "near")


def _normalize_source(src: str) -> str | None:
    """AST dump with docstrings dropped, so comments and formatting do not count.

    Returns None when the source does not parse as Python -- a Triton or HIP
    solution may not, and "not Python" is a perfectly good answer to "is this the
    reference verbatim".
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=False)


def reference_copy_verdict(solution_sources: Iterable, reference_src: str,
                           near_threshold: float = 0.95) -> CopyVerdict:
    """Did the agent hand back the reference?

    Worth detecting explicitly rather than leaving implicit in the score. A
    resubmitted reference is *correct*, so it lifts a pass-rate column without
    demonstrating any kernel engineering, and it lands near the anchor -- around
    ``S = 0.5`` by construction. Reporting it separately keeps "how often did the
    agent produce something that runs" from being read as "how often did the
    agent write a kernel".

    It is not cheating and is not penalized here; it is labelled.
    """
    ref_norm = _normalize_source(reference_src)
    best = CopyVerdict("distinct", 0.0)
    for src in solution_sources or ():
        content = getattr(src, "content", None)
        if content is None and isinstance(src, dict):
            content = src.get("content")
        if not content:
            continue
        sol_norm = _normalize_source(content)
        if sol_norm is None or ref_norm is None:
            # Not Python on at least one side: fall back to raw text ratio, which
            # still catches a reference pasted into a .py alongside other files.
            ratio = difflib.SequenceMatcher(None, content, reference_src).ratio()
        elif sol_norm == ref_norm:
            return CopyVerdict("exact", 1.0, "AST identical to the reference")
        else:
            ratio = difflib.SequenceMatcher(None, sol_norm, ref_norm).ratio()
        if ratio > best.similarity:
            kind = "near" if ratio >= near_threshold else "distinct"
            best = CopyVerdict(kind, ratio,
                               f"closest source similarity {ratio:.3f}")
    return best


# --- is the clock actually locked? -----------------------------------------

def clock_lock_state(repo_root: Path, gpu: int) -> dict:
    """Probe whether *gpu*'s clocks are really locked, from sysfs.

    The harness's own ``are_clocks_locked()`` reads the
    ``SOL_EXECBENCH_CLOCKS_LOCKED`` environment variable, which upstream's Docker
    entrypoint sets after locking. That is a *declaration*: exporting it does not
    lock anything, and exporting it on an unlocked node would make every latency
    read as authoritative while being taken at a boost clock that varies 10-30%
    under load. So it is probed here and the variable is only set when the probe
    agrees.

    ``power_dpm_force_performance_level`` reads ``perf_determinism`` when
    ``rocm-smi --setperfdeterminism`` has been applied. Read through the PCI-bus
    mapping rather than by card index, because on this node ``card1`` is torch 0
    and ``card57`` is torch 7 -- checking the wrong card would report the state of
    a different GPU, which is the same class of error as sampling the wrong GPU's
    clock (see ``scripts/gpu_map.py``).
    """
    state: dict = {"gpu": gpu, "locked": False}
    try:
        import sys

        sys.path.insert(0, str(repo_root / "scripts"))
        from gpu_map import torch_to_drm_card

        card = torch_to_drm_card().get(gpu)
        if card is None:
            state["error"] = f"no DRM card resolves to torch GPU {gpu}"
            return state
        state["drm_card"] = card
        level = (Path(card) / "device" / "power_dpm_force_performance_level")
        value = level.read_text().strip()
        state["performance_level"] = value
        state["locked"] = value in ("perf_determinism", "manual")
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"{type(exc).__name__}: {exc}"
    return state


# --- did anything move under the harness while agents were running? --------

# Only the code that decides whether a solution is correct and how fast it is.
# Deliberately not the whole repo: artifacts and docs change constantly during a
# session and would make the digest useless through noise.
INTEGRITY_PATHS = (
    "src/sol_execbench",
    "src/solexbench_rocm",
    "src/solexbench_agents",
    "scripts/agent_verify.py",
    "scripts/score_solutions.py",
)


def tree_digest(repo_root: Path, paths: Iterable[str] = INTEGRITY_PATHS) -> dict:
    """A content digest of the scoring-critical source tree.

    Recorded when a sweep starts and re-checked when it is scored. The agents run
    as local processes with a writable filesystem, so tampering cannot be
    *prevented* here; this is what makes it detectable, which is the honest claim.
    A mismatch does not prove misconduct -- an operator edit mid-sweep produces
    the same signal -- so it is reported, not judged.
    """
    files: list[tuple[str, str]] = []
    for rel in sorted(paths):
        target = repo_root / rel
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = sorted(p for p in target.rglob("*.py") if p.is_file())
        else:
            continue
        for f in candidates:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            files.append((str(f.relative_to(repo_root)), digest))

    overall = hashlib.sha256(
        "\n".join(f"{p}:{d}" for p, d in files).encode()
    ).hexdigest()
    return {"sha256": overall, "n_files": len(files), "files": dict(files)}


def compare_digests(before: dict | None, after: dict) -> dict:
    """What changed between two :func:`tree_digest` results."""
    if not before:
        return {"comparable": False,
                "note": "no digest recorded at sweep start; cannot verify"}
    old, new = before.get("files", {}), after.get("files", {})
    changed = sorted(p for p in old.keys() & new.keys() if old[p] != new[p])
    return {
        "comparable": True,
        "match": before.get("sha256") == after.get("sha256"),
        "changed": changed,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
    }
