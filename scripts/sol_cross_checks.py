#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 03 cross-checks — is T_SOL a bound, and is it the RIGHT bound?

A roofline bound is the easiest number in the project to get wrong without
noticing: a units slip, a stale clock, a precision that resolved to the wrong
peak, and the output is still a plausible-looking millisecond figure that
rescales every score by a constant. So the bound is attacked from three
directions that do not share an implementation with the SOLAR bridge:

  A  DECLARED TRAFFIC.  Every problem's own definition states the shape and
     dtype of each input and output. Their total is the traffic any correct
     kernel must move at least once. If SOLAR's memory term is BELOW that, its
     "bound" is below the unavoidable minimum and is not a bound.

  B  IMPLIED RATES.  T_SOL implies a bandwidth (bytes / time) and a throughput
     (2 * MACs / time). Neither may exceed the arch config's own peak, and a
     memory-bound workload's implied bandwidth should sit AT the DRAM peak --
     that is what "memory bound" means. A clock or unit error shows up here as
     a rate that is impossible or absurdly low.

  C  HAND-DERIVED MACs.  For eighteen problems whose arithmetic is
     unambiguous -- eleven single matmuls and seven pure memory kernels -- the
     MAC count is written out by hand from the reference and compared. This is
     the check that catches a graph extracted from the wrong shapes.

  D  T_SOL <= best measured.  Requires task 06. Reported as pending until
     `--t-b` is supplied. T_SOL is re-derived at the clock bracket the anchor
     measurement itself recorded -- never read off the stored reference-clock
     column, which is what published 120 phantom violations (D63).

Every count below is a claim about specific input files, so the report records
what it was generated from -- manifest, T_SOL tiers, T_b tree, arch config, each
with a digest -- in a machine-readable comment near the top. `verify_artifacts.py`
refuses to read the A-published count out of a report that record does not bind
to the manifest under audit. Before that, the gate passed for any manifest.

Upstream's B200 SOL times are NOT a source here: the shipped dataset carries no
per-workload SOL figures, and a comparison number invented for the occasion
would be worse than no comparison at all. Checks A-C are internal and stronger.

    python scripts/sol_cross_checks.py --out artifacts/03/cross-checks.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402

# The bound library, for section D. Imported at module level and guarded,
# because if it is unavailable section D must say so and evaluate NOTHING --
# falling back to the stored `t_sol_ms` column is the D63 read this whole file
# now exists to stop doing.
try:                                                       # pragma: no cover
    from sol_execbench.core.bench.clock_bracket import clock_interval
    from solexbench_rocm.t_sol_at import (
        MissingBoundTerms, MissingReferenceClock, bound_ms, t_sol_interval,
    )
except ImportError:                                        # pragma: no cover
    clock_interval = None

    class MissingBoundTerms(KeyError):
        pass

    class MissingReferenceClock(KeyError):
        pass

    def bound_ms(w):                                       # noqa: D103
        raise MissingReferenceClock("solexbench_rocm.t_sol_at unavailable")

    def t_sol_interval(w, a, b):                           # noqa: D103
        raise MissingBoundTerms("solexbench_rocm.t_sol_at unavailable")


#: Marker for the machine-readable record of what this report was generated
#: FROM. `verify_artifacts.py` refuses to read a count out of this report
#: unless the record names the manifest it is gating -- see `input_bindings`.
INPUTS_MARKER = "sol-cross-checks-inputs"

DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "float4_e2m1fn_x2": 1,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
}

# -- C: hand-derived MAC counts -------------------------------------------
# Each lambda takes the resolved axis dict and returns MACs. Written from the
# reference source, by hand, on purpose: an automatically derived count would
# share the failure modes of the thing it is auditing.
#
# A GEMM of (M,K) x (K,N) is M*N*K MACs. A pure memory kernel is 0, and 0 is a
# real prediction -- SOLAR reporting MACs for one means it found arithmetic
# that is not there.
HAND_MACS = {
    # C = A @ B.T, A:(M,K) B:(N,K)
    **{f"FlashInfer-Bench__{i:03d}_gemm_{tag}":
       (lambda ax, n=n, k=k: ax["M"] * n * k)
       for i, tag, n, k in [
           (4, "n128_k2048", 128, 2048),
           (5, "n256_k7168", 256, 7168),
           (6, "n2048_k4096", 2048, 4096),
           (7, "n4096_k4096", 4096, 4096),
           (8, "n4096_k14336", 4096, 14336),
           (9, "n5120_k2048", 5120, 2048),
           (10, "n6144_k4096", 6144, 4096),
           (11, "n28672_k4096", 28672, 4096),
       ]},
    # logits = hidden[:, -keep:] @ weight.T
    "L1__003_lm_head_projection_with_logit_slicing":
        lambda ax: ax["batch_size"] * ax["logits_to_keep"] * ax["hidden_size"]
        * ax["vocab_size"],
    # logits = hidden @ weight.T, over the whole sequence
    "L1__077_whisper_decoder_output_projection":
        lambda ax: ax["batch_size"] * ax["seq_len"] * ax["d_model"]
        * ax["vocab_size"],
    # both streams projected by one (hidden_dim, hidden_dim) weight
    "L2__030_flux_concatenated_sequence_processing_with_split":
        lambda ax: ax["batch_size"] * (ax["img_seq_len"] + ax["text_seq_len"])
        * ax["hidden_dim"] * ax["hidden_dim"],
    # pure memory kernels: no matrix arithmetic at all
    "FlashInfer-Bench__001_fused_add_rmsnorm_h2048": lambda ax: 0,
    "FlashInfer-Bench__002_fused_add_rmsnorm_h4096": lambda ax: 0,
    "FlashInfer-Bench__003_fused_add_rmsnorm_h7168": lambda ax: 0,
    "L1__025_video_latent_gelu_activation": lambda ax: 0,
    "L1__046_attention_softmax_with_softcapping_and_dropout": lambda ax: 0,
    "L1__069_rms_norm": lambda ax: 0,
    "L1__085_geglu_activation": lambda ax: 0,
}


def resolved_axes(definition: dict, workload_axes: dict) -> dict:
    """Every axis value: the workload's own, plus the definition's constants.

    A workload record carries only the axes that vary. `N` and `K` on a GEMM
    are declared `const` in the definition and never appear in the workload, so
    resolving from the workload alone leaves most shapes unresolvable -- it cut
    check A from 2998 workloads to 44.
    """
    axes = {}
    for name, spec in (definition.get("axes") or {}).items():
        if isinstance(spec, dict) and spec.get("value") is not None:
            axes[name] = spec["value"]
    axes.update(workload_axes or {})

    # `expr` axes are derived from the others -- `q_out_features` is
    # `num_attention_heads * head_dim`, and shapes reference the derived name,
    # not the expression. Resolving iteratively because an expression may
    # depend on another expression. Anything still unresolved after a full
    # pass with no progress is left out, and the caller declines to make a
    # claim about that workload rather than guessing a dimension.
    pending = {name: spec["expression"]
               for name, spec in (definition.get("axes") or {}).items()
               if isinstance(spec, dict) and spec.get("type") == "expr"
               and spec.get("expression") and name not in axes}
    while pending:
        progressed = False
        for name, expr in list(pending.items()):
            try:
                # Arithmetic over already-resolved axis names only: no
                # builtins, no attribute access, nothing the dataset could use
                # to run code here.
                value = eval(expr, {"__builtins__": {}}, dict(axes))  # noqa: S307
            except Exception:                              # noqa: BLE001
                continue
            axes[name] = value
            del pending[name]
            progressed = True
        if not progressed:
            break
    return axes


def declared_traffic(definition: dict, axes: dict) -> int | None:
    """Bytes of the problem's own declared inputs and outputs.

    The minimum traffic of any correct kernel: every input read once, every
    output written once. Scalars (shape None) are excluded -- they ride in a
    kernel argument, not through DRAM.
    """
    total = 0
    for group in ("inputs", "outputs"):
        for spec in (definition.get(group) or {}).values():
            shape = spec.get("shape")
            if not shape:
                continue
            n = 1
            for dim in shape:
                if isinstance(dim, int):
                    n *= dim
                elif dim in axes:
                    n *= axes[dim]
                elif str(dim).isdigit():
                    # Some shapes carry a literal as a string: ['1', 'seq'].
                    n *= int(dim)
                else:
                    return None            # unresolved symbol: no claim made
            width = DTYPE_BYTES.get(spec.get("dtype"))
            if width is None:
                return None
            total += n * width
    return total


def definition_of(data_root: str, key: str) -> dict:
    category, name = key.split("__", 1)
    return json.loads((Path(data_root) / category / name /
                       "definition.json").read_text())


def _sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_binding(path) -> dict:
    """Identity of one input file: its path, its size, its mtime, its sha256.

    `present: False` for a path that was not supplied or does not exist, which
    is a *statement* -- "this report was generated without that input" -- and is
    what lets a gate refuse instead of assuming.
    """
    if not path:
        return {"path": None, "present": False}
    p = Path(path)
    if not p.is_file():
        return {"path": str(path), "present": False}
    st = p.stat()
    return {
        "path": str(path),
        "abspath": str(p.resolve()),
        "present": True,
        "bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime,
                                            timezone.utc).isoformat(),
        "sha256": _sha256_of_file(p),
        "digest_kind": "sha256-of-contents",
    }


def dir_binding(path, pattern: str = "*.json") -> dict:
    """Identity of an input DIRECTORY -- the T_b tree.

    Digested over each file's `(name, size, mtime_ns)` rather than its bytes:
    `artifacts/06-MI355X/authoritative-merged` is 235 files and hundreds of
    megabytes, and re-reading all of it on every report would make the report
    expensive enough to skip. That is a weaker digest than sha256-of-contents
    and it says so in `digest_kind`, so nobody reads more into it than it
    carries: it detects a different tree, a re-measured tree and a re-merged
    tree, and it does not detect an edit that preserves size and mtime.
    """
    if not path:
        return {"path": None, "present": False}
    p = Path(path)
    if not p.is_dir():
        return {"path": str(path), "present": False}
    files = sorted(f for f in p.glob(pattern) if f.is_file())
    h = hashlib.sha256()
    total = 0
    for f in files:
        st = f.stat()
        total += st.st_size
        h.update(f"{f.name}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return {
        "path": str(path),
        "abspath": str(p.resolve()),
        "present": True,
        "n_files": len(files),
        "bytes": total,
        "sha256": h.hexdigest(),
        "digest_kind": "sha256-of-name+size+mtime-per-file",
    }


def input_bindings(a) -> dict:
    """What this report was generated FROM, as a machine-readable record.

    **Why a report has to carry this.** Every count below is a claim about
    specific input files, and the report used to name none of them. Task 03's
    gate then read the A-published count out of whichever `cross-checks.md`
    happened to be on disk and compared it against whichever manifest was passed
    to `verify_artifacts.py --manifest`: measured on this tree, a report
    generated against `manifest-v4.json` produced `[PASS] check A-published`
    while `--manifest manifest-v1.json` and `--manifest manifest-v2.json` were
    under audit -- and check D, which reads the manifest directly, failed on 54
    of 1801 and 10 of 2078 respectively. A gate that cannot fail must not be
    able to pass either, so the gate now refuses a report it cannot bind to the
    manifest in front of it.

    The digest is the binding, not the path: a manifest rebuilt in place at the
    same path is a different manifest, and a report generated against the old
    one is stale evidence about the new one.
    """
    return {
        "manifest": file_binding(a.manifest),
        "t_sol": file_binding(a.t_sol),
        "t_sol_traffic": file_binding(a.t_sol_traffic),
        "t_b": dir_binding(a.t_b),
        "arch": file_binding(a.arch),
        # Path only: the dataset is thousands of files, it is not an artifact of
        # this repo, and no count here is read out of it by a gate.
        "data": {"path": str(a.data), "digest_kind": "path-only"},
    }


def load_arch(path: Path) -> dict:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = float(v.strip().strip('"'))
        except ValueError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-sol", default="artifacts/03/t_sol.json")
    ap.add_argument("--t-sol-traffic", default=None,
                    help="declared-traffic tier, e.g. artifacts/03-MI355X/"
                         "t_sol_traffic.json. Supplies the declared byte "
                         "totals for A-published; defaults to a t_sol_traffic"
                         ".json beside --t-sol.")
    ap.add_argument("--arch", default="SOLAR/configs/arch/MI350X.yaml")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--t-b", default=None,
                    help="artifacts/06 directory; enables check D")
    ap.add_argument("--manifest", default=None,
                    help="scoring manifest, e.g. artifacts/09-MI355X/"
                         "manifest-v2.json. Enables D-published, which audits "
                         "max(SOLAR, traffic) after the manifest's own tier "
                         "rejection -- the bound a score is computed against. "
                         "Without it only the SOLAR tier is audited.")
    ap.add_argument("--out", default="artifacts/03/cross-checks.md")
    a = ap.parse_args()
    if a.t_sol_traffic is None:
        sibling = Path(a.t_sol).with_name("t_sol_traffic.json")
        a.t_sol_traffic = str(sibling) if sibling.is_file() else None

    doc = json.loads(Path(a.t_sol).read_text())
    arch = load_arch(Path(a.arch))
    freq_hz = arch["freq_GHz"] * 1e9
    dram_bw = arch["DRAM_byte_per_cycle"] * freq_hz
    peak_flops = {k.split("_")[3]: v * 2 * freq_hz
                  for k, v in arch.items() if k.startswith("MAC_per_cycle_")}

    problems = doc.get("problems", doc)

    # -- A + B ------------------------------------------------------------
    a_checked = a_violation = 0
    a_worst: list[tuple] = []
    b_checked = b_bw_violation = b_flops_violation = 0
    b_clocked = b_unclocked = 0
    b_worst_bw: list[tuple] = []
    unresolved = 0
    declared_by: dict[tuple[str, str], int] = {}

    for key, entry in sorted(problems.items()):
        category, name = key.split("__", 1)
        defn_path = Path(a.data) / category / name / "definition.json"
        if not defn_path.exists():
            continue
        definition = json.loads(defn_path.read_text())
        precision = entry.get("precision")
        for uuid, w in (entry.get("workloads") or {}).items():
            if w.get("t_sol_cycles") is None:
                continue
            axes = resolved_axes(definition, w.get("axes") or {})
            declared = declared_traffic(definition, axes)
            solar_bytes = w.get("memory_bytes")
            if declared is None or not solar_bytes:
                unresolved += 1
            else:
                declared_by[(key, uuid)] = declared
                a_checked += 1
                ratio = solar_bytes / declared
                if ratio < 0.999:                # below the unavoidable minimum
                    a_violation += 1
                    a_worst.append((ratio, key, uuid, declared, solar_bytes))

            # Compare in TIME, not by re-deriving time from cycles at the arch
            # clock. `t_sol_cycles` is expressed at the bound's own reference
            # clock -- the sibling field is literally `memory_cycles_at_f_ref`,
            # and on MI355X f_ref is 1.8 GHz while the arch config declares
            # 2.4. Dividing by the arch clock therefore inflated every
            # memory-bound workload's implied bandwidth by exactly 2.4/1.8 =
            # 1.3333, and this check reported 1327 workloads "over DRAM peak"
            # when not one of them was: every reading was 1.33x, the same
            # number, which is the signature of a constant, not of 1327
            # independent bad bounds.
            #
            # `t_sol_ms` is the bound in time and is what the manifest
            # publishes. Verified against the invariant: for a memory-bound
            # workload, memory_bytes / t_sol_ms lands on DRAM peak exactly.
            #
            # Read through `t_sol_at.bound_ms` -- the choke point -- rather than
            # off the record, so that a record which does not say what clock its
            # column is on is COUNTED as such instead of being read anyway.
            # Today every record in `artifacts/03-MI355X/t_sol.json` is
            # unstamped, so the count is the whole population and the check
            # keeps its coverage; when the tier is re-derived with `f_ref_mhz`
            # the count goes to zero and the check becomes a clock-attributed
            # one without any other change here. Falling back is deliberate and
            # is not a silent one: an unstamped column still catches a units
            # slip, which is what B is for, and dropping those records would
            # take this check from 2998 workloads to 0 -- a gate that cannot
            # fail, which is worse than one reading an ambiguous column in the
            # open.
            try:
                t_ms = bound_ms(w)
                b_clocked += 1
            except MissingReferenceClock:
                t_ms = w.get("t_sol_ms")
                b_unclocked += 1
            except KeyError:
                t_ms = None
            seconds = (t_ms / 1000.0) if t_ms else (w["t_sol_cycles"] / freq_hz)
            if seconds <= 0:
                continue
            b_checked += 1
            implied_bw = (solar_bytes or 0) / seconds
            implied_flops = 2 * (w.get("macs") or 0) / seconds
            if implied_bw > dram_bw * 1.001:
                b_bw_violation += 1
            peak = peak_flops.get(precision or "")
            if peak and implied_flops > peak * 1.001:
                b_flops_violation += 1
            if w.get("bottleneck") in ("memory", "dram", "DRAM"):
                b_worst_bw.append((implied_bw / dram_bw, key, uuid))

    # -- C ----------------------------------------------------------------
    c_rows: list[tuple] = []
    for key, fn in sorted(HAND_MACS.items()):
        entry = problems.get(key)
        if not entry:
            c_rows.append((key, None, None, "no T_SOL recorded"))
            continue
        for uuid, w in sorted((entry.get("workloads") or {}).items()):
            if w.get("t_sol_cycles") is None:
                continue
            try:
                expect = fn(resolved_axes(definition_of(a.data, key),
                                          w.get("axes") or {}))
            except KeyError as e:
                c_rows.append((key, None, None, f"axis missing: {e}"))
                break
            got = w.get("macs") or 0
            verdict = ("exact" if got == expect else
                       "within 1%" if expect and
                       abs(got - expect) / max(expect, 1) < 0.01 else
                       "MISMATCH" if expect or got else "exact")
            c_rows.append((key, expect, got, verdict))
            break                                  # one workload per problem

    c_bad = sum(1 for _, _, _, v in c_rows if v == "MISMATCH")

    # -- D ----------------------------------------------------------------
    # T_SOL is re-derived at the clock bracket the ANCHOR MEASUREMENT recorded,
    # exactly as `verify_artifacts._bound_for` does for the submissions. It used
    # to read `w["t_sol_ms"]` -- the stored reference-clock column -- and on
    # MI355X that column is written at 1.8 GHz while the anchors were measured
    # near 2.4, so this section published "**120 VIOLATIONS**, each one a config
    # error" for the length of the D63 correction that removed them. Re-derived
    # at each winner's own bracket the same 2694 workloads give 0; at a flat
    # 2400 MHz they also give 0. The 120 were clock arithmetic, all on the same
    # 13 problems the D63 tier fix moved (L1__035, L1__037, L1__054, L2__002,
    # L2__019 among them), and the only gate that had been red because of that
    # number went green in the same session -- so the claim went on being
    # published with nothing watching it.
    #
    # A winner with no usable bracket, or a T_SOL record with no terms, is
    # counted as NOT CHECKABLE and named in the status line. It is not silently
    # compared against the stored column: that is the read this correction
    # exists to retire, and a fallback would quietly reintroduce it for exactly
    # the records least able to survive it.
    d_rows: list[tuple] = []
    d_status = "PENDING — needs task 06 (`--t-b artifacts/06/authoritative`)"
    if a.t_b and clock_interval is None:
        d_status = ("NOT EVALUATED — `solexbench_rocm.t_sol_at` is not "
                    "importable here, and this section will not fall back to "
                    "the stored reference-clock column to produce a number")
    elif a.t_b:
        tb_dir = Path(a.t_b)
        checked = viol = no_bracket = no_terms = 0
        for f in sorted(tb_dir.glob("*.json")):
            tb = json.loads(f.read_text())
            entry = problems.get(tb.get("problem", ""), {})
            for uuid, win in (tb.get("winner_by_workload") or {}).items():
                w = (entry.get("workloads") or {}).get(uuid)
                # Bounded-ness is the cycles column, not the millisecond one:
                # `t_sol_ms` is the ambiguous column this section stopped
                # reading, and it should not go on deciding what gets read.
                if not w or w.get("t_sol_cycles") is None:
                    continue
                span = clock_interval(win)
                if span is None:
                    no_bracket += 1
                    continue
                try:
                    t_sol = t_sol_interval(w, *span)["t_sol_ms_published"]
                except (MissingBoundTerms, ValueError):
                    no_terms += 1
                    continue
                checked += 1
                if t_sol > win["t_b_ms"] * 1.0001:
                    viol += 1
                    d_rows.append((tb["problem"], uuid, t_sol,
                                   win["t_b_ms"], win["variant"]))
        d_status = (
            f"{checked - viol}/{checked} workloads satisfy T_SOL <= T_b, with "
            f"T_SOL re-derived at each anchor's own clock bracket (minimum-clock "
            f"end, the tightest); {no_bracket} anchors carry no usable bracket "
            f"and {no_terms} T_SOL records carry no separable terms, so those "
            f"are NOT CHECKABLE rather than compared against the stored "
            f"reference-clock column"
            + ("" if not viol else
               f" — **{viol} VIOLATIONS**, each one a config error"))

    # -- D as PUBLISHED ----------------------------------------------------
    # Everything above audits ONE tier: the SOLAR bound in --t-sol. That is not
    # what a score is computed against. The manifest publishes max(SOLAR,
    # declared-traffic) and, crucially, REJECTS a tier that exceeds the
    # measured T_b (`solar_rejected_above_t_b`, `traffic_rejected_above_t_b`).
    # So the tier-level count overstates the shipped damage -- 148 here against
    # 41 in the manifest -- and a gate reading the tier count fails for
    # something nobody publishes, the same way task 06's gate audited a T_b
    # tree nobody published.
    #
    # The published bound is READ rather than recomputed. Re-deriving the
    # max-and-reject rule here would duplicate build_manifest's selection and
    # the two would drift, at which point the cross-check and the manifest
    # would disagree with no way to tell which is right.
    d_pub_status = ("not evaluated — pass --manifest to audit the bound that is "
                    "actually published, not just the SOLAR tier")
    d_pub_rows: list[tuple] = []
    if a.manifest and Path(a.manifest).is_file():
        man = json.loads(Path(a.manifest).read_text())
        pchecked = pviol = 0
        for key, entry in (man.get("problems") or {}).items():
            wls = entry.get("workloads") or {}
            for uuid, w in (wls.items() if isinstance(wls, dict) else []):
                if not isinstance(w, dict) or not w.get("scoreable"):
                    continue
                ts = w.get("t_sol_ms_published") or w.get("t_sol_ms")
                tb = w.get("t_b_ms")
                if ts is None or tb is None:
                    continue
                pchecked += 1
                if ts > tb * 1.0001:
                    pviol += 1
                    d_pub_rows.append((key, uuid, ts, tb,
                                       w.get("t_sol_source")))
        d_pub_status = (
            f"{pchecked - pviol}/{pchecked} PUBLISHED workloads satisfy "
            f"T_SOL <= T_b" + ("" if not pviol else
            f" — **{pviol} VIOLATIONS across "
            f"{len({r[0] for r in d_pub_rows})} problems**. Scores on those "
            f"problems are not results."))

    # -- A as PUBLISHED ----------------------------------------------------
    # Check A above audits ONE tier and is red by construction: 1021 of 2998
    # SOLAR memory terms sit below the declared minimum, and on 1000 of them
    # the manifest never publishes that number -- either the traffic tier wins
    # the max, or SOLAR's *compute* term already puts the fused bound above the
    # traffic floor. An always-red check cannot gate anything, so A stays a
    # judgement item and this is the gate: does the bound a score is ACTUALLY
    # computed against sit at or above the problem's own declared traffic?
    #
    # The one sanctioned exception, and the reason this is not simply
    # `max(SOLAR, floor)`: where a definition declares a tensor the kernel
    # INDEXES rather than streams -- a 131072-position KV cache -- the declared
    # total is not a floor at all, and the measurement says so, because the
    # floor lands ABOVE the measured T_b. A floor above a time that was
    # actually achieved is refuted, not violated. Those are counted separately
    # and named, never silently dropped: the excuse is a measurement, so if T_b
    # is missing there is no excuse and the workload counts as a violation.
    #
    # Direction matters, and it is CONDITIONAL. This comment used to state it
    # as though it were not ("too small for any submission slower than T_b,
    # which is most of them, so S is INFLATED"), the emitted violation line
    # below inherited the unqualified form, and so did
    # `docs/issues/mi355x-bound-quality.md` ("S is inflated for everyone") --
    # while this same file's check-A report body says the opposite of all three
    # ("understated, not inflated ... the safe direction"). Differentiating the
    # score settles it:
    #
    #     S = (T_b - T_SOL) / (T_b + T_k - 2*T_SOL)
    #     dS/dT_SOL = (T_b - T_k) / (T_b + T_k - 2*T_SOL)^2
    #
    # so a published T_SOL below the true floor DEFLATES S for a submission
    # FASTER than T_b and inflates it only for one SLOWER than T_b. "Most of
    # them" was wrong on this corpus: measured over `artifacts/10`, 1549 of
    # 2078 PASSED MI355X records (74.5%) are faster than their own T_b, so the
    # majority effect of a too-small bound is deflation.
    #
    # It is still a gate, and the reason is unchanged and is not the sign of
    # dS: a bound below the unavoidable minimum is not a bound, and NOTHING
    # downstream can see it -- unlike T_SOL > T_b, which check D-published
    # catches by comparison with a real measurement.
    #
    # Know what this check cannot see. The floor is READ from the traffic
    # tier's own `memory_bytes` (below), so it moves whenever that tier's byte
    # count moves -- the D18 gather correction (264 workloads) and the causal
    # mask (64) each lowered the published bound and its floor together, and
    # this check reported the same counts before and after by construction. It
    # gates the published bound against the traffic tier; it does not audit the
    # traffic tier.
    a_pub_status = ("not evaluated — pass --manifest to audit the published "
                    "bound against the declared-traffic floor")
    a_pub_rows: list[tuple] = []
    a_pub_refuted: list[tuple] = []
    if a.manifest and Path(a.manifest).is_file():
        man_a = json.loads(Path(a.manifest).read_text())
        # The traffic tier's `memory_bytes` IS the declared total, by
        # construction, so it is read rather than re-derived -- and its
        # `rejected` list carries the declared bytes for the workloads it
        # dropped, which are precisely the ones this check must still see.
        traffic_path = Path(a.t_sol_traffic) if a.t_sol_traffic else None
        if traffic_path and traffic_path.is_file():
            tdoc = json.loads(traffic_path.read_text())
            for key, entry in (tdoc.get("problems") or {}).items():
                for uuid, w in (entry.get("workloads") or {}).items():
                    if w.get("memory_bytes"):
                        declared_by[(key, uuid)] = w["memory_bytes"]
            for r in (tdoc.get("rejected") or []):
                if r.get("declared_bytes"):
                    declared_by[(r["problem"], r["workload"])] = \
                        r["declared_bytes"]
        pchecked = pviol = prefuted = punknown = 0
        for key, entry in (man_a.get("problems") or {}).items():
            wls = entry.get("workloads") or {}
            for uuid, w in (wls.items() if isinstance(wls, dict) else []):
                if not isinstance(w, dict) or not w.get("scoreable"):
                    continue
                declared = declared_by.get((key, uuid))
                ts = w.get("t_sol_ms_published") or w.get("t_sol_ms")
                if declared is None or ts is None:
                    punknown += 1
                    continue
                # Clock-invariant: bytes over a bandwidth fixed in bytes/second.
                floor_ms = declared / dram_bw * 1e3
                pchecked += 1
                if ts >= floor_ms * 0.999:
                    continue
                tb = w.get("t_b_ms")
                if tb is not None and floor_ms > tb:
                    prefuted += 1
                    a_pub_refuted.append((key, uuid, floor_ms, tb))
                else:
                    pviol += 1
                    a_pub_rows.append((floor_ms / ts, key, uuid, declared,
                                       ts, floor_ms, w.get("t_sol_source")))
        # The zero is STATED, not implied by the absence of a clause. While it
        # was implied, "no VIOLATIONS string in this section" and "no
        # A-published section at all" were the same observation, and
        # `verify_artifacts.py`'s unanchored search for that string ran 3123
        # characters past this section into section D and gated on D's count --
        # reporting "120 published bounds below the floor" against a report
        # whose own A-published line says 3688/3717 sit at or above it. A
        # reader is owed what the parser is owed: the number this section
        # found, whatever it is.
        a_pub_status = (
            f"{pchecked - pviol - prefuted}/{pchecked} PUBLISHED workloads "
            f"sit at or above their declared-traffic floor; {prefuted} have a "
            f"floor refuted by measurement (floor > T_b); {punknown} not "
            f"checkable" + (" — **0 VIOLATIONS**." if not pviol else
            f" — **{pviol} VIOLATIONS across "
            f"{len({r[1] for r in a_pub_rows})} problems**. Those bounds are "
            f"below the unavoidable minimum, so they are not bounds: S is "
            f"deflated for every submission faster than its T_b and inflated "
            f"for every one slower, and no measurement can detect either."))

    # -- report -----------------------------------------------------------
    bindings = input_bindings(a)
    man_b = bindings["manifest"]
    man_line = (
        f"Generated against manifest `{man_b['path']}` "
        f"(sha256 `{man_b['sha256'][:16]}`, {man_b['bytes']:,} bytes, mtime "
        f"{man_b['mtime_utc']}). Every count in this section is a claim about "
        f"THAT file and about no other."
        if man_b.get("present") else
        "Generated with **no --manifest**, so this section makes no claim about "
        "any published bound.")
    L = [
        "# Task 03 — T_SOL cross-checks",
        "",
        f"<!-- {json.dumps(stamp('03-cross-checks')['_provenance'], default=str)} -->",
        "",
        # Machine-readable, and load-bearing: verify_artifacts.py refuses to
        # read the A-published count out of this report unless this record
        # binds it to the manifest under audit.
        f"<!-- {INPUTS_MARKER} {json.dumps(bindings, default=str)} -->",
        "",
        "Upstream's B200 SOL times are not used as a comparison anywhere in "
        "this document. The shipped dataset carries no per-workload SOL "
        "figures, so there is nothing to compare against that was not invented "
        "here — and an invented comparison would be worse than none. The three "
        "checks below are internal to this platform and are stronger for it.",
        "",
        "## A — SOLAR's memory term vs the problem's own declared traffic",
        "",
        "Every definition states the shape and dtype of each input and output. "
        "Their sum is what any correct kernel must move at least once. A "
        "memory term below it is not a bound.",
        "",
        f"* checked: **{a_checked}** workloads",
        f"* below declared minimum: **{a_violation}**",
        f"* not checkable (unresolved symbol or dtype): {unresolved}",
        "",
    ]
    if a_worst:
        by_problem: dict[str, list] = {}
        for r, key, _u, dec, sol in a_worst:
            by_problem.setdefault(key, []).append((r, dec, sol))
        L += [
            "The shortfall is concentrated: "
            f"**{len(by_problem)} problems**, not a scatter across the set. "
            "Two mechanisms produce it, and only one of them is benign.",
            "",
            "*Benign* — a preallocated cache or table that the kernel indexes "
            "rather than streams. `L1/018` declares a KV cache of 131072 "
            "positions and touches one sequence's worth of it, so the declared "
            "total is not a floor for that kernel and the ratio near zero is "
            "expected.",
            "",
            "*Not benign* — a graph SOLAR traced incompletely, which shows up "
            "as a missing weight matrix. Where the largest declared tensor is "
            "a weight and the ratio is ~0.001, SOLAR did not see the matmul "
            "that consumes it.",
            "",
            "**Direction of the error.** A T_SOL below the true bound makes "
            "`(T_b − T_SOL)` too large, so scores computed against it are "
            "*understated*, not inflated. That is the safe direction — no "
            "kernel is flattered by it — but it is still wrong, and these "
            "problems carry the ratio into the manifest so a consumer can see "
            "which bounds are loose.",
            "",
            "| worst ratio | workloads | problem | declared bytes | SOLAR bytes |",
            "|---|---|---|---|---|",
        ]
        for key, rows in sorted(by_problem.items(), key=lambda kv: min(kv[1])[0]):
            r, dec, sol = min(rows)
            L.append(f"| {r:.4f} | {len(rows)} | {key} | {dec:,} | {sol:,} |")
        L.append("")

    L += [
        "## A-published — the bound a score is computed against, vs that floor",
        "",
        "**This is the gate; A above is a judgement item.** A alone is red by "
        "construction and cannot gate: on 1000 of its 1021 the SOLAR memory "
        "term never reaches a score, because either the traffic tier wins the "
        "`max` or SOLAR's compute term already lifts the fused bound above "
        "the floor. What can be gated is the published number.",
        "",
        "A floor that lands ABOVE the measured T_b is refuted rather than "
        "violated — the kernel demonstrably moved less than the definition "
        "declares, which is what an indexed cache looks like. Refuted rows are "
        "counted and listed, and a missing T_b is no excuse: without a "
        "measurement to refute it, the floor stands.",
        "",
        man_line,
        "",
        a_pub_status,
        "",
    ]
    if a_pub_rows:
        L += ["| floor / published | problem | workload | declared bytes | "
              "published T_SOL ms | floor ms | tier |",
              "|---|---|---|---|---|---|---|"]
        for r, key, u, dec, ts, fl, src in sorted(a_pub_rows, reverse=True)[:25]:
            L.append(f"| {r:.3f}x | {key} | `{u[:8]}` | {dec:,} | {ts:.6g} | "
                     f"{fl:.6g} | {src} |")
        L.append("")
    if a_pub_refuted:
        by_p: dict[str, int] = {}
        for key, _u, _f, _t in a_pub_refuted:
            by_p[key] = by_p.get(key, 0) + 1
        L += ["Floors refuted by measurement, by problem: "
              + ", ".join(f"`{k}` ({n})" for k, n in sorted(by_p.items())), ""]

    L += [
        "## B — rates implied by T_SOL",
        "",
        f"Arch config: DRAM {dram_bw/1e12:.2f} TB/s at {arch['freq_GHz']} GHz.",
        "",
        f"* checked: **{b_checked}** workloads",
        f"* implied bandwidth above DRAM peak: **{b_bw_violation}**",
        f"* implied FLOPS above the precision's peak: **{b_flops_violation}**",
        f"* read through `t_sol_at.bound_ms` at the record's own stated clock: "
        f"**{b_clocked}**",
        f"* read off an unstamped `t_sol_ms` column (no `f_ref_mhz`, so the "
        f"clock is the file's word and not the record's): **{b_unclocked}**",
        "",
        "## C — hand-derived MAC counts",
        "",
        "Eleven single matmuls and seven pure memory kernels, counted by hand "
        "from the reference source. A pure memory kernel's expected count is "
        "zero, and zero is a real prediction: MACs reported for one would mean "
        "SOLAR found arithmetic that is not in the kernel.",
        "",
        "| problem | hand-derived MACs | SOLAR MACs | verdict |",
        "|---|---|---|---|",
    ]
    for key, expect, got, verdict in c_rows:
        e = f"{expect:,}" if isinstance(expect, int) else "—"
        g = f"{got:,}" if isinstance(got, int) else "—"
        L.append(f"| {key} | {e} | {g} | {verdict} |")
    L += ["", f"MISMATCHes: **{c_bad}**", "",
          "## D — T_SOL <= best measured time", "", d_status, ""]
    if d_rows:
        L += ["| problem | workload | T_SOL ms | T_b ms | variant |",
              "|---|---|---|---|---|"]
        for p, u, ts, tb, v in d_rows[:25]:
            L.append(f"| {p} | `{u[:8]}` | {ts:.6g} | {tb:.6g} | {v} |")
        L.append("")

    L += ["## D-published — the bound a score is actually computed against", "",
          "Section D above audits the SOLAR tier alone. The manifest publishes",
          "max(SOLAR, declared-traffic) and rejects a tier that exceeds the",
          "measured T_b, so the tier count overstates the shipped damage.", "",
          d_pub_status, ""]
    if d_pub_rows:
        L += ["| problem | workload | T_SOL ms | T_b ms | bound tier |",
              "|---|---|---|---|---|"]
        for p, u, ts, tb, src in sorted(d_pub_rows, key=lambda r: -r[2] / r[3])[:25]:
            L.append(f"| {p} | `{u[:8]}` | {ts:.6g} | {tb:.6g} | {src} |")
        L.append("")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    print(f"  A  {a_checked} checked, {a_violation} below declared minimum "
          f"(judgement — see A-published for the gate)")
    print(f"  A-pub  {a_pub_status}")
    print(f"  D-pub  {d_pub_status}")
    print(f"  B  {b_checked} checked, {b_bw_violation} over BW peak, "
          f"{b_flops_violation} over FLOPS peak")
    print(f"  C  {len(c_rows)} rows, {c_bad} mismatches")
    print(f"  D  {d_status}")
    # A is NOT in the exit condition, and dropping it is the point of
    # A-published. A counts SOLAR memory terms below the declared minimum; 1000
    # of the 1021 on MI355X are never published, so a gate on A is red on every
    # run and stops meaning anything. A-published counts the bounds that a
    # score is actually computed against, which is the thing that can be wrong.
    return 1 if (len(a_pub_rows) or b_bw_violation or b_flops_violation
                 or c_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
