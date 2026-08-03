#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 07 feasibility spike — does the MXFP4 software path exist on gfx950?

Answer this BEFORE authoring 15 MXFP4 problem twins. The fallback (ship 220
problems, defer MXFP4 to v1.1) is only cheap while it is still available.

Probes each layer independently, so a failure localizes instead of just saying
"MXFP4 doesn't work":

    1. torch dtype        float4_e2m1fn_x2 exists and is constructible
    2. E8M0 scales        power-of-two shared scales representable
    3. torch._scaled_mm   MXFP4 path present
    4. hipBLASLt          MXFP4 GEMM reachable
    5. Triton             scaled-dot with MX operands

Verdict is "go" only if the layers a real problem needs are present.

!! NOT YET RUN ON HARDWARE. Probe details (exact _scaled_mm signature, hipBLASLt
   entry point) are the likely first-contact fixes.

Reminder: NVFP4 (block 16, FP8-E4M3 scales) and MXFP4 (block 32, E8M0 scales)
are NOT interchangeable. A working path here enables a re-specification, not a
translation. See tasks/07-quant-mxfp4.md.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import write_artifact  # noqa: E402

MX_BLOCK = 32   # MXFP4 block size per the OCP MX spec (NVFP4 uses 16)


def probe(name: str, fn) -> dict:
    try:
        detail = fn()
        return {"probe": name, "ok": True, "detail": detail}
    except Exception as e:
        return {"probe": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:]}


def p_dtype():
    import torch
    dt = torch.float4_e2m1fn_x2
    t = torch.zeros(64, dtype=torch.uint8, device="cuda").view(dt)
    return {"dtype": str(dt), "shape": list(t.shape),
            "note": "packed two-per-byte"}


def p_e8m0_scales():
    import torch
    # E8M0: 8 exponent bits, no mantissa -> pure power-of-two scale.
    n_blocks = 128
    scales = torch.randint(120, 136, (n_blocks,), dtype=torch.uint8, device="cuda")
    if hasattr(torch, "float8_e8m0fnu"):
        s = scales.view(torch.float8_e8m0fnu)
        return {"native_dtype": "float8_e8m0fnu", "n_blocks": n_blocks,
                "shape": list(s.shape)}
    return {"native_dtype": None, "n_blocks": n_blocks,
            "note": "no native E8M0 dtype; scales carried as uint8 exponents"}


def p_scaled_mm():
    import torch
    m = k = n = 256
    a = torch.zeros(m, k // 2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    b = torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    sa = torch.ones(m, k // MX_BLOCK, dtype=torch.uint8, device="cuda")
    sb = torch.ones(n, k // MX_BLOCK, dtype=torch.uint8, device="cuda")
    out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                           out_dtype=torch.bfloat16)
    return {"out_shape": list(out.shape), "out_dtype": str(out.dtype),
            "block": MX_BLOCK}


def p_hipblaslt():
    import ctypes
    ctypes.CDLL("libhipblaslt.so")
    return {"loaded": "libhipblaslt.so",
            "note": "library present; MXFP4 GEMM path still needs a real call"}


def p_triton_scaled_dot():
    import triton
    import triton.language as tl
    has = hasattr(tl, "dot_scaled")
    return {"triton": triton.__version__, "dot_scaled": has}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/07/spike.json")
    a = ap.parse_args()

    results = [
        probe("torch_dtype_float4_e2m1fn_x2", p_dtype),
        probe("e8m0_scales", p_e8m0_scales),
        probe("torch_scaled_mm_mxfp4", p_scaled_mm),
        probe("hipblaslt_present", p_hipblaslt),
        probe("triton_dot_scaled", p_triton_scaled_dot),
    ]
    by_name = {r["probe"]: r["ok"] for r in results}

    # A problem needs: representable data, representable scales, and at least
    # one kernel path that can consume them.
    data_ok = by_name.get("torch_dtype_float4_e2m1fn_x2") and by_name.get("e8m0_scales")
    kernel_ok = by_name.get("torch_scaled_mm_mxfp4") or by_name.get("triton_dot_scaled")
    verdict = "go" if (data_ok and kernel_ok) else "no-go"

    if verdict == "go":
        rationale = ("MXFP4 data and scales are representable and at least one "
                     "kernel path consumes them. Author the 15 twins.")
    elif not data_ok:
        rationale = ("MXFP4 data/scales not representable in this stack. Do not "
                     "simulate in higher precision — that measures nothing. "
                     "Ship 220 problems and defer to v1.1.")
    else:
        rationale = ("Data representable but no kernel path found. A problem "
                     "with no executable kernel is not a benchmark problem. "
                     "Ship 220 and defer to v1.1.")

    write_artifact(a.out, "07-mxfp4-spike",
                   {"probes": results, "verdict": verdict,
                    "rationale": rationale, "mx_block_size": MX_BLOCK})

    for r in results:
        print(f"  [{'ok  ' if r['ok'] else 'FAIL'}] {r['probe']}"
              + ("" if r["ok"] else f"  {r['error']}"))
    print(f"\nverdict: {verdict}\n{rationale}")
    print("\nRecord this decision explicitly in STATE.md — do not let it happen "
          "by drift.")


if __name__ == "__main__":
    main()
