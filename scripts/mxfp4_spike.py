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
    """MXFP4 through torch._scaled_mm: float4 data, E8M0 scales, block 32.

    Scales are float8_e8m0fnu, not uint8. Passing uint8 makes this fail with a
    dtype complaint that looks like "MXFP4 unsupported" but is really "you
    passed the wrong scale type" -- a false negative on the decision that
    governs 15 problems.
    """
    import torch
    m = k = n = 256
    a = torch.zeros(m, k // 2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    b = torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    sa = torch.full((m, k // MX_BLOCK), 127, dtype=torch.uint8,
                    device="cuda").view(torch.float8_e8m0fnu)
    sb = torch.full((n, k // MX_BLOCK), 127, dtype=torch.uint8,
                    device="cuda").view(torch.float8_e8m0fnu)
    out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                           out_dtype=torch.bfloat16)
    return {"out_shape": list(out.shape), "out_dtype": str(out.dtype),
            "block": MX_BLOCK, "scale_dtype": "float8_e8m0fnu"}


def p_scaled_mm_nvfp4():
    """The NVFP4 shape (block 16, E4M3 scales) for contrast.

    Worth probing precisely because the 15 Quant problems are written for it:
    if this works, they need no respec at all; if it does not, the respec is
    forced rather than chosen.
    """
    import torch
    m = k = n = 256
    a = torch.zeros(m, k // 2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    b = torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    sa = torch.ones(m, k // 16, dtype=torch.float8_e4m3fn, device="cuda")
    sb = torch.ones(n, k // 16, dtype=torch.float8_e4m3fn, device="cuda")
    out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                           out_dtype=torch.bfloat16)
    return {"out_shape": list(out.shape), "block": 16,
            "scale_dtype": "float8_e4m3fn"}


def p_hipblaslt():
    import ctypes
    ctypes.CDLL("libhipblaslt.so")
    return {"loaded": "libhipblaslt.so",
            "note": "library present; MXFP4 GEMM path still needs a real call"}


def p_triton_scaled_dot():
    """COMPILE AND RUN a Triton MXFP4 dot, and check the numbers.

    `hasattr(tl, "dot_scaled")` proves only that the Python binding exists. The
    question this spike exists to answer is whether gfx950 has a working MXFP4
    kernel path, which is decided by the compiler backend, not the frontend --
    so the kernel is compiled, launched, and its result compared against a
    dequantized bf16 reference.
    """
    import torch
    import triton
    import triton.language as tl

    if not hasattr(tl, "dot_scaled"):
        raise RuntimeError("triton.language has no dot_scaled")

    BLOCK_M = BLOCK_N = 32
    BLOCK_K = 64                      # 2 MX blocks of 32 along K

    @triton.jit
    def mxfp4_dot(a_ptr, b_ptr, sa_ptr, sb_ptr, out_ptr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k2 = tl.arange(0, BLOCK_K // 2)     # packed: 2 values per byte
        offs_s = tl.arange(0, BLOCK_K // 32)     # one E8M0 scale per 32

        a = tl.load(a_ptr + offs_m[:, None] * (BLOCK_K // 2) + offs_k2[None, :])
        b = tl.load(b_ptr + offs_n[:, None] * (BLOCK_K // 2) + offs_k2[None, :])
        sa = tl.load(sa_ptr + offs_m[:, None] * (BLOCK_K // 32) + offs_s[None, :])
        sb = tl.load(sb_ptr + offs_n[:, None] * (BLOCK_K // 32) + offs_s[None, :])

        acc = tl.dot_scaled(a, sa, "e2m1", tl.trans(b), tl.trans(sb), "e2m1")
        tl.store(out_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)

    # 0x22 packs two E2M1 values of +1.0; E8M0 127 is a scale of 2^0 = 1.
    a = torch.full((BLOCK_M, BLOCK_K // 2), 0x22, dtype=torch.uint8, device="cuda")
    b = torch.full((BLOCK_N, BLOCK_K // 2), 0x22, dtype=torch.uint8, device="cuda")
    sa = torch.full((BLOCK_M, BLOCK_K // 32), 127, dtype=torch.uint8, device="cuda")
    sb = torch.full((BLOCK_N, BLOCK_K // 32), 127, dtype=torch.uint8, device="cuda")
    out = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")

    mxfp4_dot[(1,)](a, b, sa, sb, out, BLOCK_M, BLOCK_N, BLOCK_K)
    torch.cuda.synchronize()

    # Every element is 1.0, so each output is a sum of BLOCK_K ones.
    expected = float(BLOCK_K)
    got = float(out[0, 0])
    if abs(got - expected) > 1e-3:
        raise RuntimeError(
            f"MXFP4 dot ran but produced {got}, expected {expected}: the path "
            f"exists but is numerically wrong, which is worse than absent"
        )
    return {"triton": triton.__version__, "executed": True,
            "out_00": got, "expected": expected, "block": 32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/07/spike.json")
    a = ap.parse_args()

    results = [
        probe("torch_dtype_float4_e2m1fn_x2", p_dtype),
        probe("e8m0_scales", p_e8m0_scales),
        probe("torch_scaled_mm_mxfp4", p_scaled_mm),
        probe("torch_scaled_mm_nvfp4", p_scaled_mm_nvfp4),
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
