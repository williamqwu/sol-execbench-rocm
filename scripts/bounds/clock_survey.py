"""Clock held by a sample of real agent kernels, violating and not.

If the clock is a property of the workload's power draw rather than of the
problems that happened to be beaten, then a non-violating fp32 problem must
clock just as high as a violating one -- its bound is equally inflated and
simply was not reached. That is the control.
"""
import json, os, sys, threading, time
from pathlib import Path
import torch

sys.path.insert(0, "/work/scripts")
from gpu_map import torch_to_amdsmi
import amdsmi

sys.path.insert(0, "/work/scripts/runners")
from _common import exec_reference, prepare_inputs, load_problem  # noqa: E402

amdsmi.amdsmi_init()
h = amdsmi.amdsmi_get_processor_handles()[torch_to_amdsmi()[0]]
BASE = "/work/data/SOL-ExecBench/benchmark"
KDIR = "/work/artifacts/10/glm-sweep-2/kernels"


def clock_during(fn, seconds=3.0):
    clocks, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                c = amdsmi.amdsmi_get_clock_info(h, amdsmi.AmdSmiClkType.GFX)
                v = c.get("clk") or c.get("cur_clk")
                if v:
                    clocks.append(v)
            except Exception:
                pass
            time.sleep(0.002)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = threading.Thread(target=sampler, daemon=True); t.start()
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < seconds:
        fn(); n += 1
    torch.cuda.synchronize()
    stop.set(); t.join(timeout=1)
    cl = sorted(clocks)
    if not cl:
        return None
    return {"iters": n, "clk_median": cl[len(cl)//2],
            "clk_p10": cl[len(cl)//10], "clk_p90": cl[9*len(cl)//10],
            "samples": len(cl)}


CASES = [
    ("L2__073_feedforward_mlp_backward",                       "violating", "float32"),
    ("L2__068_gelu_approximate_feedforward_backward",          "violating", "float32"),
    ("L2__035_convnextv2_block_with_grn",                      "violating", "float32"),
    ("L2__036_convnextv2_layer_with_nhwc_persistence_backward","control",   "?"),
    ("L1__074_fused_gated_mlp_silu",                           "control",   "?"),
    ("L1__002_vae_conv3x3_groupnorm_silu_residual_fused",      "control",   "?"),
    ("L1__005_conv_gated_projection_with_causal_conv",         "violating", "bfloat16"),
]

out = {}
for key, kind, dt in CASES:
    kf = os.path.join(KDIR, key + ".py")
    cat, name = key.split("__", 1)
    pdir = os.path.join(BASE, cat, name)
    if not (os.path.exists(kf) and os.path.exists(pdir)):
        out[key] = {"skipped": "no kernel or no problem dir"}
        continue
    try:
        definition, workloads = load_problem(Path(pdir))
        run_ref, ns = exec_reference(definition)
        w = max(workloads, key=lambda x: json.dumps(getattr(x, "axes", {}) or {}))
        inputs = prepare_inputs(definition, w, ns)
        src = open(kf).read()
        kns = {}
        exec(compile(src, key, "exec"), kns)
        run = kns["run"]
        call = (lambda: run(**inputs)) if isinstance(inputs, dict) else (lambda: run(*inputs))
        dtypes = sorted({str(v.dtype) for v in (inputs.values() if isinstance(inputs, dict) else inputs)
                         if hasattr(v, "dtype")})
        r = clock_during(call)
        out[key] = {"kind": kind, "input_dtypes": dtypes, **(r or {})}
    except Exception as e:
        out[key] = {"kind": kind, "error": f"{type(e).__name__}: {e}"[:160]}
    print(json.dumps({key: out[key]}), flush=True)

print("\nSUMMARY")
print(json.dumps(out, indent=2))
amdsmi.amdsmi_shut_down()
