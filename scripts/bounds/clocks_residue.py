"""Sustained clock of the five kernels whose bounds v1.1 does not explain.

Last time three of these would not load: the kernel source was exec'd from a
string and triton's @jit refuses a function with no file behind it. They are
written to a real file here, which is also how the harness runs them, so the
three that were missing are the point of this run.
"""
import importlib.util, json, os, sys, tempfile, threading, time
import torch

sys.path.insert(0, "/work/scripts")
from gpu_map import torch_to_amdsmi
sys.path.insert(0, "/work/scripts/runners")
from _common import exec_reference, prepare_inputs, load_problem
import amdsmi
from pathlib import Path

amdsmi.amdsmi_init()
H = amdsmi.amdsmi_get_processor_handles()[torch_to_amdsmi()[0]]
BASE = Path("/work/data/SOL-ExecBench/benchmark")
KDIR = Path("/work/artifacts/10/glm-sweep-2/kernels")

CASES = ["L1__005_conv_gated_projection_with_causal_conv",
         "L1__006_hyena_depthwise_conv1d_split_gate",
         "L1__054_audio_attention_qkv_projection_with_normalization",
         "L1__057_mtp_shifted_embedding_with_dual_rms_norm_fusion",
         "L2__045_audio_encoder_to_language_model_multimodal_fusion"]


def load_kernel(key):
    """Import the kernel from a real file, the way the harness does."""
    src = (KDIR / f"{key}.py").read_text()
    d = tempfile.mkdtemp(prefix="diag_", dir="/var/tmp/solbench/diag")
    f = Path(d) / "kernel.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(f"k_{abs(hash(key))}", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.run


def clock_during(fn, hold=6.0):
    clocks, stop = [], threading.Event()

    def samp():
        while not stop.is_set():
            c = amdsmi.amdsmi_get_clock_info(H, amdsmi.AmdSmiClkType.GFX)
            v = c.get("clk") or c.get("cur_clk")
            if v:
                clocks.append(v)
            time.sleep(0.002)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = threading.Thread(target=samp, daemon=True); t.start()
    t0 = time.perf_counter(); n = 0
    # Bounded queue: sync every 50 so the hold is wall time, not a backlog.
    while time.perf_counter() - t0 < hold:
        for _ in range(50):
            fn()
        torch.cuda.synchronize(); n += 50
    stop.set(); t.join(timeout=1)
    cl = sorted(clocks)
    if not cl:
        return None
    return {"iters": n, "per_iter_ms": 1e3 * (time.perf_counter()-t0)/n,
            "clk_median": cl[len(cl)//2], "clk_p10": cl[len(cl)//10],
            "clk_p90": cl[9*len(cl)//10], "samples": len(cl)}


out = {}
for key in CASES:
    cat, name = key.split("__", 1)
    try:
        definition, workloads = load_problem(BASE / cat / name)
        run = load_kernel(key)
        run_ref, ns = exec_reference(definition)
        # Largest workload: the one most likely to be at the roofline.
        w = max(workloads, key=lambda x: sum(
            v for v in (getattr(x, "axes", {}) or {}).values() if isinstance(v, int)))
        inputs = prepare_inputs(definition, w, ns)
        call = ((lambda: run(**inputs)) if isinstance(inputs, dict)
                else (lambda: run(*inputs)))
        dts = sorted({str(v.dtype) for v in
                      (inputs.values() if isinstance(inputs, dict) else inputs)
                      if hasattr(v, "dtype")})
        out[key] = {"dtypes": dts, "axes": getattr(w, "axes", None),
                    **(clock_during(call) or {})}
    except Exception as e:
        out[key] = {"error": f"{type(e).__name__}: {e}"[:200]}
    print(json.dumps({key: out[key]}), flush=True)

print("\nRESULT " + json.dumps(out, indent=2))
amdsmi.amdsmi_shut_down()
