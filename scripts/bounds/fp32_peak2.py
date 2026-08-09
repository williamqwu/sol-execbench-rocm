"""fp32 matmul rate on MI350X, with the clock sampled DURING the timing.

MAC/cycle is a ratio and the denominator is not assumed: a rate quoted
against F_LOCK when the card is running elsewhere is off by exactly the
clock error, in the direction that flatters whatever it is being compared
to. The clock is read from amdsmi for the physical card torch is on --
resolved through PCI bus identity, because the two index spaces disagree
on this node (scripts/gpu_map.py).
"""
import json, sys, threading, time
import torch

sys.path.insert(0, "/work/scripts")
from gpu_map import torch_to_amdsmi  # noqa: E402

import amdsmi  # noqa: E402

dev = "cuda:0"
amdsmi.amdsmi_init()
handles = amdsmi.amdsmi_get_processor_handles()
h = handles[torch_to_amdsmi()[0]]


def sample_clock(stop, out):
    while not stop.is_set():
        try:
            c = amdsmi.amdsmi_get_clock_info(h, amdsmi.AmdSmiClkType.GFX)
            out.append(c.get("clk") or c.get("cur_clk"))
        except Exception:
            pass
        time.sleep(0.002)


def peak(n, dtype, iters=60, warm=20):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(warm):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    clocks, stop = [], threading.Event()
    t = threading.Thread(target=sample_clock, args=(stop, clocks), daemon=True)
    t.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    s = (time.perf_counter() - t0) / iters
    stop.set(); t.join(timeout=1)
    cl = sorted(c for c in clocks if c)
    mhz = cl[len(cl) // 2] if cl else None
    macs = n ** 3
    r = {"n": n, "seconds": s, "tflops": 2 * macs / s / 1e12,
         "gfx_clk_mhz_median": mhz, "gfx_clk_mhz_max": max(cl) if cl else None,
         "n_clock_samples": len(cl)}
    r["mac_per_cycle_at_1300"] = macs / s / 1300e6
    if mhz:
        r["mac_per_cycle_at_measured_clock"] = macs / s / (mhz * 1e6)
    return r


out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
       "table_fp32_sm": 32768, "table_bf16_tc": 524288, "modes": {}}
for label, tf32 in (("fp32_tf32_off", False), ("fp32_tf32_on", True)):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    out["modes"][label] = {"allow_tf32": tf32, **peak(8192, torch.float32)}
torch.backends.cuda.matmul.allow_tf32 = False
out["modes"]["bf16_control"] = peak(8192, torch.bfloat16)
out["modes"]["fp32_vector_fma_control"] = None
print(json.dumps(out, indent=2))
amdsmi.amdsmi_shut_down()
