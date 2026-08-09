"""What clock does the card actually hold while a violating kernel runs?

T_SOL_ms = t_sol_cycles / F_LOCK. F_LOCK is 1300 MHz, measured in task 01.
If a given kernel runs at a different clock, its bound in milliseconds is
wrong by exactly that ratio -- and in the direction that makes the bound
beatable, if the kernel clocks higher than the calibration load did.
"""
import json, sys, threading, time
import torch

sys.path.insert(0, "/work/scripts")
from gpu_map import torch_to_amdsmi
import amdsmi

amdsmi.amdsmi_init()
h = amdsmi.amdsmi_get_processor_handles()[torch_to_amdsmi()[0]]


def clock_during(fn, seconds=4.0):
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

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = threading.Thread(target=sampler, daemon=True); t.start()
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < seconds:
        fn(); n += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    stop.set(); t.join(timeout=1)
    cl = sorted(clocks)
    return {"iters": n, "per_iter_ms": 1e3 * dt / n,
            "clk_p10": cl[len(cl)//10], "clk_median": cl[len(cl)//2],
            "clk_p90": cl[9*len(cl)//10], "samples": len(cl)}


out = {"note": "median GFX clock sampled every 2 ms while the workload loops"}

# 1. the calibration-style load: dense bf16 matmul, matrix cores, power hungry
a = torch.randn(8192, 8192, device="cuda:0", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda:0", dtype=torch.bfloat16)
out["bf16_matmul_8192"] = clock_during(lambda: torch.matmul(a, b))

# 2. dense fp32 matmul, the datapath the violating problems are priced against
af = torch.randn(8192, 8192, device="cuda:0", dtype=torch.float32)
bf = torch.randn(8192, 8192, device="cuda:0", dtype=torch.float32)
out["fp32_matmul_8192"] = clock_during(lambda: torch.matmul(af, bf))

# 3. the real thing: L2__073's own kernel, on its own largest shape
sys.path.insert(0, "/var/tmp/solbench/bounds")
src = open("/work/artifacts/10/glm-sweep-2/kernels/"
           "L2__073_feedforward_mlp_backward.py").read()
ns = {}
exec(compile(src, "kernel", "exec"), ns)
run = ns["run"]
B, S, H, I = 1, 8192, 2048, 8192
g = lambda *s: torch.randn(*s, device="cuda:0", dtype=torch.float32)
args = (g(B, S, H), g(B, S, H), g(I, H), g(I), g(H, I), g(H),
        g(B, S, I), g(B, S, I))
out["L2__073_kernel"] = clock_during(lambda: run(*args))

print(json.dumps(out, indent=2))
amdsmi.amdsmi_shut_down()
