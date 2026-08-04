"""Prototype a low-overhead launch path: cache CompiledKernel + stream, skip JIT dispatch."""
import time
import torch, triton
import triton.language as tl
import kernel as K


def cpu_only(fn, iters=2000):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t) / iters * 1e6
    torch.cuda.synchronize()
    return dt


def wall(fn, iters=2000):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


B, S, D = 1, 512, 256
u = torch.randn((B, 768, S), device="cuda")
w = torch.randn((768, 1, 3), device="cuda")
bi = torch.randn((768,), device="cuda")

print(f"{'variant':40s} {'cpu_us':>9s} {'wall_us':>9s}")
print(f"{'current run()':40s} {cpu_only(lambda: K.run(u,w,bi)):9.2f} {wall(lambda: K.run(u,w,bi)):9.2f}")

# --- build a pre-compiled launcher ---
from triton.runtime import driver

blk, nw, nsb, even = K._cfg(S)
DS = D * S
out = torch.empty((3, B, D, S), device="cuda")

ck = K._hyena_fused.warmup(
    u, w, bi, out, S, DS, B * DS, 3 * DS, 0,
    BLOCK_S=blk, EVEN=even, num_warps=nw, num_stages=1,
    grid=(nsb, D, B),
)
ck._init_handles()
print("compiled ok; n_regs", ck.n_regs, "shared", ck.metadata.shared)

dev = driver.active.get_current_device()
stream = driver.active.get_current_stream(dev)
run_fn = ck.run
fn_h = ck.function
pmeta = ck.packed_metadata
from triton import knobs
eh, xh = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
print("hooks:", eh, xh)


def fast():
    o = torch.empty((3, B, D, S), device="cuda")
    run_fn(nsb, D, B, stream, fn_h, pmeta, None, eh, xh,
           u, w, bi, o, S, DS, B * DS, 3 * DS, 0)
    return o[0], o[1], o[2]


try:
    r = fast()
    torch.cuda.synchronize()
    ref = K.run(u, w, bi)
    print("fast path matches:", all(torch.equal(a, b) for a, b in zip(r, ref)))
    print(f"{'fast precompiled launch':40s} {cpu_only(fast):9.2f} {wall(fast):9.2f}")
except Exception as e:
    import traceback; traceback.print_exc()

# how much is the alloc + slicing?
def alloc_only():
    o = torch.empty((3, B, D, S), device="cuda")
    return o[0], o[1], o[2]
print(f"{'alloc+3 slices only':40s} {cpu_only(alloc_only):9.2f}")

def launch_only():
    run_fn(nsb, D, B, stream, fn_h, pmeta, None, eh, xh,
           u, w, bi, out, S, DS, B*DS, 3*DS, 0)
print(f"{'launch only (no alloc)':40s} {cpu_only(launch_only):9.2f} {wall(launch_only):9.2f}")
