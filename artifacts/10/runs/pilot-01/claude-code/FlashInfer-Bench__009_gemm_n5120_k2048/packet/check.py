import json, torch, importlib, sys
import kernel
importlib.reload(kernel)

N, K = 5120, 2048
dev = "cuda:0"
wls = [json.loads(l) for l in open("workload.jsonl")]

def gpu_time(fn, iters=50):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True); ts=[]
    for _ in range(3):
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)/iters*1e3)
    return min(ts)

allpass = True
tot_mine = tot_ref = 0.0
print(f"{'M':>6} {'ok':>4} {'maxerr':>9} {'atol':>8} {'ratio':>6} {'mine':>8} {'ref':>8} {'spd':>5}")
for w in sorted(wls, key=lambda x: x["axes"]["M"]):
    M = w["axes"]["M"]
    tol = w["tolerance"]
    atol, rtol = tol["max_atol"], tol["max_rtol"]
    need = tol["required_matched_ratio"]
    torch.manual_seed(M)
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    Bt = torch.randn(N, K, device=dev, dtype=torch.float16)
    out = kernel.run(A, Bt)
    ref = torch.matmul(A, Bt.T)
    assert out.shape == ref.shape and out.dtype == ref.dtype, (out.shape, out.dtype)
    d = (out.float() - ref.float()).abs()
    thr = atol + rtol * ref.float().abs()
    matched = (d <= thr).float().mean().item()
    mx = d.max().item()
    ok = matched >= need
    allpass &= ok
    it = 50 if M < 2000 else 10
    tm = gpu_time(lambda: kernel.run(A, Bt), iters=it)
    tr = gpu_time(lambda: torch.matmul(A, Bt.T), iters=it)
    tot_mine += tm; tot_ref += tr
    print(f"{M:>6} {'OK' if ok else 'FAIL':>4} {mx:>9.4f} {atol:>8.4f} {matched:>6.3f} {tm:>8.2f} {tr:>8.2f} {tr/tm:>5.2f}x")

print("\nALL PASS" if allpass else "\nSOME FAILED")
print(f"total mine {tot_mine:.1f}us  ref {tot_ref:.1f}us  speedup {tot_ref/tot_mine:.3f}x")
