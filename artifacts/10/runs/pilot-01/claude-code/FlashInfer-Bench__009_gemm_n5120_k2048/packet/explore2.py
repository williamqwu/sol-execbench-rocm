import torch, time, json
N, K = 5120, 2048
dev = "cuda:0"
B = torch.randn(N, K, device=dev, dtype=torch.float16)

# Measure GPU time using CUDA graph replay to remove CPU dispatch overhead.
def gpu_time_graph(fn, iters=100):
    # warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    ts = []
    for _ in range(5):
        st.record(); g.replay(); en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)

def wall_time(fn, iters=100):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    ts=[]
    for _ in range(5):
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)/iters*1e3)
    return min(ts)

print("== clone (42MB) ==", "graph", round(gpu_time_graph(lambda: B.clone()),2), "wall", round(wall_time(lambda: B.clone()),2))

print(f"{'M':>6} {'graph us':>9} {'wall us':>9} {'GB/s(graph)':>11}")
for M in (1, 2, 8, 16, 32, 64, 128, 172, 289, 492, 952):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    g = gpu_time_graph(lambda: torch.matmul(A, B.T))
    w = wall_time(lambda: torch.matmul(A, B.T))
    tb = N*K*2 + M*K*2 + M*N*2
    print(f"{M:>6} {g:>9.2f} {w:>9.2f} {tb/g*1e-3:>11.0f}")

for M in (8828, 16294):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    g = gpu_time_graph(lambda: torch.matmul(A, B.T), iters=20)
    print(M, "graph", round(g,1), "TFLOPs", round(2*M*N*K/g*1e-6))
