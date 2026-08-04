import torch, time, sys
DEV = "cuda:0"
torch.manual_seed(0)
w = torch.randn(1024, 5120, device=DEV, dtype=torch.bfloat16)


def gt(fn, iters=300):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = max(1, iters // 20)
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (n * 20) * 1e6


for (b, s) in [(1, 128), (8, 128), (16, 128), (64, 128), (1, 8192)]:
    M = b * s
    h = torch.randn(b, s, 5120, device=DEV, dtype=torch.bfloat16)
    h2 = h.view(M, 5120)
    o = torch.empty(M, 1024, device=DEV, dtype=torch.bfloat16)
    ot = torch.empty(b, 8, s, 128, device=DEV, dtype=torch.bfloat16)

    t_mm = gt(lambda: torch.mm(h2, w.t(), out=o))
    t_lin = gt(lambda: torch.nn.functional.linear(h, w))
    src = o.view(b, s, 8, 128).transpose(1, 2)
    t_tr = gt(lambda: ot.copy_(src))
    # pure bandwidth reference: read hidden
    dummy = torch.empty_like(h)
    t_cp = gt(lambda: dummy.copy_(h))
    print(f"B={b:3d} S={s:5d} M={M:6d}  mm={t_mm:7.2f}u linear={t_lin:7.2f}u transpose_copy={t_tr:7.2f}u  copy_h={t_cp:7.2f}u")
    del h, h2, o, ot, dummy
    torch.cuda.empty_cache()

# empty-kernel launch floor inside graph
e = torch.empty(1, device=DEV)
print("noop-ish (add_) floor:", gt(lambda: e.add_(1.0)), "us")
