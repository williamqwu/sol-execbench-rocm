import torch, time, itertools, sys

torch.manual_seed(0)
dev = "cuda:0"
H = 2048

import reference

def timeit(fn, args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    # use cuda events
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0  # us

SHAPES = [(64,128),(16,128),(2,4096),(4,512),(16,256),(4,1024),(8,1024),(8,128),
          (1,256),(1,1024),(4,256),(32,256),(8,256),(4,128),(2,1024),(32,128)]

def mk(B,S):
    go = torch.randn(B,S,H, device=dev, dtype=torch.bfloat16)
    r  = torch.randn(B,S,H, device=dev, dtype=torch.bfloat16)
    w  = torch.randn(H,H, device=dev, dtype=torch.bfloat16)
    return go,r,w

@torch.no_grad()
def torch_bf16(go, r, w):
    B,S,_ = go.shape
    g2 = go.reshape(-1,H); r2 = r.reshape(-1,H)
    gw = g2.t() @ r2
    gr = g2 @ w
    ga = gr.view(B,S,32,64).permute(0,2,1,3).contiguous()
    return ga, gw

if __name__ == "__main__":
    Ms = sorted(set(B*S for B,S in SHAPES))
    print("M values:", Ms)
    print(f"{'B,S':>10} {'M':>6} {'ref us':>10} {'bf16 us':>10} {'gemm1':>8} {'gemm2':>8} {'perm':>8}")
    for B,S in SHAPES:
        go,r,w = mk(B,S)
        tr = timeit(reference.run, (go,r,w), iters=20)
        tb = timeit(torch_bf16, (go,r,w), iters=50)
        g2 = go.reshape(-1,H); r2 = r.reshape(-1,H)
        t1 = timeit(lambda a,b: a.t()@b, (g2,r2), iters=50)
        t2 = timeit(lambda a,b: a@b, (g2,w), iters=50)
        gr = g2@w
        t3 = timeit(lambda x: x.view(B,S,32,64).permute(0,2,1,3).contiguous(), (gr,), iters=50)
        print(f"{B:4d},{S:5d} {B*S:6d} {tr:10.1f} {tb:10.1f} {t1:8.1f} {t2:8.1f} {t3:8.1f}")
        # accuracy
        ra, rw = reference.run(go,r,w)
        ma, mw = torch_bf16(go,r,w)
        for nm,(x,y) in (("ga",(ma,ra)),("gw",(mw,rw))):
            d = (x.float()-y.float()).abs()
            tol = 0.117 + 0.0078125*y.float().abs()
            bad = (d>tol).float().mean().item()
            print(f"      {nm}: maxabs={d.max().item():.4f} badfrac={bad:.5f} refmax={y.float().abs().max().item():.1f}")
