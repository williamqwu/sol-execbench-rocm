import torch, json, importlib
import reference
from chk import tol_ok

dev="cuda:0"; H=2048
import kernel; importlib.reload(kernel)

def timeit(fn, iters=100, warmup=25):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

wl=[json.loads(l) for l in open("workload.jsonl") if l.strip()]
allok=True
tot_m=tot_r=0.0
for i,W in enumerate(wl):
    B=W["axes"]["batch_size"]; S=W["axes"]["seq_len"]
    atol=W["tolerance"]["max_atol"]; rtol=W["tolerance"]["max_rtol"]
    need=W["tolerance"]["required_matched_ratio"]
    ok_all=True
    for seed in (0,1,2):
        torch.manual_seed(seed)
        go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
        r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
        w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
        ra,rw=reference.run(go,r,w)
        ma,mw=kernel.run(go,r,w)
        assert ma.shape==ra.shape and ma.dtype==ra.dtype, (ma.shape,ra.shape,ma.dtype)
        assert mw.shape==rw.shape and mw.dtype==rw.dtype
        assert ma.is_contiguous() and mw.is_contiguous()
        q1,d1=tol_ok(ma,ra,atol,rtol); q2,d2=tol_ok(mw,rw,atol,rtol)
        if q1<need or q2<need:
            ok_all=False
            print(f"  FAIL wl{i} B={B} S={S} seed{seed} ga_ratio={q1:.5f} gw_ratio={q2:.5f} d1={d1:.3f} d2={d2:.3f} atol={atol:.3f}")
    tm=timeit(lambda: kernel.run(go,r,w))
    tr=timeit(lambda: reference.run(go,r,w), iters=20, warmup=5)
    tot_m+=tm; tot_r+=tr
    allok &= ok_all
    print(f"wl{i:2d} B={B:3d} S={S:5d} M={B*S:5d} {'OK ' if ok_all else 'BAD'} mine={tm:7.1f}us ref={tr:8.1f}us speedup={tr/tm:5.2f}x")
print("ALL PASS" if allok else "SOME FAILED")
print(f"total mine={tot_m:.1f}us ref={tot_r:.1f}us  overall={tot_r/tot_m:.2f}x")
