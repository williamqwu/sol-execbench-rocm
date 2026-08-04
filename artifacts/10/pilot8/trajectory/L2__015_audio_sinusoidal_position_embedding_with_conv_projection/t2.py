import torch, torch.nn.functional as F, time, math, itertools
import reference as R, tk
dev=torch.device('cuda')
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

cfgs=[(bm,0,bk,nw,ns) for bm in [32,64,128] for bk in [64,128,384] for nw in [4,8] for ns in [1,2]]+[(64,128,64,4,1)]
for (B,T) in [(32,4328),(1,3000),(64,128)]:
    d=R.get_inputs({"batch_size":B,"time_dim":T},dev)
    x0=d['input_features']
    w1=d['conv2d1_weight'].reshape(384,9).contiguous()
    w2=d['conv2d2_weight'].permute(2,3,1,0).contiguous()
    w3=d['conv2d3_weight'].permute(2,3,1,0).contiguous()
    y1,T2=tk.conv1(x0,w1,d['conv2d1_bias'])
    res2=[];res3=[]
    for c in cfgs:
        try:
            y2,F2,T3=tk.convn(y1,w2,d['conv2d2_bias'],40,T2,0,B,c)
            res2.append((bench(lambda: tk.convn(y1,w2,d['conv2d2_bias'],40,T2,0,B,c),10),c))
            res3.append((bench(lambda: tk.convn(y2,w3,d['conv2d3_bias'],F2,T3,1,B,c),10),c))
        except Exception as e: pass
    res2.sort();res3.sort()
    print(f"B={B} T={T}")
    print("  c2:", res2[:5])
    print("  c3:", res3[:5])
