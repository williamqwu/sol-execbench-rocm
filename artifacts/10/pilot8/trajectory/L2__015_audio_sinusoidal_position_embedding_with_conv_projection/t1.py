import torch, torch.nn.functional as F, time, math
import reference as R, tk
dev=torch.device('cuda')
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

for (B,T) in [(32,4328),(1,3000),(16,2048),(64,128),(2,256)]:
    d=R.get_inputs({"batch_size":B,"time_dim":T},dev)
    x0=d['input_features']
    w1=d['conv2d1_weight'].reshape(384,9).contiguous()
    w2=d['conv2d2_weight'].permute(2,3,1,0).contiguous()
    w3=d['conv2d3_weight'].permute(2,3,1,0).contiguous()
    ref1=F.gelu(F.conv2d(x0,d['conv2d1_weight'],d['conv2d1_bias'],stride=2,padding=1))
    y1,T2=tk.conv1(x0,w1,d['conv2d1_bias'])
    e1=(y1.permute(0,3,1,2).float()-ref1.float()).abs().max().item()
    ref2=F.gelu(F.conv2d(ref1,d['conv2d2_weight'],d['conv2d2_bias'],stride=2,padding=1))
    cfg=(64,128,64,4,2)
    y2,F2,T3=tk.convn(y1,w2,d['conv2d2_bias'],40,T2,0,B,cfg)
    e2=(y2.permute(0,3,1,2).float()-ref2.float()).abs().max().item()
    ref3=F.gelu(F.conv2d(ref2,d['conv2d3_weight'],d['conv2d3_bias'],stride=2,padding=1))
    y3,F3,T4=tk.convn(y2,w3,d['conv2d3_bias'],F2,T3,1,B,cfg)
    rf=ref3.permute(0,3,1,2).reshape(B,T4,3840)
    e3=(y3.float()-rf.float()).abs().max().item()
    print(f"B={B} T={T} err {e1:.4f} {e2:.4f} {e3:.4f} | ref3 absmax {ref3.abs().max().item():.2f}")
    print("  c1", bench(lambda: tk.conv1(x0,w1,d['conv2d1_bias'])), "c2", bench(lambda: tk.convn(y1,w2,d['conv2d2_bias'],40,T2,0,B,cfg)), "c3", bench(lambda: tk.convn(y2,w3,d['conv2d3_bias'],F2,T3,1,B,cfg)))
