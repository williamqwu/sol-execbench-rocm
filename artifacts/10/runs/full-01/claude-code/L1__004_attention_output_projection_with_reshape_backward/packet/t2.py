import torch, triton
import tk

dev="cuda:0"; H=2048
torch.manual_seed(0)

def timeit(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

SH=[(1,256),(4,128),(8,128),(4,512),(16,256),(64,128)]

GA_CFGS=[(128,128,64,8,8,2),(128,256,64,8,8,2),(256,128,64,8,8,2),
         (64,128,64,8,4,2),(128,128,128,8,8,2),(64,256,64,8,4,2),
         (32,128,64,8,4,2),(128,64,64,8,4,2),(64,64,64,8,4,2),
         (32,64,64,8,4,2),(32,256,64,8,4,2),(16,128,64,8,4,2),
         (64,128,128,8,4,2),(32,128,128,8,4,2),(64,128,64,8,8,2),
         (128,128,64,8,8,3),(64,256,128,8,8,2)]

print("=== GA (go @ w -> permuted (B,32,S,64)) ===")
for B,S in SH:
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    g2=go.reshape(-1,H); M=B*S
    ref=(g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous()
    tt=timeit(lambda: (g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous())
    best=[]
    for cfg in GA_CFGS:
        try:
            out=tk.ga_run(g2,w,B,S,cfg)
            err=(out.float()-ref.float()).abs().max().item()
            if err>0.01:
                print("   ERR",cfg,err); continue
            t=timeit(lambda: tk.ga_run(g2,w,B,S,cfg))
            best.append((t,cfg))
        except Exception as e:
            print("   EX",cfg,str(e)[:80])
    best.sort()
    print(f" M={M:5d} torch={tt:8.1f} " + " | ".join(f"{t:.1f}{cfg}" for t,cfg in best[:4]))
