import torch, triton
import tk, tf, reference
from chk import tol_ok

dev="cuda:0"; H=2048
torch.manual_seed(0)

def timeit(fn, iters=100, warmup=25):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

SH=[(1,256),(4,128),(8,128),(4,512),(16,256),(64,128)]
GW_CFG={256:(256,128,64,8,8,2),512:(128,128,64,8,8,2),1024:(128,128,128,8,8,2),
        2048:(128,128,64,8,8,3),4096:(128,128,64,8,8,3),8192:(128,128,64,8,8,3)}
GA_CFG={256:(128,64,64,8,4,2),512:(32,128,128,8,4,2),1024:(64,64,64,8,4,2),
        2048:(128,128,64,8,8,3),4096:(128,128,64,8,8,2),8192:(128,128,64,8,8,2)}
FU_CFG={256:(64,64,128,8,4,2),512:(64,128,64,8,4,2),1024:(128,256,64,8,8,2),
        2048:(128,256,64,8,8,2),4096:(128,256,64,8,8,2),8192:(128,256,64,8,8,2)}

print(f"{'M':>6} {'torchAll':>9} {'tGW+myGA':>9} {'myGW+myGA':>10} {'fused':>8} | {'tGW':>7} {'myGW':>7} {'tGA':>7} {'myGA':>7}")
for B,S in SH:
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    g2=go.reshape(-1,H); r2=r.reshape(-1,H); M=B*S
    gwc=GW_CFG[M]; gac=GA_CFG[M]; fuc=FU_CFG[M]

    def torch_all():
        gw=g2.t()@r2
        ga=(g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous()
        return ga,gw
    def mix():
        gw=g2.t()@r2
        ga=tk.ga_run(g2,w,B,S,gac)
        return ga,gw
    def allmine():
        gw=tk.gw_run(g2,r2,gwc)
        ga=tk.ga_run(g2,w,B,S,gac)
        return ga,gw

    t_all=timeit(torch_all); t_mix=timeit(mix); t_mine=timeit(allmine)
    t_fu=timeit(lambda: tf.fused_run(g2,r2,w,B,S,fuc))
    t_tgw=timeit(lambda: g2.t()@r2)
    t_mgw=timeit(lambda: tk.gw_run(g2,r2,gwc))
    t_tga=timeit(lambda: (g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous())
    t_mga=timeit(lambda: tk.ga_run(g2,w,B,S,gac))
    print(f"{M:6d} {t_all:9.1f} {t_mix:9.1f} {t_mine:10.1f} {t_fu:8.1f} | {t_tgw:7.1f} {t_mgw:7.1f} {t_tga:7.1f} {t_mga:7.1f}", flush=True)
