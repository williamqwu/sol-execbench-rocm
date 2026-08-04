import torch, time
dev=torch.device("cuda:0"); bf=torch.bfloat16
H,F=4096,14336
def bench(f,n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for M in (64,256,512,2048):
    a=torch.randn(M,F,device=dev,dtype=bf); s=torch.randn(M,H,device=dev,dtype=bf)
    w=torch.randn(F,H,device=dev,dtype=bf); w2=torch.randn(H,F,device=dev,dtype=bf)
    t_dw = bench(lambda: torch.mm(a.t(), s))         # (F,M)@(M,H) -> F,H
    t_dw2= bench(lambda: torch.mm(s.t(), a))         # (H,M)@(M,F) -> H,F
    t_ga = bench(lambda: torch.mm(s, w2))            # (M,H)@(H,F)
    t_gb = bench(lambda: torch.mm(a, w))             # (M,F)@(F,H)
    print(f"M={M:5d} gradW(F,H)={t_dw:7.3f} gradW2(H,F)={t_dw2:7.3f} A:(M,H)@(H,F)={t_ga:7.3f} B:(M,F)@(F,H)={t_gb:7.3f}")
# pure bandwidth reference
x=torch.empty(F*H,device=dev,dtype=bf); y=torch.empty(F*H,device=dev,dtype=bf)
t=bench(lambda: y.copy_(x))
print(f"copy 117MB r+w: {t*1e3:.1f}us -> {2*x.numel()*2/t*1e-9:.0f} GB/s")
t=bench(lambda: x.zero_())
print(f"zero 117MB w  : {t*1e3:.1f}us -> {x.numel()*2/t*1e-9:.0f} GB/s")
