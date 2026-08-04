import torch, time, json
import reference as R, kernel as K, triton
names = ["grad_output","token_indices","selected_tokens","w1_output","gate_output","up_output","gated_output","expert_output","selected_weights","w1_weight","w2_weight","w3_weight"]
dev=torch.device("cuda:0")
def bench(f,n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
for M in (64,256,2048):
    ax=dict(batch_seq_len=4096,num_tokens=M,hidden_dim=4096,ffn_dim=14336)
    torch.manual_seed(1); inp=R.get_inputs(ax,dev)
    go,ti,st,w1o,gao,upo,gdo,eo,sw,w1,w2,w3 = [inp[n] for n in names]
    B,H=go.shape; F=w1.shape[0]; bf=torch.bfloat16
    gwo=torch.empty((M,H),device=dev,dtype=bf); geo=torch.empty((M,H),device=dev,dtype=bf); gsw=torch.empty(M,device=dev,dtype=torch.float32)
    def pre():
        K._kpre[(M,)](go,ti,eo,sw,gwo,geo,gsw,M,H,go.stride(0),go.stride(1),eo.stride(0),eo.stride(1),BH=1024,num_warps=8)
    g1=torch.empty((M,F),device=dev,dtype=bf); gu=torch.empty((M,F),device=dev,dtype=bf)
    ca=K._cfg_a(M); ga=(triton.cdiv(M,ca["BM"])*triton.cdiv(F,ca["BN"]),)
    def ka():
        K._ka[ga](gwo,w2,sw,upo,gao,w1o,gu,g1,M,F,H,gwo.stride(0),gwo.stride(1),w2.stride(0),w2.stride(1),g1.stride(0),g1.stride(1),BM=ca["BM"],BN=ca["BN"],BK=ca["BK"],GM=ca["GM"],num_warps=ca["num_warps"],num_stages=ca["num_stages"])
    cb=K._cfg_b(M); gst=torch.zeros((M,H),device=dev,dtype=torch.float32)
    gb=(triton.cdiv(M,cb["BM"])*triton.cdiv(H,cb["BN"]),cb["SPLIT_K"])
    def kb():
        K._kb[gb](g1,gu,w1,w3,gst,M,H,F,g1.stride(0),g1.stride(1),w1.stride(0),w1.stride(1),gst.stride(0),gst.stride(1),BM=cb["BM"],BN=cb["BN"],BK=cb["BK"],SPLIT_K=cb["SPLIT_K"],GM=cb["GM"],num_warps=cb["num_warps"],num_stages=cb["num_stages"])
    ghs=torch.zeros((B,H),device=dev,dtype=bf)
    print(f"M={M}")
    print(f"  pre  {bench(pre):8.1f}us")
    print(f"  kA   {bench(ka):8.1f}us")
    print(f"  mmw2 {bench(lambda: torch.mm(geo.t(),gdo)):8.1f}us")
    print(f"  mmw3 {bench(lambda: torch.mm(gu.t(),st)):8.1f}us")
    print(f"  mmw1 {bench(lambda: torch.mm(g1.t(),st)):8.1f}us")
    print(f"  kB   {bench(kb):8.1f}us")
    print(f"  scat {bench(lambda: ghs.index_copy_(0,ti,gst.to(bf))):8.1f}us")
    print(f"  zero {bench(lambda: torch.zeros((B,H),device=dev,dtype=bf)):8.1f}us")
    print(f"  full {bench(lambda: K.run(*[inp[n] for n in names])):8.1f}us")
