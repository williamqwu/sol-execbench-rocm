import torch, triton
from bench import make, bench, H
import v2 as V, v_csr as C

for M in [131, 1024, 8192]:
    final, src, idx = make(M)
    N = src.shape[0]; dev=final.device
    B=1024; g=(triton.cdiv(N,B),)
    cnt = torch.zeros(M+1, dtype=torch.int32, device=dev)
    V._count[g](idx, cnt, N, B)
    M1=M+1; SB=triton.next_power_of_2(M1)
    off_t = torch.empty(M1, dtype=torch.int32, device=dev)
    for nw in [4,8,16]:
        try:
            t=bench(lambda: V._scan[(1,)](cnt, off_t, M1, SB, num_warps=nw), ())
            print(f"  M={M} scan nw={nw} SB={SB}: {t:.1f}")
        except Exception as ex: print("  scan fail", nw, str(ex)[:60])
    t=bench(lambda: torch.cumsum(cnt,0,dtype=torch.int32), ())
    print(f"  M={M} torch.cumsum: {t:.1f}")
    off = torch.cumsum(cnt,0,dtype=torch.int32)
    cur=off[:M].clone(); perm=torch.empty(N,dtype=torch.int32,device=dev)
    V._scatter[g](idx,cur,perm,N,B)
    out=torch.empty_like(final)
    t=bench(lambda: C._gather[(M,H//1024)](final,src,perm,off,out,H,1024,num_warps=2), ())
    print(f"  M={M} gather1d BH1024 nw2: {t:.1f}")
    for KB in [2,4,8,16]:
        for BH,nw in [(1024,4),(1024,2),(512,2)]:
            t=bench(lambda: V._gather2d[(M,H//BH)](final,src,perm,off,out,H,BH,KB,num_warps=nw), ())
            print(f"  M={M} gather2d KB={KB} BH={BH} nw={nw}: {t:.1f}")
    print()
