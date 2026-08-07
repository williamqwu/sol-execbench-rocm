import torch, triton, triton.language as tl
from bench import make, bench, H, TOPK
import v_csr as V

for M in [131, 1024, 8192]:
    final, src, idx = make(M)
    N = src.shape[0]; dev = final.device
    B=1024; g=(triton.cdiv(N,B),)
    cnt = torch.zeros(M+1, dtype=torch.int32, device=dev)
    def do_count():
        cnt.zero_(); V._count[g](idx, cnt, N, B)
    t_cnt = bench(do_count, ())
    cnt.zero_(); V._count[g](idx, cnt, N, B)
    off = torch.cumsum(cnt,0,dtype=torch.int32)
    t_cs = bench(lambda: torch.cumsum(cnt,0,dtype=torch.int32), ())
    perm = torch.empty(N, dtype=torch.int32, device=dev)
    def do_sc():
        cur = off[:M].clone(); V._scatter[g](idx, cur, perm, N, B)
    t_sc = bench(do_sc, ())
    do_sc()
    t_zero = bench(lambda: cnt.zero_(), ())
    out = torch.empty_like(final)
    best=None
    for BH,nw in [(1024,4),(512,2),(512,4),(1024,8),(256,2),(2048,8),(1024,2)]:
        if H % BH: continue
        t_g = bench(lambda: V._gather[(M,H//BH)](final,src,perm,off,out,H,BH,num_warps=nw), ())
        print(f"  M={M:5d} BH={BH:5d} nw={nw} gather={t_g:8.1f}")
    print(f"M={M:5d} zero={t_zero:.1f} count={t_cnt:.1f} cumsum={t_cs:.1f} scatter={t_sc:.1f}")
    print()
