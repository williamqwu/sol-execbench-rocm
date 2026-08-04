import torch, time, sys, importlib
import reference
import kernel
importlib.reload(kernel)

CASES = [
 (4,256,256, 0.0008066094814807926),
 (8,373,449, 0.0004683943698241626),
 (4,1024,2048, 0.0001047769081813608),
 (64,128,128, 0.0015484569320991571),
 (2,256,512, 0.0004090330504814736),
 (32,691,773, 0.0002761074419450505),
 (8,128,128, 0.001547113477189327),
 (32,512,512, 0.00041390737866093717),
 (4,211,293, 0.0007046196800759221),
 (8,256,256, 0.0008080159826677671),
 (16,128,256, 0.0007898209027846018),
 (1,1024,1024, 0.00020904094400653493),
 (16,256,512, 0.00040882542334228856),
 (32,128,128, 0.0015481723024959578),
 (1,512,512, 0.0004164314483008953),
 (1,4096,4096, 5.2390086450778385e-05),
]
RTOL = 0.0078125
dev = torch.device("cuda:0")

def bench(fn, args, iters=20):
    for _ in range(3): fn(*args)
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(iters): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/iters*1e3

only = sys.argv[1:]
allok=True
for (b,sq,skv,atol) in CASES:
    if only and f"{b},{sq},{skv}" not in only: continue
    torch.manual_seed(0)
    inp = reference.get_inputs({"batch_size":b,"seq_len_q":sq,"seq_len_kv":skv}, dev)
    args = tuple(inp.values())
    ref = reference.run(*args)
    got = kernel.run(*args)
    ok=True; msgs=[]
    for i,(r,g) in enumerate(zip(ref,got)):
        assert r.shape==g.shape, (r.shape,g.shape)
        rf=r.float(); gf=g.float()
        err=(rf-gf).abs()
        thr = atol + RTOL*rf.abs()
        ratio = (err<=thr).float().mean().item()
        me = err.max().item()
        if ratio < 0.99: ok=False
        msgs.append(f"out{i} ratio={ratio:.5f} maxerr={me:.3e}")
    tr = bench(reference.run, args)
    tk = bench(kernel.run, args)
    allok &= ok
    print(f"{'PASS' if ok else 'FAIL'} b={b} q={sq} kv={skv} | {' | '.join(msgs)} | ref={tr:.3f}ms mine={tk:.3f}ms x{tr/tk:.2f}")
    del inp,args,ref,got
    torch.cuda.empty_cache()
print("ALL OK" if allok else "SOME FAILED")
