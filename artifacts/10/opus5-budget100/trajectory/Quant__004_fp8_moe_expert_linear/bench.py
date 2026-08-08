import sys, os, math, torch, time
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import reference as R
import importlib
tk = importlib.import_module(os.environ.get("TKMOD", "tk5"))

dev = torch.device("cuda")
torch.manual_seed(0)


def bench(fn, n=20, w=5):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


TOK = [384, 640, 896, 1024, 1152, 1536, 1792, 1920, 2048, 2176, 2432, 2816,
       3072, 3584, 3712, 4096]
sel = [int(x) for x in sys.argv[1:]] or TOK

geo = 0.0
for nt in sel:
    inp = R.get_inputs({"num_tokens": nt}, dev)
    ref = R.run(**inp).float()
    out = tk.moe(**inp).float()
    err = (out - ref).abs()
    bad = (err > 0.005 + 0.02 * ref.abs()).float().mean().item()
    tr = bench(lambda: R.run(**inp))
    tm = bench(lambda: tk.moe(**inp))
    geo += math.log(tr / tm)
    ok = "OK " if bad < 0.01 else "BAD"
    print(f"{nt:5d} {ok} ref {tr:7.3f} mine {tm:7.4f} {tr/tm:6.2f}x  "
          f"maxabs {err.max().item():.4f} unmatched {bad*100:.3f}%", flush=True)
print(f"geomean {math.exp(geo/len(sel)):.3f}x")
