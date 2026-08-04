import torch, sys, importlib
sys.path.insert(0, '/var/tmp/solbench/agent/pilot8/L2__069_joint_transformer_block_residual_path')
import reference as R

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else 'kernel_v1')
dev = torch.device('cuda')


def bench(fn, iters=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


shapes = [(1, 1024, 77), (1, 2048, 77), (32, 256, 77), (64, 128, 77), (1, 2053, 77),
          (1, 8192, 77), (2, 256, 154), (2, 2048, 77), (1, 293, 77), (8, 256, 77),
          (4, 512, 77), (16, 512, 77), (1, 128, 154), (4, 1024, 77), (2, 4096, 77),
          (4, 2048, 77)]
tolmap = {(1,1024,77):0.02465,(1,2048,77):0.02234,(32,256,77):0.01913,(64,128,77):0.01758,
          (1,2053,77):0.02234,(1,8192,77):0.02291,(2,256,154):0.02002,(2,2048,77):0.02436,
          (1,293,77):0.02307,(8,256,77):0.01872,(4,512,77):0.01954,(16,512,77):0.01951,
          (1,128,154):0.01831,(4,1024,77):0.02171,(2,4096,77):0.02067,(4,2048,77):0.01987}
import math
tot = 0.0
for sh in shapes:
    b, s_, c = sh
    inp = R.get_inputs({'batch_size': b, 'seq_len': s_, 'context_len': c}, dev)
    ref = R.run(**inp)
    out = mod.run(**inp)
    atol = tolmap[sh]
    ok = True
    msg = []
    for i, (o, r) in enumerate(zip(out, ref)):
        err = (o.float() - r).abs()
        frac = (err > atol + 1.19e-7 * r.abs()).float().mean().item()
        msg.append(f"o{i} max {err.max().item():.4f} bad {frac*100:.3f}%")
        if 1 - frac < 0.99:
            ok = False
    tr = bench(lambda: R.run(**inp))
    tm = bench(lambda: mod.run(**inp))
    tot += math.log(tr / tm)
    print(f"{'OK ' if ok else 'BAD'} b{b} s{s_} c{c}: ref {tr:.3f} mine {tm:.3f} = {tr/tm:.2f}x   " + " | ".join(msg))
print(f"geomean {math.exp(tot/len(shapes)):.3f}x")
