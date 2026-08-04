import torch, time, math, sys
sys.path.insert(0, '/var/tmp/solbench/agent/pilot8/L2__069_joint_transformer_block_residual_path')
import reference as R

dev = torch.device('cuda')
torch.cuda.set_device(0)
print(torch.cuda.get_device_name(0))


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


shapes = [(1, 1024, 77), (1, 8192, 77), (32, 256, 77), (64, 128, 77), (4, 2048, 77)]
for b, s_, c in shapes:
    inp = R.get_inputs({'batch_size': b, 'seq_len': s_, 'context_len': c}, dev)
    t = bench(lambda: R.run(**inp))
    o = R.run(**inp)
    print(f"b={b} s={s_} c={c}: {t:.4f} ms  absmax hs={o[1].abs().max().item():.3f} ehs={o[0].abs().max().item():.3f} std={o[1].std().item():.3f}")
