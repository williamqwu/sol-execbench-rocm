import torch, time, sys, importlib
import reference

sys.path.insert(0, '.')
import kernel
importlib.reload(kernel)

dev = torch.device('cuda:0')
AX = dict(input_seq_len=3000, output_seq_len=1500, num_mel_bins=80,
          d_model=5120, encoder_ffn_dim=20480, num_heads=20, head_dim=256)

def bench(fn, args, iters=3):
    for _ in range(2): fn(*args)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): o = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/iters*1e3, o

for B in [int(x) for x in sys.argv[1:]] or [3, 16]:
    torch.manual_seed(0)
    ax = dict(AX); ax['batch_size'] = B
    inp = reference.get_inputs(ax, dev)
    args = list(inp.values())
    tr, ref = bench(reference.run, args)
    tk, out = bench(kernel.run, args)
    err = (out.float()-ref.float()).abs()
    rel = err / (ref.float().abs() + 1e-6)
    ok = (err <= 2.48) | (rel <= 0.0078125)
    print(f"B={B:3d} ref={tr:8.2f}ms mine={tk:8.2f}ms speedup={tr/tk:.2f}x "
          f"maxerr={err.max().item():.4f} matched={ok.float().mean().item():.5f}")
    del inp, args, ref, out, err, rel, ok
    torch.cuda.empty_cache()
