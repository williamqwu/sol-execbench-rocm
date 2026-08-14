import json, time, sys
import torch

DEV = 'cuda:0'
SHAPES = []
for line in open('workload.jsonl'):
    w = json.loads(line)
    SHAPES.append((w['axes']['batch_size'], 256, w['axes']['seqlen'], w['tolerance']['max_atol']))


def ref(x):
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen
    x_f32 = x.to(torch.float32)
    x_freq = torch.fft.rfft(x_f32, n=fft_size)
    x_freq = x_freq / fft_size
    return x_freq.real.contiguous(), x_freq.imag.contiguous()


def bench(fn, x, iters=20, warmup=5):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2] * 1e3


if __name__ == '__main__':
    mod = None
    if len(sys.argv) > 1:
        import importlib
        mod = importlib.import_module(sys.argv[1])
    print(f"{'B':>4} {'S':>7} {'ref ms':>9} {'mine ms':>9} {'speedup':>8}  {'maxerr':>10} {'atol':>10} ok")
    for (B, C, S, atol) in SHAPES:
        torch.manual_seed(1234)
        x = torch.randn(B, C, S, device=DEV, dtype=torch.float32)
        tr = bench(lambda t: ref(t), x)
        line = f"{B:>4} {S:>7} {tr:9.3f}"
        if mod is not None:
            try:
                r0, i0 = ref(x)
                r1, i1 = mod.run(x)
                assert r1.shape == r0.shape, (r1.shape, r0.shape)
                assert r1.dtype == r0.dtype
                er = (r1 - r0).abs().max().item()
                ei = (i1 - i0).abs().max().item()
                e = max(er, ei)
                # replicate matched-ratio style check
                ok_r = (torch.isclose(r1, r0, atol=atol, rtol=1.1920928955078125e-07).float().mean().item())
                ok_i = (torch.isclose(i1, i0, atol=atol, rtol=1.1920928955078125e-07).float().mean().item())
                ratio = min(ok_r, ok_i)
                tm = bench(lambda t: mod.run(t), x)
                line += f" {tm:9.3f} {tr/tm:8.2f}x  {e:10.3e} {atol:10.3e} {'OK ' if (e<=atol or ratio>=0.99) else 'BAD'} ratio={ratio:.4f}"
            except Exception as ex:
                line += f"  EXC {type(ex).__name__}: {ex}"
        print(line, flush=True)
