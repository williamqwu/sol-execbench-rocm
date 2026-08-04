import torch, time
import kernel2 as K


def bench(fn, iters=5000, warmup=1500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


f = torch.randn(4, 512, 64, device='cuda')
K.run(f, 1.0)
torch.cuda.synchronize()
run_, func, pm = K._LAUNCH[64]
dev = 'cuda'
sh = (4, 512)

print('full run             %.4f' % bench(lambda: K.run(f, 1.0)))

# component costs
print()
print('empty (2,4,512,128)  %.4f' % bench(lambda: torch.empty((2, 4, 512, 128), dtype=torch.bfloat16, device=dev)))
o = torch.empty((2, 4, 512, 128), dtype=torch.bfloat16, device=dev)
print('  + o[0],o[1]        %.4f' % bench(lambda: (lambda t: (t[0], t[1]))(torch.empty((2, 4, 512, 128), dtype=torch.bfloat16, device=dev))))
print('  index only         %.4f' % bench(lambda: (o[0], o[1])))
print('  unbind only        %.4f' % bench(lambda: o.unbind(0)))
print('  select only        %.4f' % bench(lambda: (torch.select(o, 0, 0), torch.select(o, 0, 1))))
print()
print('current_stream()     %.4f' % bench(lambda: torch.cuda.current_stream().cuda_stream))
import triton
from triton.runtime import driver
dr = driver.active
print('driver get_stream    %.4f' % bench(lambda: dr.get_current_stream(0)))
st = torch.cuda.current_stream().cuda_stream
print()
print('is_contiguous        %.4f' % bench(lambda: f.is_contiguous()))
print('numel                %.4f' % bench(lambda: f.numel()))
print('shape slice          %.4f' % bench(lambda: f.shape[:-1] + (128,)))
print('float()              %.4f' % bench(lambda: float(1.0)))
print()
c = o[0]; s_ = o[1]
print('bare raw launch      %.4f' % bench(lambda: run_(256, 1, 1, st, func, pm, None, None, None, f, c, s_, 2048, 1.0, 64, 8)))
