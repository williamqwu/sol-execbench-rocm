import torch, triton, triton.language as tl, time
from triton.runtime import driver
import kernel as K


def bench(fn, iters=4000, warmup=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


f = torch.randn(4, 512, 64, device='cuda')
cos = torch.empty(4, 512, 128, dtype=torch.bfloat16, device='cuda')
sin = torch.empty(4, 512, 128, dtype=torch.bfloat16, device='cuda')

kf = K._cos_sin_emb_kernel
kf[(2048,)](f, cos, sin, 2048, 1.0, D=64, BLOCK_R=1, num_warps=1, num_stages=1)
torch.cuda.synchronize()
print('JIT launch          %.4f' % bench(lambda: kf[(2048,)](f, cos, sin, 2048, 1.0, D=64, BLOCK_R=1, num_warps=1, num_stages=1)))

ck = None
for dc in kf.device_caches.values():
    for cache in dc:
        if isinstance(cache, dict) and cache:
            for v in cache.values():
                if hasattr(v, 'packed_metadata'):
                    ck = v
print('compiled:', getattr(ck, 'name', None))
print('signature:', ck.src.signature)
print('constants:', ck.src.constants)
print('global_scratch_size:', getattr(ck.metadata, 'global_scratch_size', 'NA'),
      'align:', getattr(ck.metadata, 'global_scratch_align', 'NA'))
print('profile_scratch_size:', getattr(ck.metadata, 'profile_scratch_size', 'NA'))

ck._init_handles()
dev = driver.active.get_current_device()
stream = driver.active.get_current_stream(dev)

# CompiledKernel.__getitem__ path (skips JIT specialization)
runner = ck[(2048, 1, 1)]
runner(f, cos, sin, 2048, 1.0, 64, 1)
torch.cuda.synchronize()
print('ck[grid] runner     %.4f' % bench(lambda: runner(f, cos, sin, 2048, 1.0, 64, 1)))
print('ck[grid] +stream    %.4f' % bench(lambda: runner(f, cos, sin, 2048, 1.0, 64, 1, stream=stream)))

# raw launcher: try with trailing global scratch arg
run_ = ck.run
func = ck.function
pm = ck.packed_metadata
for extra in ([64, 1],):
    try:
        run_(2048, 1, 1, stream, func, pm, None, None, None, f, cos, sin, 2048, 1.0, *extra)
        torch.cuda.synchronize()
        print('RAW OK with extra:', extra)
        args = (2048, 1, 1, stream, func, pm, None, None, None, f, cos, sin, 2048, 1.0, *extra)
        print('raw launch          %.4f' % bench(lambda: run_(*args)))
        pf = f.data_ptr(); pc = cos.data_ptr(); ps = sin.data_ptr()
        args2 = (2048, 1, 1, stream, func, pm, None, None, None, pf, pc, ps, 2048, 1.0, *extra)
        print('raw + int ptrs      %.4f' % bench(lambda: run_(*args2)))
        break
    except TypeError as e:
        print('extra', extra, '->', e)

print()
print('2x empty            %.4f' % bench(lambda: (torch.empty(4, 512, 128, dtype=torch.bfloat16, device='cuda'), torch.empty(4, 512, 128, dtype=torch.bfloat16, device='cuda'))))
print('1x empty + index    %.4f' % bench(lambda: (lambda t: (t[0], t[1]))(torch.empty(2, 4, 512, 128, dtype=torch.bfloat16, device='cuda'))))
print('1x empty + unbind   %.4f' % bench(lambda: torch.empty(2, 4, 512, 128, dtype=torch.bfloat16, device='cuda').unbind(0)))
print('empty flat+2 view   %.4f' % bench(lambda: (lambda t: (t[0].view(4, 512, 128), t[1].view(4, 512, 128)))(torch.empty(2, 4 * 512 * 128, dtype=torch.bfloat16, device='cuda'))))
