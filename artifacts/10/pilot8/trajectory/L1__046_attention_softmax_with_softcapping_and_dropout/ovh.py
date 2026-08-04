import torch, triton, triton.language as tl


@triton.jit
def _nop(X, Y):
    pass


x = torch.randn(8, 8, 128, 128, device='cuda', dtype=torch.bfloat16)
y = torch.empty_like(x)


def t(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


print("empty_like        %.2f us" % t(lambda: torch.empty_like(x)))
print("nop triton launch %.2f us" % t(lambda: _nop[(1024,)](x, y)))
print("empty+nop         %.2f us" % t(lambda: _nop[(1024,)](x, torch.empty_like(x))))
print("torch.tanh        %.2f us" % t(lambda: torch.tanh(x)))
print("x.contiguous()    %.2f us" % t(lambda: x.contiguous()))
