import sys, torch, time, triton, triton.language as tl
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk
dev = 'cuda'


@triton.jit
def noop(X):
    pass


def bench(fn, n=200, w=50):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


x = torch.zeros(1, device=dev)
print("noop launch us:", bench(lambda: noop[(1,)](x)) * 1e3)
print("4 noop launches us:", bench(lambda: [noop[(1,)](x) for _ in range(4)]) * 1e3)

# weight quant tuning
H, I = 3584, 2048
guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16)
dw = torch.randn(H, I, device=dev, dtype=torch.bfloat16)
for nw in [1, 2, 4, 8, 16]:
    q = torch.empty_like(guw, dtype=torch.float8_e4m3fn)
    s = torch.empty((2 * I // 128, H // 128), dtype=torch.float32, device=dev)
    t = bench(lambda: tk._quant_w_128x128[(2 * I // 128, H // 128)](
        guw, q, s, 2 * I, H, guw.stride(0), s.stride(0), num_warps=nw))
    print(f"wq gate_up nw={nw}: {t*1e3:.1f} us  ({(2*I*H*3)/(t/1e3)/1e12:.2f} TB/s)")

M = 4096
hs = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
for bm in [8, 16, 32, 64, 128]:
    for nw in [1, 2, 4, 8]:
        q = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
        s = torch.empty((M, H // 128), dtype=torch.float32, device=dev)
        t = bench(lambda: tk._quant_act_1x128[(triton.cdiv(M, bm), H // 128)](
            hs, q, s, M, H, hs.stride(0), q.stride(0), s.stride(0),
            BLOCK_M=bm, num_warps=nw))
        print(f"qa bm={bm} nw={nw}: {t*1e3:.1f} us ({(M*H*3)/(t/1e3)/1e12:.2f} TB/s)")
