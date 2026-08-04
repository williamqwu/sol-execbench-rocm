import torch, math, sys
sys.path.insert(0, '/var/tmp/solbench/agent/pilot8/L2__069_joint_transformer_block_residual_path')
dev = torch.device('cuda')


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


M, K, N = 8192, 1536, 6144
a32 = torch.randn(M, K, device=dev)
b32 = torch.randn(N, K, device=dev)
a16 = a32.half(); b16 = b32.half()
ab16 = a32.bfloat16(); bb16 = b32.bfloat16()

flops = 2 * M * K * N
print("fp32 :", f"{bench(lambda: a32 @ b32.t()):.3f} ms", f"{flops/bench(lambda: a32 @ b32.t())/1e9:.0f} TF")
torch.backends.cuda.matmul.allow_tf32 = True
print("tf32 :", f"{bench(lambda: a32 @ b32.t()):.3f} ms", f"{flops/bench(lambda: a32 @ b32.t())/1e9:.0f} TF")
torch.backends.cuda.matmul.allow_tf32 = False
print("fp16 :", f"{bench(lambda: a16 @ b16.t()):.3f} ms", f"{flops/bench(lambda: a16 @ b16.t())/1e9:.0f} TF")
print("bf16 :", f"{bench(lambda: ab16 @ bb16.t()):.3f} ms")

# accuracy
ref = a32 @ b32.t()
o16 = (a16 @ b16.t()).float()
print("fp16 relerr", ((o16 - ref).abs().max() / ref.abs().max()).item(), (o16-ref).abs().mean().item(), ref.abs().mean().item())

# cast cost
print("cast weight 6144x1536:", f"{bench(lambda: b32.half()):.4f} ms")

# skinny
for (M2, K2, N2) in [(77, 1152, 6144), (77, 1536, 1536), (308, 1152, 6144), (2048, 1536, 4608)]:
    x32 = torch.randn(M2, K2, device=dev); w32 = torch.randn(N2, K2, device=dev)
    x16 = x32.half(); w16 = w32.half()
    t32 = bench(lambda: x32 @ w32.t()); t16 = bench(lambda: x16 @ w16.t())
    print(f"M{M2} K{K2} N{N2}: fp32 {t32*1000:.1f}us  fp16 {t16*1000:.1f}us  castw {bench(lambda: w32.half())*1000:.1f}us")
