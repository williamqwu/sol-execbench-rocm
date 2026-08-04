import json, time, torch, sys

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"

H, D, N = 128, 128, 7168
K = H * D


def ref(attn_output, w):
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    x = attn_output.transpose(1, 2).reshape(bsz, seq_len, num_heads * v_head_dim)
    return torch.matmul(x, w.t())


def copy_only(attn_output, w):
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    return attn_output.transpose(1, 2).reshape(bsz, seq_len, num_heads * v_head_dim)


def mm_only(x2d, w):
    return torch.matmul(x2d, w.t())


def bench(fn, args, iters=20, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


shapes = []
for l in open("workload.jsonl"):
    a = json.loads(l)["axes"]
    shapes.append((a["batch_size"], a["seq_len"]))

w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
print(f"{'B':>3} {'S':>5} {'M':>6} {'ref ms':>9} {'copy ms':>9} {'mm ms':>9} {'TFLOPs':>8} {'GB/s':>8}")
for (B, S) in shapes:
    a = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    M = B * S
    t_ref = bench(ref, (a, w))
    t_copy = bench(copy_only, (a, w))
    x2d = copy_only(a, w).reshape(M, K)
    t_mm = bench(mm_only, (x2d, w))
    flops = 2 * M * K * N
    byts = (M * K + N * K + M * N) * 2
    print(f"{B:>3} {S:>5} {M:>6} {t_ref:9.3f} {t_copy:9.3f} {t_mm:9.3f} {flops/t_mm*1e-9:8.1f} {byts/t_ref*1e-6:8.1f}")
    del a, x2d
    torch.cuda.empty_cache()
