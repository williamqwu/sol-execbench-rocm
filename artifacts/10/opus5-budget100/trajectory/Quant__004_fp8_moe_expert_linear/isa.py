import sys, torch, triton, re, collections
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk5 as tk
dev = 'cuda'
H, I = 3584, 2048
M = 4096
aq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
asc = torch.empty(M, H // 128, device=dev)
wq = torch.empty((2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
wsc = torch.empty(2 * I // 128, H // 128, device=dev)
gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
gs = torch.empty((M, I // 128), device=dev)
k = tk._gemm1[(triton.cdiv(M, 128), I // 128)](
    aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0), wq.stride(0),
    wsc.stride(0), gq.stride(0), gs.stride(0), BLOCK_M=128, NUM_K=H // 128,
    num_warps=8, num_stages=2, waves_per_eu=2)
asm = k.asm['amdgcn']
open(D + "/g1.s", "w").write(asm)
# isolate the main loop body
c = collections.Counter(l.strip().split()[0] for l in asm.splitlines()
                        if l.strip() and l.strip()[0] not in ';.$/' and
                        not l.strip().endswith(':'))
for op, n in c.most_common(30):
    print(f"{n:6d} {op}")
print("VGPRs:", re.findall(r'vgpr_count.*|agpr_count.*|occupancy.*|lds_size.*', asm)[:6])
