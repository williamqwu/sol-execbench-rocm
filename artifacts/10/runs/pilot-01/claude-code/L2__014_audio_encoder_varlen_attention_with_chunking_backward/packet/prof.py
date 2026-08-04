import json, sys, time
import torch
sys.path.insert(0, ".")
import reference, kernel

CONST = {"d_model": 1280, "num_heads": 20, "head_dim": 64}
ORDER = ["grad_output", "hidden_states", "query_states", "key_states", "value_states",
         "cu_seqlens", "q_weight", "k_weight", "v_weight", "out_weight"]
dev = torch.device("cuda:0")
wls = [json.loads(l) for l in open("workload.jsonl")]


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6  # us


for i in [int(x) for x in (sys.argv[1:] or ["13", "8", "9", "0"])]:
    ax = dict(CONST); ax.update(wls[i]["axes"])
    torch.manual_seed(5 + i)
    inp = reference.get_inputs(ax, dev)
    args = [inp[k] for k in ORDER]
    tot = bench(lambda: kernel.run(*args))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            kernel.run(*args)
        torch.cuda.synchronize()
    evs = {}
    for e in prof.key_averages():
        if e.device_time_total > 0:
            evs[e.key] = e.device_time_total / 20
    print(f"=== wl{i} N={ax['total_seq_len']} nc={ax['num_chunks']}  total(wall)={tot:.1f}us "
          f"gpu_sum={sum(evs.values()):.1f}us")
    for kk, vv in sorted(evs.items(), key=lambda x: -x[1])[:12]:
        print(f"    {vv:8.1f}us  {kk[:78]}")
